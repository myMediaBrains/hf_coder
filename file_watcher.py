# file_watcher.py (hf_coder)
#
# VS Code에서 파일을 저장하면(디스크 변경 이벤트) 감지해서, git diff를 뽑아
# LLM에게 자동으로 리뷰를 맡기고, 결과를 CodeChatMessage로 저장하는 모듈.
#
# 안전장치:
# - 디바운스(기본 2.5초): 타이핑 중 여러 번 저장돼도 마지막 것만 반응
# - 자기-쓰기 가드: apply_patch()로 우리가 직접 쓴 내용과 해시가 같으면
#   재분석을 건너뛴다 (안 그러면 "적용 → 다시 감지 → 다시 리뷰 → ..." 무한루프)
# - 감시는 기본 꺼짐 - /watcher/start를 명시적으로 호출해야 켜짐
# - 한 번에 하나의 세션만 구독 가능 (로컬 개인용 앱이라 단순하게 유지)
import difflib
import hashlib
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import model_router
import patch_utils

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 2.5


class _State:
    """감시 상태를 하나의 전역 객체로 관리 (개인용 로컬 앱이라 싱글턴으로 충분)."""
    observer: Optional[Observer] = None
    watching: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    last_event_at: Optional[float] = None
    # 자기-쓰기 가드: {절대경로: 방금 우리가 쓴 내용의 sha256}
    recent_self_writes: dict[str, str] = {}
    # 디바운스용 타이머: {절대경로: threading.Timer}
    pending_timers: dict[str, threading.Timer] = {}
    lock = threading.Lock()


_state = _State()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def mark_self_write(abs_path: Path, content: str):
    """apply_patch()로 우리가 직접 쓴 직후 호출 - 다음 감시 이벤트에서 재분석을 건너뛰게 함."""
    with _state.lock:
        _state.recent_self_writes[str(abs_path)] = _content_hash(content)


def get_status() -> dict:
    with _state.lock:
        return {
            "watching": _state.watching,
            "session_id": _state.session_id,
            "last_event_at": _state.last_event_at,
        }


def _run_git(root: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", "-C", str(root)] + args,
        capture_output=True, text=True, timeout=15,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _get_diff_for_file(root: Path, rel_path: str, new_content: str) -> str:
    """마지막 커밋 대비 이 파일의 diff. 새 파일(untracked)이면 빈 문자열 -> 전체를
    추가된 것으로 간주해 difflib로 직접 만든다."""
    code, out, err = _run_git(root, ["diff", "--", rel_path])
    if out.strip():
        return out

    # git diff가 비어있으면: 커밋된 적 없는 새 파일이거나, 이미 스테이징된 상태일 수 있음.
    # 둘 다 처리: 마지막 커밋 시점 내용(git show HEAD:path)과 비교, 없으면 빈 문자열과 비교.
    code2, old_content, _ = _run_git(root, ["show", f"HEAD:{rel_path}"])
    if code2 != 0:
        old_content = ""
    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
    )
    return "".join(diff_lines)


def _analyze_and_store(root: Path, abs_path: Path, save_message_fn):
    """실제 분석 - 디바운스 타이머가 만료되면 백그라운드 스레드에서 호출됨."""
    rel_path = str(abs_path.relative_to(root))

    try:
        new_content = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"[file_watcher] 파일 읽기 실패 ({rel_path}): {e}")
        return

    # 자기-쓰기 가드: 방금 우리가 이 내용 그대로 쓴 거면 재분석 건너뜀
    with _state.lock:
        recent_hash = _state.recent_self_writes.get(str(abs_path))
    if recent_hash and recent_hash == _content_hash(new_content):
        logger.info(f"[file_watcher] 자기-쓰기로 판단, 재분석 건너뜀: {rel_path}")
        return

    diff_text = _get_diff_for_file(root, rel_path, new_content)
    if not diff_text.strip():
        logger.info(f"[file_watcher] 변경 없음 (diff 비어있음): {rel_path}")
        return

    system_prompt = (
        "You are reviewing a code change that was just saved in the developer's local "
        "editor (VS Code). You will be given the git diff and the full current content "
        "of the file. Write a short review in Korean (2-5 sentences): what changed, "
        "and any concerns (bugs, style issues, missed edge cases).\n\n"
        f"If — and only if — you believe the file needs further revision, propose it.\n{patch_utils.PATCH_FORMAT_INSTRUCTIONS}\n"
        "If the change looks fine as-is, do not include any SEARCH/REPLACE block."
    )
    user_prompt = f"파일: {rel_path}\n\ndiff:\n```diff\n{diff_text}\n```\n\n현재 전체 내용:\n```\n{new_content}\n```"

    try:
        response_text = model_router.chat(
            task="review_diff",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        logger.error(f"[file_watcher] LLM 리뷰 호출 실패: {e}")
        response_text = f"⚠️ 자동 리뷰 중 오류가 발생했습니다: {e}"

    review_comment, edits = patch_utils.extract_edits(response_text)
    if not review_comment:
        review_comment = "(수정 제안을 아래에서 확인해주세요)" if edits else "변경사항을 확인했습니다."

    with _state.lock:
        _state.last_event_at = time.time()

    save_message_fn(
        rel_path=rel_path,
        review_comment=review_comment,
        edits=edits,
        model_used=model_router.model_for_task("review_diff"),
    )


class _SaveEventHandler(FileSystemEventHandler):
    def __init__(self, root: Path, excluded_dirs: set, extensions: set, save_message_fn):
        self.root = root
        self.excluded_dirs = excluded_dirs
        self.extensions = extensions
        self.save_message_fn = save_message_fn

    def _should_track(self, path_str: str) -> bool:
        p = Path(path_str)
        if p.suffix not in self.extensions:
            return False
        if any(part in self.excluded_dirs for part in p.parts):
            return False
        return True

    def on_modified(self, event):
        if event.is_directory or not self._should_track(event.src_path):
            return
        self._debounce(Path(event.src_path))

    def on_created(self, event):
        if event.is_directory or not self._should_track(event.src_path):
            return
        self._debounce(Path(event.src_path))

    def _debounce(self, abs_path: Path):
        key = str(abs_path)
        with _state.lock:
            existing = _state.pending_timers.get(key)
            if existing:
                existing.cancel()
            timer = threading.Timer(
                DEBOUNCE_SECONDS,
                _analyze_and_store,
                args=(self.root, abs_path, self.save_message_fn),
            )
            timer.daemon = True
            _state.pending_timers[key] = timer
            timer.start()


def start(root: Path, excluded_dirs: set, extensions: set, session_id: str,
          user_id: Optional[str], save_message_fn) -> bool:
    with _state.lock:
        if _state.watching:
            return False
        handler = _SaveEventHandler(root, excluded_dirs, extensions, save_message_fn)
        observer = Observer()
        observer.schedule(handler, str(root), recursive=True)
        observer.start()
        _state.observer = observer
        _state.watching = True
        _state.session_id = session_id
        _state.user_id = user_id
        _state.last_event_at = None
    logger.info(f"[file_watcher] 감시 시작: {root} (session={session_id})")
    return True


def stop() -> bool:
    with _state.lock:
        if not _state.watching or not _state.observer:
            return False
        _state.observer.stop()
        _state.observer.join(timeout=5)
        _state.observer = None
        _state.watching = False
    logger.info("[file_watcher] 감시 중지")
    return True
    