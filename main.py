# main.py (hf_coder)
# hf_crawler의 /codeanalysis/* 기능만 독립시킨 서비스. 별도 포트(:8100 기본)로
# 뜨고, 별도 DB(hf_coder.db)를 쓴다. hf_crawler와는 HTTP로만 통신한다.
import os
import json
import logging
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import ollama
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from dotenv import load_dotenv

import model_router
import file_watcher
import github_client
import patch_utils
import code_map
import lint_runner
import sandbox
import vector_search
import agent_tools
from database import engine, get_session, create_db_and_tables
from models import CodeChatMessage, CodeEmbedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan, title="hf_coder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경 - hf-frontend(별도 포트)에서 직접 호출하므로 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 코드베이스 파일트리 / 파일 읽기
# VS Code로 편집 중인 이 프로젝트 폴더를 파일시스템으로 직접 읽는다.
# 원격 연결이 아니라 같은 맥북 안에서 경로만 공유하는 것 - VS Code에서
# 저장(⌘S)하면 그 즉시 아래 API가 최신 내용을 읽는다.
# ============================================================

CODE_PROJECT_ROOT = Path(
    os.getenv("CODE_PROJECT_ROOT", "/Users/catchhub/AI/Code/WebCrawler/hf_crawler")
).resolve()

CODE_FILE_EXTENSIONS = {".py", ".jsx", ".js", ".ts", ".tsx", ".css", ".md", ".json", ".html"}
CODE_EXCLUDED_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build", ".vite", "uploads"
}


def _build_code_file_tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    paths = []
    for p in root.rglob("*"):
        if any(part in CODE_EXCLUDED_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix in CODE_FILE_EXTENSIONS:
            paths.append(str(p.relative_to(root)))
    return sorted(paths)


def _resolve_code_path(rel_path: str) -> Path:
    """경로 탈출(../) 방지 - 반드시 CODE_PROJECT_ROOT 하위여야 함."""
    target = (CODE_PROJECT_ROOT / rel_path).resolve()
    if not str(target).startswith(str(CODE_PROJECT_ROOT)):
        raise HTTPException(status_code=400, detail="허용되지 않은 경로입니다.")
    return target


agent_tools.configure(
    build_file_tree_fn=lambda: _build_code_file_tree(CODE_PROJECT_ROOT),
    resolve_path_fn=_resolve_code_path,
    project_root=CODE_PROJECT_ROOT,
)


@app.get("/codeanalysis/files")
def list_code_files():
    return {
        "root": str(CODE_PROJECT_ROOT),
        "files": _build_code_file_tree(CODE_PROJECT_ROOT),
    }


@app.get("/codeanalysis/repo-map")
def get_repo_map(force: bool = Query(False)):
    """Tree-sitter 기반 함수/클래스 요약. tree-sitter 미설치 시 available=False로 표시."""
    if not code_map.is_available():
        return {"available": False, "map_text": ""}
    file_list = _build_code_file_tree(CODE_PROJECT_ROOT)
    return {"available": True, "map_text": code_map.get_repo_map(CODE_PROJECT_ROOT, file_list, force=force)}


@app.get("/codeanalysis/file")
def read_code_file(path: str = Query(...)):
    target = _resolve_code_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 실패: {e}")
    return {"path": path, "content": content}


# ============================================================
# 코딩분석 채팅
# ============================================================

class CodeChatRequest(BaseModel):
    session_id: str
    message: str
    included_files: list[str] = []
    user_id: Optional[str] = None
    architect: bool = False  # 2026-08-13: 켜면 "설계 → 패치 작성" 2단계로 진행
    fast_mode: bool = False  # 2026-08-13: 켜면 qwen2.5-coder:7b(빠름, 메모리 절약)로 답변


def _build_project_index_text() -> str:
    """전체 파일목록 + (가능하면) repo map. code_analysis/architect 양쪽에서 재사용."""
    file_list = _build_code_file_tree(CODE_PROJECT_ROOT)
    repo_map_text = code_map.get_repo_map(CODE_PROJECT_ROOT, file_list)
    index_parts = [f"Full project file list (every file that exists in this project):\n{chr(10).join(file_list)}"]
    if repo_map_text:
        index_parts.append(
            "Additional detail - functions/classes per code file "
            "(only .py/.js/.jsx; files not listed here still exist, just have no "
            "functions/classes detected, e.g. simple scripts):\n" + repo_map_text
        )
    return "\n\n".join(index_parts)


def _has_repetition_loop(text: str, window: int = 150, min_repeats: int = 3) -> bool:
    """
    2026-08-13(신규): 로컬 소형 모델(특히 14b 빠른 모드)이 num_predict 끝까지
    같은 문단을 무한 반복하는 증상이 실제로 있었다. 샘플링 파라미터(repeat_penalty
    등)로 예방을 강화했지만 100% 막긴 어려워서, 스트리밍 중 최근 구간이 그대로
    여러 번 반복되면 서버가 직접 감지해서 생성을 조기 중단시키는 안전장치.
    """
    if len(text) < window * min_repeats:
        return False
    tail = text[-window:]
    return text.count(tail) >= min_repeats


def _build_context_blocks(included_files: list[str]) -> list[str]:
    context_blocks = []
    for rel_path in included_files:
        try:
            target = _resolve_code_path(rel_path)
            content = target.read_text(encoding="utf-8", errors="ignore")
            context_blocks.append(f"### {rel_path}\n```\n{content}\n```")
        except Exception:
            continue
    return context_blocks


@app.post("/codeanalysis/chat/stream")
async def code_analysis_chat_stream(request: CodeChatRequest, session: Session = Depends(get_session)):
    history_rows = session.exec(
        select(CodeChatMessage)
        .where(CodeChatMessage.session_id == request.session_id)
        .order_by(CodeChatMessage.created_at)
    ).all()

    context_blocks = _build_context_blocks(request.included_files)
    index_text = _build_project_index_text()

    base_intro = (
        "You are an expert coding assistant analyzing a local project "
        "(FastAPI backend `hf_crawler` + React/Vite frontend `hf-frontend`). "
        "Answer in Korean unless the user writes in another language.\n\n"
        f"{index_text}\n\n"
        + ("Included file contents:\n\n" + "\n\n".join(context_blocks) if context_blocks
           else "No file contents were explicitly included for this question — "
                "answer from the structure above and conversation context, and ask the user "
                "to check relevant files in the left sidebar if you need to see actual code.")
        + "\n\nCRITICAL: Only state something as fact if you can point to the exact "
          "code/text you were shown that proves it (quote the specific line or "
          "function name). Never state a claim you haven't verified against the "
          "actual content given to you - if you're not sure, say so explicitly "
          "('확인하지 못했습니다', '~일 수도 있으나 확실하지 않습니다') instead of "
          "guessing. This applies especially to naming conventions, whether code is "
          "'unused', and claims about parts of a file you weren't shown."
    )

    session.add(CodeChatMessage(
        session_id=request.session_id,
        user_id=request.user_id,
        role="user",
        content=request.message,
        included_files=json.dumps(request.included_files, ensure_ascii=False),
    ))
    session.commit()

    history_messages = [{"role": r.role, "content": r.content} for r in history_rows]

    async def event_generator() -> AsyncGenerator[str, None]:
        full_display_text = ""  # 화면에 보여줄 전체 텍스트 (설계 단계 포함)
        editor_text = ""        # 실제 SEARCH/REPLACE 추출 대상 (작성 단계만)
        model_used_final = model_router.model_for_task(
            "code_analysis_light" if request.fast_mode else "code_analysis"
        )

        try:
            if request.architect:
                # ---------- 1단계: 설계 (형식 신경 안 쓰고 자유롭게 계획만) ----------
                architect_prompt = (
                    base_intro
                    + "\n\nThe user wants a code change. First, THINK THROUGH the plan in Korean: "
                      "which file(s) need to change, what exactly should change, and why. "
                      "Do NOT write any SEARCH/REPLACE blocks or code diffs yet - just the plan in prose."
                )
                architect_messages = (
                    [{"role": "system", "content": architect_prompt}]
                    + history_messages
                    + [{"role": "user", "content": request.message}]
                )

                phase_marker = "🏗️ **설계 단계**\n\n"
                full_display_text += phase_marker
                yield f"data: {json.dumps({'delta': phase_marker}, ensure_ascii=False)}\n\n"

                architect_text = ""
                async for chunk in model_router.achat_stream(task="architect_plan", messages=architect_messages):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        architect_text += delta
                        full_display_text += delta
                        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                        if _has_repetition_loop(architect_text):
                            note = "\n\n⚠️ 반복 패턴이 감지되어 생성을 중단했습니다."
                            architect_text += note
                            full_display_text += note
                            yield f"data: {json.dumps({'delta': note}, ensure_ascii=False)}\n\n"
                            break

                # ---------- 2단계: 작성 (계획을 그대로 패치로 옮기기만) ----------
                transition = "\n\n---\n\n✍️ **패치 작성 단계**\n\n"
                full_display_text += transition
                yield f"data: {json.dumps({'delta': transition}, ensure_ascii=False)}\n\n"

                editor_prompt = (
                    base_intro
                    + "\n\nAn architect has already planned this change:\n\n"
                    + f"### Plan\n{architect_text}\n\n"
                    + "Your job now is ONLY to implement this plan exactly as specified - do not "
                      "reconsider the approach, just translate the plan into concrete edits.\n"
                    + patch_utils.PATCH_FORMAT_INSTRUCTIONS
                )
                editor_messages = (
                    [{"role": "system", "content": editor_prompt}]
                    + history_messages
                    + [{"role": "user", "content": request.message}]
                )

                async for chunk in model_router.achat_stream(task="code_analysis", messages=editor_messages):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        editor_text += delta
                        full_display_text += delta
                        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                        if _has_repetition_loop(editor_text):
                            note = "\n\n⚠️ 반복 패턴이 감지되어 생성을 중단했습니다."
                            editor_text += note
                            full_display_text += note
                            yield f"data: {json.dumps({'delta': note}, ensure_ascii=False)}\n\n"
                            break

                model_used_final = model_router.model_for_task("architect_plan") + " + " + model_router.model_for_task("code_analysis")

            else:
                # ---------- 기존 단일 단계 (fast_mode에 따라 7b/30b 선택) ----------
                task_name = "code_analysis_light" if request.fast_mode else "code_analysis"
                normal_prompt = (
                    base_intro
                    + f"\n\nIf the user asks you to make a code change, propose it as an edit.\n{patch_utils.PATCH_FORMAT_INSTRUCTIONS}"
                )
                if request.fast_mode:
                    # 2026-08-13(신규): 7b는 "SEARCH/REPLACE 형식을 못 지킴" +
                    # "요청 범위를 벗어난 대규모 리팩토링을 지어내서 제안함"이라는
                    # 두 종류의 실패가 실제로 관찰됐다. 매번 사후에 잡는 대신,
                    # 모델 스스로 "이건 내가 감당 못한다"를 인식하고 정직하게
                    # 손을 떼도록 명시적으로 지침을 준다.
                    normal_prompt += (
                        "\n\nYou are running in a lightweight/fast mode meant only for "
                        "quick lookups and simple single-function edits. If this request "
                        "requires deep understanding of a large file, a broad refactor, "
                        "splitting code into multiple files/classes, or anything you're "
                        "not fully confident about, do NOT attempt it or invent "
                        "placeholder code - instead say clearly in Korean that this "
                        "request needs the full (non-fast) mode and ask the user to turn "
                        "off fast mode. Never fabricate classes, functions, or code that "
                        "isn't actually in the file you were shown."
                    )
                normal_messages = (
                    [{"role": "system", "content": normal_prompt}]
                    + history_messages
                    + [{"role": "user", "content": request.message}]
                )
                async for chunk in model_router.achat_stream(task=task_name, messages=normal_messages):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        editor_text += delta
                        full_display_text += delta
                        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                        if _has_repetition_loop(editor_text):
                            note = "\n\n⚠️ 반복 패턴이 감지되어 생성을 중단했습니다."
                            editor_text += note
                            full_display_text += note
                            yield f"data: {json.dumps({'delta': note}, ensure_ascii=False)}\n\n"
                            break

        except Exception as e:
            logger.error(f"코딩분석 채팅 스트리밍 오류: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return
        finally:
            if full_display_text:
                # edits는 "작성 단계"(editor_text) 결과에서만 추출 - 설계 단계
                # 텍스트에는 애초에 SEARCH/REPLACE가 없게 프롬프트로 막아뒀음.
                clean_editor, edits = patch_utils.extract_edits(editor_text)
                # 화면에 보여줄 내용은 설계+작성 전체에서, 추출된 patch 블록만 제거.
                display_content = full_display_text
                if edits:
                    display_content = full_display_text.replace(editor_text, clean_editor or "(수정 제안을 아래에서 확인해주세요)")
                elif patch_utils.looks_like_failed_patch_attempt(editor_text, edits):
                    # 모델이 "수정 제안하겠다"고 언급은 했는데 실제 SEARCH/REPLACE
                    # 형식을 못 만든 경우 - 조용히 "제안 없음"으로 넘기지 않고 명시적으로 알림.
                    display_content += (
                        "\n\n⚠️ 모델이 수정 제안을 언급했지만 적용 가능한 형식으로 "
                        "만들지 못했습니다. 다시 시도하거나, 빠른 모드를 껐다가 "
                        "다시 물어보시는 걸 권장합니다."
                    )
                with Session(engine) as save_session:
                    save_session.add(CodeChatMessage(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        role="assistant",
                        content=display_content or "(수정 제안을 아래에서 확인해주세요)",
                        model_used=model_used_final,
                        proposed_edits=json.dumps(edits, ensure_ascii=False) if edits else None,
                    ))
                    save_session.commit()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/codeanalysis/history/{session_id}")
def get_code_chat_history(session_id: str, session: Session = Depends(get_session)):
    rows = session.exec(
        select(CodeChatMessage)
        .where(CodeChatMessage.session_id == session_id)
        .order_by(CodeChatMessage.created_at)
    ).all()
    return [
        {
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "created_at": r.created_at.isoformat(),
            "source": r.source,
            "proposed_edits": json.loads(r.proposed_edits) if r.proposed_edits else None,
            "apply_results": json.loads(r.apply_results) if r.apply_results else None,
            "applied": r.applied,
        }
        for r in rows
    ]


@app.get("/codeanalysis/llm-status")
def get_code_llm_status():
    code_model_name = model_router.TIER_MODELS[model_router.ModelTier.CODE]

    loaded_models = []
    try:
        ps_result = ollama.ps()
        for m in ps_result.models:
            loaded_models.append({
                "name": m.model,
                "size_gb": round((m.size or 0) / (1024 ** 3), 2),
                "size_vram_gb": round((m.size_vram or 0) / (1024 ** 3), 2) if m.size_vram else None,
                "expires_at": m.expires_at.isoformat() if m.expires_at else None,
            })
    except Exception as e:
        logger.warning(f"ollama.ps() 조회 실패: {e}")

    code_model_loaded = any(m["name"] == code_model_name for m in loaded_models)

    return {
        "is_generating": model_router.is_generating(),
        "code_model": code_model_name,
        "code_model_loaded": code_model_loaded,
        "loaded_models": loaded_models,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "hf_coder"}


# ============================================================
# VS Code 연동 루프 (신규, 2026-08-13)
# 저장 감지(watchdog) -> git diff 추출 -> LLM 자동 리뷰(+제안) -> 사람이 승인 시
# apply-patch로 반영 -> 준비되면 사람이 직접 commit-push 호출.
# 커밋/푸시는 절대 자동으로 하지 않는다 - 항상 사람이 버튼을 눌러야 한다.
# ============================================================

class WatcherStartRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None


def _save_watcher_message(rel_path: str, review_comment: str,
                           edits: list[dict], model_used: str):
    """file_watcher가 백그라운드 스레드에서 분석을 마치면 이 콜백으로 DB에 저장한다."""
    with Session(engine) as session:
        session.add(CodeChatMessage(
            session_id=file_watcher.get_status()["session_id"],
            user_id=None,
            role="user",
            content=f"[자동 저장 감지] {rel_path} 변경됨",
            source="watcher",
        ))
        session.add(CodeChatMessage(
            session_id=file_watcher.get_status()["session_id"],
            user_id=None,
            role="assistant",
            content=review_comment,
            source="watcher",
            proposed_edits=json.dumps(edits, ensure_ascii=False) if edits else None,
            model_used=model_used,
        ))
        session.commit()


@app.post("/watcher/start")
def start_watcher(request: WatcherStartRequest):
    ok = file_watcher.start(
        root=CODE_PROJECT_ROOT,
        excluded_dirs=CODE_EXCLUDED_DIRS,
        extensions=CODE_FILE_EXTENSIONS,
        session_id=request.session_id,
        user_id=request.user_id,
        save_message_fn=_save_watcher_message,
    )
    if not ok:
        raise HTTPException(status_code=409, detail="이미 감시 중입니다. 먼저 중지해주세요.")
    return {"status": "started", **file_watcher.get_status()}


@app.post("/watcher/stop")
def stop_watcher():
    ok = file_watcher.stop()
    if not ok:
        raise HTTPException(status_code=409, detail="감시 중이 아닙니다.")
    return {"status": "stopped"}


@app.get("/watcher/status")
def watcher_status():
    return file_watcher.get_status()


class ApplyPatchRequest(BaseModel):
    message_id: int


@app.post("/codeanalysis/apply-patch")
def apply_patch(request: ApplyPatchRequest, session: Session = Depends(get_session)):
    msg = session.get(CodeChatMessage, request.message_id)
    if not msg or not msg.proposed_edits:
        raise HTTPException(status_code=404, detail="적용할 제안을 찾을 수 없습니다.")

    edits = json.loads(msg.proposed_edits)
    results = patch_utils.apply_edits(edits, _resolve_code_path, file_watcher.mark_self_write)

    # 2026-08-13: 적용에 성공한 파일에 한해 자동으로 lint를 돌린다 (ruff/eslint).
    # 실패한 edit은 애초에 파일이 안 바뀌었으니 lint 대상이 아니다.
    for r in results:
        if r["status"] == "applied":
            r["lint"] = lint_runner.run_lint(CODE_PROJECT_ROOT, r["path"])

    msg.applied = any(r["status"] == "applied" for r in results)
    msg.apply_results = json.dumps(results, ensure_ascii=False)
    session.add(msg)
    session.commit()

    return {"status": "done", "results": results}


@app.get("/codeanalysis/git/status")
def git_status():
    """
    2026-08-13: 커밋하기 전에 "지금 git add -A를 하면 실제로 뭐가 딸려 들어가는지"
    미리 보여준다. commit-push가 git add -A로 그 순간의 모든 변경사항을 쓸어담다
    보니, 의도치 않은 파일(예: 최근에 저장만 하고 아직 안 커밋한 다른 작업)까지
    같이 커밋되는 사고가 실제로 있었다 - 이 엔드포인트로 사전에 확인할 수 있게 한다.
    """
    proc = subprocess.run(
        ["git", "-C", str(CODE_PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, timeout=15,
    )
    files = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        status_code = line[:2].strip()
        path = line[3:].strip()
        files.append({"status": status_code, "path": path})
    return {"files": files, "clean": len(files) == 0}


class CommitPushRequest(BaseModel):
    message: str
    push: bool = True
    # 2026-08-13: 지정하면 이 파일들만 스테이징(git add <path> ...). 지정 안 하면
    # 예전처럼 git add -A(전체) - 하위호환용 기본값일 뿐, 프론트엔드는 항상
    # git/status로 미리 확인한 파일 목록을 명시적으로 넘기도록 바뀌었다.
    # (다른 무관한 파일이 실수로 같이 커밋되던 사고의 근본 원인이었음)
    paths: Optional[list[str]] = None


@app.post("/codeanalysis/git/commit-push")
def commit_and_push(request: CommitPushRequest):
    """항상 사람이 버튼을 눌러야만 호출되는 엔드포인트 - 절대 자동 트리거되지 않는다."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="커밋 메시지를 입력해주세요.")

    if request.paths is not None and len(request.paths) == 0:
        raise HTTPException(status_code=400, detail="커밋할 파일을 하나 이상 선택해주세요.")
    add_args = ["add"] + (request.paths if request.paths is not None else ["-A"])

    steps = []
    for args in (add_args, ["commit", "-m", request.message]):
        proc = subprocess.run(
            ["git", "-C", str(CODE_PROJECT_ROOT)] + args,
            capture_output=True, text=True, timeout=30,
        )
        steps.append({"cmd": " ".join(args), "returncode": proc.returncode,
                       "stdout": proc.stdout, "stderr": proc.stderr})
        if args[0] == "commit" and proc.returncode != 0 and "nothing to commit" not in proc.stdout:
            return {"status": "error", "steps": steps}

    if request.push:
        proc = subprocess.run(
            ["git", "-C", str(CODE_PROJECT_ROOT), "push"],
            capture_output=True, text=True, timeout=60,
        )
        steps.append({"cmd": "push", "returncode": proc.returncode,
                       "stdout": proc.stdout, "stderr": proc.stderr})
        if proc.returncode != 0:
            return {"status": "error", "steps": steps}

    return {"status": "success", "steps": steps}


@app.get("/codeanalysis/git/last-commit")
def last_commit_info():
    """되돌리기 전 확인용 - 지금 HEAD가 뭔지 사람이 보고 판단할 수 있게."""
    proc = subprocess.run(
        ["git", "-C", str(CODE_PROJECT_ROOT), "log", "-1", "--pretty=%h|%s|%ci"],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise HTTPException(status_code=404, detail="커밋 이력이 없습니다.")
    sha, message, date = proc.stdout.strip().split("|", 2)
    return {"sha": sha, "message": message, "date": date}


class UndoRequest(BaseModel):
    push: bool = True


@app.post("/codeanalysis/git/undo")
def undo_last_commit(request: UndoRequest):
    """
    마지막 커밋을 git revert로 되돌린다. reset이 아니라 revert를 쓰는 이유:
    커밋과 푸시가 항상 같이 일어나는 구조라, undo를 누르는 시점엔 이미 GitHub에
    반영돼 있을 가능성이 높다. reset은 로컬 히스토리를 지워서 원격과 어긋나게
    만들지만, revert는 "반대로 되돌리는 새 커밋"을 하나 더 추가하는 방식이라
    이미 푸시된 상태에서도 안전하다.
    항상 사람이 버튼을 눌러야만 호출되는 엔드포인트 - 절대 자동 트리거되지 않는다.
    """
    steps = []
    proc = subprocess.run(
        ["git", "-C", str(CODE_PROJECT_ROOT), "revert", "--no-edit", "HEAD"],
        capture_output=True, text=True, timeout=30,
    )
    steps.append({"cmd": "revert --no-edit HEAD", "returncode": proc.returncode,
                   "stdout": proc.stdout, "stderr": proc.stderr})
    if proc.returncode != 0:
        # 충돌 등으로 revert 자체가 실패하면 안전하게 중단하고 사람이 직접 보게 함
        subprocess.run(
            ["git", "-C", str(CODE_PROJECT_ROOT), "revert", "--abort"],
            capture_output=True, text=True, timeout=10,
        )
        return {"status": "error", "steps": steps}

    if request.push:
        push_proc = subprocess.run(
            ["git", "-C", str(CODE_PROJECT_ROOT), "push"],
            capture_output=True, text=True, timeout=60,
        )
        steps.append({"cmd": "push", "returncode": push_proc.returncode,
                       "stdout": push_proc.stdout, "stderr": push_proc.stderr})
        if push_proc.returncode != 0:
            return {"status": "error", "steps": steps}

    return {"status": "success", "steps": steps}


# ============================================================
# GitHub 브라우징 (신규, 2026-08-13)
# clone 없이 GitHub REST API(Trees/Contents/Commits)로만 조회하는 읽기전용
# 탐색 기능. 두 가지 용도를 같은 엔드포인트가 공유한다:
#  - "타 GitHub 탐색": 프론트에서 owner/repo를 직접 입력
#  - "내 GitHub 확인": /github/my-repo가 로컬 git remote를 읽어 owner/repo를
#    자동으로 알아내고, 커밋&푸시 직후 반영 여부를 최신 커밋으로 확인
# ============================================================

def _github_error_response(e: github_client.GitHubAPIError):
    raise HTTPException(status_code=400, detail=str(e))


@app.get("/github/my-repo")
def github_my_repo():
    """CODE_PROJECT_ROOT의 origin remote로 owner/repo를 자동 감지."""
    try:
        return github_client.get_my_repo(CODE_PROJECT_ROOT)
    except github_client.GitHubAPIError as e:
        _github_error_response(e)


@app.get("/github/repo-info")
def github_repo_info(owner: str = Query(...), repo: str = Query(...)):
    try:
        return github_client.repo_info(owner, repo)
    except github_client.GitHubAPIError as e:
        _github_error_response(e)


@app.get("/github/tree")
def github_tree(owner: str = Query(...), repo: str = Query(...), branch: str = Query(...)):
    try:
        return github_client.file_tree(owner, repo, branch)
    except github_client.GitHubAPIError as e:
        _github_error_response(e)


@app.get("/github/file")
def github_file(owner: str = Query(...), repo: str = Query(...),
                 path: str = Query(...), branch: str = Query(...)):
    try:
        return github_client.file_content(owner, repo, path, branch)
    except github_client.GitHubAPIError as e:
        _github_error_response(e)


@app.get("/github/commits")
def github_commits(owner: str = Query(...), repo: str = Query(...),
                    branch: str = Query(...), limit: int = Query(10)):
    try:
        return github_client.recent_commits(owner, repo, branch, limit)
    except github_client.GitHubAPIError as e:
        _github_error_response(e)


# ============================================================
# Docker 샌드박스 (신규, 2026-08-13)
# GitHub 오픈소스를 clone해서 격리 컨테이너 안에서만 실행. 로컬 프로젝트
# (CODE_PROJECT_ROOT)와는 완전히 별개 - 여기서 무슨 일이 나도 실제 작업
# 중인 코드에는 영향이 없다.
# ============================================================

@app.get("/sandbox/status")
def sandbox_status():
    return {"available": sandbox.is_available()}


class SandboxCloneRequest(BaseModel):
    owner: str
    repo: str
    branch: Optional[str] = None


@app.post("/sandbox/clone")
def sandbox_clone(request: SandboxCloneRequest):
    if not sandbox.is_available():
        raise HTTPException(status_code=400, detail="Docker가 설치돼 있지 않습니다.")
    try:
        return sandbox.clone_repo(request.owner, request.repo, request.branch)
    except sandbox.SandboxError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SandboxRunRequest(BaseModel):
    workspace_id: str
    command: str
    timeout: int = 60
    allow_network: bool = False  # 기본 차단 - 정말 필요할 때만 명시적으로 허용


@app.post("/sandbox/run")
def sandbox_run(request: SandboxRunRequest):
    try:
        return sandbox.run_in_sandbox(
            request.workspace_id, request.command, request.timeout, request.allow_network
        )
    except sandbox.SandboxError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SandboxCleanupRequest(BaseModel):
    workspace_id: str


@app.post("/sandbox/cleanup")
def sandbox_cleanup(request: SandboxCleanupRequest):
    ok = sandbox.cleanup_workspace(request.workspace_id)
    if not ok:
        raise HTTPException(status_code=404, detail="워크스페이스를 찾을 수 없습니다.")
    return {"status": "cleaned"}


# ============================================================
# 툴콜링 에이전트 루프 (신규, 2026-08-13)
# LLM이 스스로 읽기전용 도구(파일 읽기, repo map, git 상태, GitHub 조회)를
# 여러 번 호출해가며 조사한 뒤 최종 답변(+SEARCH/REPLACE 제안)을 내놓는다.
# 도구 목록에 파일 쓰기/커밋/푸시는 절대 없음 - 적용은 지금처럼 항상 사람이
# "적용" 버튼을 눌러야 한다.
# ============================================================

AGENT_MAX_ITERATIONS = 10


class AgentChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = None
    fast_mode: bool = False  # 2026-08-13: 켜면 30b 대신 14b(CODE_LIGHT)로 툴콜링 - 속도 비교 실험용


@app.post("/codeanalysis/agent/stream")
async def agent_chat_stream(request: AgentChatRequest, session: Session = Depends(get_session)):
    history_rows = session.exec(
        select(CodeChatMessage)
        .where(CodeChatMessage.session_id == request.session_id)
        .order_by(CodeChatMessage.created_at)
    ).all()

    system_prompt = (
        "You are an expert coding assistant with access to read-only research tools "
        "(lint_file, semantic_search, search_in_project, list_files, read_file, "
        "read_lines, repo_map, git_status, "
        "github_repo_info, github_tree, github_file) for a local project (FastAPI "
        "backend `hf_crawler` + React/Vite frontend `hf-frontend`). Use these tools as "
        "many times as needed to gather the information you need before answering - do "
        "not guess about file contents you haven't read. Answer in Korean unless the "
        "user writes in another language.\n\n"
        "IMPORTANT for speed: if the user asks about bugs, errors, or code quality in "
        "a specific file, call lint_file FIRST - it catches real issues instantly with "
        "no LLM reasoning needed, and is far more reliable than reading raw code "
        "yourself. Only read the file afterward if you need to check logic-level "
        "issues that a linter can't catch.\n\n"
        "IMPORTANT for speed: if the user is asking about something specific (a "
        "function, endpoint, variable, feature), use search_in_project FIRST to jump "
        "straight to it - do not read a large file sequentially from the start unless "
        "the user explicitly wants a full-file review/summary. Reading a big file "
        "chunk-by-chunk is slow and should be a last resort.\n\n"
        "CRITICAL - do not confuse units: search_in_project results show 'file:LINE: "
        "...' (a line number). read_file's offset parameter is a CHARACTER count, a "
        "completely different unit - passing a line number as read_file's offset will "
        "land you in the wrong part of the file. After a search_in_project hit, use "
        "read_lines(path, start_line, end_line) instead, which takes the actual line "
        "numbers directly.\n\n"
        "Files can be long. read_file returns content in chunks with an explicit offset "
        "for continuing - if a chunk doesn't clearly say '이 파일의 실제 끝' (this is the "
        "actual end of the file), there is MORE content after it; never conclude a file "
        "is incomplete, broken, or that an import is unused just because one chunk ended. "
        "For large files, prefer calling repo_map first to see the function/class "
        "structure, then use read_file only for the specific part you actually need.\n\n"
        "Only report a bug or issue if you can point to the EXACT code you read that "
        "shows it - never report something as a problem based on a part of the file you "
        "have not seen. If you are not fully certain, say so explicitly rather than "
        "stating it as fact.\n\n"
        "Once you have enough information, give your final answer. "
        f"If proposing a code change, use this format:\n{patch_utils.PATCH_FORMAT_INSTRUCTIONS}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for row in history_rows:
        messages.append({"role": row.role, "content": row.content})
    messages.append({"role": "user", "content": request.message})

    session.add(CodeChatMessage(
        session_id=request.session_id,
        user_id=request.user_id,
        role="user",
        content=request.message,
    ))
    session.commit()

    async def event_generator() -> AsyncGenerator[str, None]:
        full_display_text = ""
        final_answer_text = ""
        agent_task = "agent_loop_light" if request.fast_mode else "agent_loop"

        try:
            for iteration in range(AGENT_MAX_ITERATIONS):
                thinking_note = f"\n\n🤔 *(조사 {iteration + 1}/{AGENT_MAX_ITERATIONS})*\n" if iteration > 0 else ""
                if thinking_note:
                    full_display_text += thinking_note
                    yield f"data: {json.dumps({'delta': thinking_note}, ensure_ascii=False)}\n\n"

                response = await asyncio.to_thread(
                    model_router.chat_with_tools, agent_task, messages, agent_tools.TOOL_SCHEMAS
                )
                msg = response["message"]
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else msg.tool_calls

                if not tool_calls:
                    final_answer_text = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                    full_display_text += final_answer_text
                    yield f"data: {json.dumps({'delta': final_answer_text}, ensure_ascii=False)}\n\n"
                    break

                # assistant의 도구 호출 메시지를 대화에 추가 (다음 라운드에 필요)
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", "") if isinstance(msg, dict) else (msg.content or ""),
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc, dict) else tc.function
                    name = fn.get("name") if isinstance(fn, dict) else fn.name
                    args = fn.get("arguments") if isinstance(fn, dict) else fn.arguments

                    call_note = f"\n🔧 **{name}**({json.dumps(args, ensure_ascii=False)})\n"
                    full_display_text += call_note
                    yield f"data: {json.dumps({'delta': call_note, 'tool_call': {'name': name, 'args': args}}, ensure_ascii=False)}\n\n"

                    result_text = agent_tools.call_tool(name, args)
                    preview = result_text[:300] + ("..." if len(result_text) > 300 else "")
                    result_note = f"→ {preview}\n"
                    full_display_text += result_note
                    yield f"data: {json.dumps({'delta': result_note, 'tool_result': {'name': name, 'result': preview}}, ensure_ascii=False)}\n\n"

                    messages.append({"role": "tool", "content": result_text, "tool_name": name})
            else:
                # 반복 한도 초과 - 도구 없이 마지막으로 한 번 더 요청해서 강제로 답을 받는다.
                # 2026-08-13(개정): "본 것만으로 정리해달라"는 요청이 오히려 모델이
                # 못 읽은 부분까지 자신 있게 단정하며 없는 버그를 지적하게 만들었다
                # (실제로 관찰된 사례: 방금 읽은 조각 안에 있는 null 체크를 "없다"고
                # 반대로 말함). 이제는 "못 본 부분은 절대 단정하지 말고, 어디까지
                # 읽었는지부터 솔직히 밝히라"고 명시한다.
                messages.append({
                    "role": "user",
                    "content": (
                        "조사 한도에 도달했습니다. 지금까지 읽은 내용만으로 최종 답변을 "
                        "작성하되, 반드시 다음을 지키세요:\n"
                        "1. 먼저 어떤 파일의 어느 구간(offset)까지 읽었는지 명시하세요.\n"
                        "2. 읽지 못한 부분에 대해서는 버그나 문제를 단정적으로 지적하지 "
                        "마세요 - '이 부분은 확인하지 못했습니다'라고 솔직히 말하세요.\n"
                        "3. 실제로 읽은 코드 조각 안에서 직접 확인한 것만 문제로 지적하세요. "
                        "확실하지 않으면 '~일 수도 있어 보이나 확실하지 않습니다'처럼 표현하세요."
                    ),
                })
                response = await asyncio.to_thread(model_router.chat_with_tools, agent_task, messages, [])
                msg = response["message"]
                final_answer_text = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                full_display_text += final_answer_text
                yield f"data: {json.dumps({'delta': final_answer_text}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"에이전트 루프 오류: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return
        finally:
            if full_display_text:
                clean_final, edits = patch_utils.extract_edits(final_answer_text)
                # 화면엔 조사 과정 전체 + 최종 답변(patch 블록만 제거)을 같이 보여준다
                display_content = full_display_text
                if edits and final_answer_text:
                    display_content = full_display_text.replace(
                        final_answer_text, clean_final or "(수정 제안을 아래에서 확인해주세요)"
                    )
                elif patch_utils.looks_like_failed_patch_attempt(final_answer_text, edits):
                    display_content += (
                        "\n\n⚠️ 모델이 수정 제안을 언급했지만 적용 가능한 형식으로 "
                        "만들지 못했습니다. 다시 시도해보시는 걸 권장합니다."
                    )
                with Session(engine) as save_session:
                    save_session.add(CodeChatMessage(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        role="assistant",
                        content=display_content or "(응답을 받지 못했습니다)",
                        model_used=model_router.model_for_task(agent_task),
                        proposed_edits=json.dumps(edits, ensure_ascii=False) if edits else None,
                    ))
                    save_session.commit()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# 벡터 검색 (신규, 2026-08-13)
# nomic-embed-text로 코드 청크를 임베딩해서 "의미로 찾기"를 지원한다.
# search_in_project(grep)가 "정확한 단어"만 찾는다면, 이건 "인증 로직 어디?"
# 처럼 정확한 이름을 몰라도 개념으로 찾을 수 있게 한다. sqlite-vec 같은 네이티브
# 확장 없이 순수 파이썬 코사인 유사도로 계산 - 이 프로젝트 규모에선 충분히 빠름.
# ============================================================

@app.get("/codeanalysis/vector-status")
def vector_status():
    return {"available": vector_search.is_available()}


class VectorIndexRequest(BaseModel):
    force: bool = False  # True면 변경 여부와 상관없이 전부 다시 임베딩


@app.post("/codeanalysis/vector-index")
def vector_index(request: VectorIndexRequest, session: Session = Depends(get_session)):
    if not vector_search.is_available():
        raise HTTPException(status_code=400, detail="nomic-embed-text 모델을 쓸 수 없습니다 (ollama pull nomic-embed-text).")
    file_list = _build_code_file_tree(CODE_PROJECT_ROOT)
    result = vector_search.build_index(session, CODE_PROJECT_ROOT, file_list, force=request.force)
    return result


@app.get("/codeanalysis/vector-search")
def vector_search_endpoint(query: str = Query(...), top_k: int = Query(5), session: Session = Depends(get_session)):
    if not vector_search.is_available():
        raise HTTPException(status_code=400, detail="nomic-embed-text 모델을 쓸 수 없습니다 (ollama pull nomic-embed-text).")
    results = vector_search.search(session, query, top_k=top_k)
    return {"results": results}