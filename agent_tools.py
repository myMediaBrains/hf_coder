# agent_tools.py (hf_coder)
#
# 툴콜링 에이전트 루프에서 LLM이 스스로 호출할 수 있는 도구 정의.
#
# 핵심 원칙: 여기 있는 도구는 전부 "읽기 전용"이다. 파일 쓰기(apply_edit),
# 커밋/푸시, 샌드박스 실행 같은 건 절대 포함하지 않는다 - 그런 행동은
# 지금까지처럼 항상 사람이 버튼을 눌러야만 실행된다. 에이전트는 여러 파일을
# 스스로 조사만 하고, 최종 답변에서 SEARCH/REPLACE 제안을 내놓으면 그걸
# 사람이 검토 후 적용하는 기존 흐름을 그대로 탄다.
from pathlib import Path
from typing import Callable

import code_map
import github_client
import lint_runner
import vector_search
from sqlmodel import Session
from database import engine

# main.py가 주입해주는 콜백들 (CODE_PROJECT_ROOT에 접근해야 하는 함수들이라
# main.py 쪽 헬퍼를 그대로 재사용 - 여기서 CODE_PROJECT_ROOT를 새로 정의하지 않음)
_build_file_tree_fn: Callable[[], list[str]] = None
_resolve_path_fn: Callable[[str], Path] = None
_project_root: Path = None


def configure(build_file_tree_fn, resolve_path_fn, project_root: Path):
    global _build_file_tree_fn, _resolve_path_fn, _project_root
    _build_file_tree_fn = build_file_tree_fn
    _resolve_path_fn = resolve_path_fn
    _project_root = project_root


def _list_files(_args: dict) -> str:
    files = _build_file_tree_fn()
    return "\n".join(files) if files else "(파일 없음)"


READ_FILE_CHUNK_SIZE = 16000


def _read_file(args: dict) -> str:
    """
    2026-08-13(개정): 큰 파일(main.py 등 수천 줄)을 8000자에서 그냥 잘라버리기만
    하고 이어서 볼 방법이 없었다 - 그러다 보니 모델이 "파일이 여기서 끝났다"고
    착각하고, 뒷부분에 있는 실제 사용처를 못 보고 "안 쓰는 import"라고 잘못
    판단하는 일이 있었다. offset 파라미터로 이어서 읽을 수 있게 하고, 잘렸을
    때는 "파일이 끝난 게 아니라 더 있다"는 걸 모호함 없이 명시한다.
    """
    path = args.get("path", "")
    offset = max(0, int(args.get("offset", 0) or 0))
    try:
        target = _resolve_path_fn(path)
        if not target.is_file():
            return f"파일을 찾을 수 없습니다: {path}"
        full_content = target.read_text(encoding="utf-8", errors="ignore")
        total_len = len(full_content)

        if total_len <= READ_FILE_CHUNK_SIZE and offset == 0:
            return full_content

        chunk = full_content[offset:offset + READ_FILE_CHUNK_SIZE]
        next_offset = offset + READ_FILE_CHUNK_SIZE
        remaining = total_len - next_offset

        if remaining > 0:
            note = (
                f"\n\n[이 파일은 총 {total_len}자입니다. 지금은 {offset}~{next_offset}자 구간만 "
                f"보여드렸고, 이후 {remaining}자가 더 있습니다 - 파일이 여기서 끝난 게 아닙니다. "
                f"이어서 보려면 read_file(path=\"{path}\", offset={next_offset})를 호출하세요.]"
            )
        else:
            note = f"\n\n[여기가 파일의 실제 끝입니다 (총 {total_len}자 중 {offset}자부터 끝까지).]"

        return chunk + note
    except Exception as e:
        return f"읽기 실패: {e}"


def _read_lines(args: dict) -> str:
    """
    2026-08-13(신규): search_in_project는 결과를 "파일:줄번호" 형식으로 준다.
    그런데 read_file의 offset은 "글자 수"라서 단위가 다르다 - 모델이 검색
    결과의 줄번호를 그대로 read_file의 offset에 넘겨서 엉뚱한 위치를 읽는
    사고가 실제로 있었다. 이 도구는 줄 번호를 그대로 받아서 정확한 위치를 읽는다.
    """
    path = args.get("path", "")
    if not path:
        return "path를 지정해주세요."
    try:
        start_line = int(args.get("start_line", 1) or 1)
    except (TypeError, ValueError):
        start_line = 1
    try:
        end_line = int(args.get("end_line", start_line + 50) or (start_line + 50))
    except (TypeError, ValueError):
        end_line = start_line + 50

    try:
        target = _resolve_path_fn(path)
        if not target.is_file():
            return f"파일을 찾을 수 없습니다: {path}"
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        return f"읽기 실패: {e}"

    total_lines = len(lines)
    start_idx = max(0, start_line - 1)
    end_idx = min(total_lines, max(end_line, start_line))
    if start_idx >= total_lines:
        return f"이 파일은 총 {total_lines}줄입니다. {start_line}번째 줄은 범위를 벗어났습니다."

    selected = lines[start_idx:end_idx]
    numbered = "\n".join(f"{start_idx + i + 1}: {line}" for i, line in enumerate(selected))
    return f"{path} ({start_idx + 1}~{end_idx}번째 줄, 전체 {total_lines}줄):\n{numbered}"


def _repo_map(_args: dict) -> str:
    if not code_map.is_available():
        return "(tree-sitter 미설치 - repo map 사용 불가)"
    files = _build_file_tree_fn()
    result = code_map.get_repo_map(_project_root, files)
    return result or "(함수/클래스 정보 없음)"


def _git_status(_args: dict) -> str:
    import subprocess
    proc = subprocess.run(
        ["git", "-C", str(_project_root), "status", "--porcelain"],
        capture_output=True, text=True, timeout=15,
    )
    return proc.stdout.strip() or "(변경된 파일 없음)"


def _lint_file(args: dict) -> str:
    """
    2026-08-13(신규): "이 파일에 오류 있는지 봐줘" 같은 요청에 LLM이 파일
    전체를 다 읽게 하는 대신, ruff/eslint 같은 결정론적 도구로 즉시(LLM 추론
    없이, 공짜로) 실제 문법/스타일 오류를 먼저 잡게 한다. Kimi류 MoE 모델이
    "필요한 전문가만 깨우는" 것과 같은 원리 - 코드를 눈으로 훑기 전에, 이미
    있는 정확한 도구부터 써야 한다.
    """
    path = args.get("path", "")
    if not path:
        return "path를 지정해주세요."
    result = lint_runner.run_lint(_project_root, path)
    if not result.get("available"):
        return f"이 파일 형식은 자동 lint 대상이 아니거나 도구가 설치돼 있지 않습니다: {result.get('detail', '')}"
    if result.get("ok"):
        return f"{result['tool']} 통과 - 발견된 문제 없음."
    issues = result.get("issues") or []
    if not issues and result.get("error"):
        return f"{result['tool']} 실행 오류: {result['error']}"
    return f"{result['tool']} 결과 ({len(issues)}건):\n" + "\n".join(issues)


def _semantic_search(args: dict) -> str:
    """
    2026-08-13(신규): search_in_project(grep)는 정확한 단어만 찾는다. 이건
    "인증 로직이 어디 있어?"처럼 정확한 함수/변수 이름을 몰라도, 의미가
    비슷한 코드를 찾아준다(nomic-embed-text 로컬 임베딩 기반). 인덱스가
    아직 없으면(처음 사용) 조용히 안내만 하고 실패하지 않는다.
    """
    query = args.get("query", "").strip()
    if not query:
        return "검색어(query)를 지정해주세요."
    if not vector_search.is_available():
        return "의미 검색을 쓸 수 없습니다 (nomic-embed-text 모델 필요) - search_in_project를 대신 쓰세요."

    with Session(engine) as session:
        results = vector_search.search(session, query, top_k=5)

    if not results:
        return "인덱스가 비어있습니다. 먼저 벡터 인덱스를 빌드해야 합니다 (아직 색인 전이거나, 관련된 코드가 없을 수 있습니다)."
    if results and "error" in results[0]:
        return f"검색 실패: {results[0]['error']}"

    lines = [f"'{query}' 의미 검색 결과 (유사도 높은 순):"]
    for r in results:
        lines.append(f"- {r['path']} [{r['label']}] (유사도 {r['score']}): {r['preview'][:150]}")
    return "\n".join(lines)


def _github_repo_info(args: dict) -> str:
    try:
        info = github_client.repo_info(args.get("owner", ""), args.get("repo", ""))
        return (f"{info['full_name']}: {info.get('description') or '(설명 없음)'}\n"
                f"기본 브랜치: {info['default_branch']}, 언어: {info.get('language')}, "
                f"⭐{info.get('stars')}")
    except github_client.GitHubAPIError as e:
        return f"조회 실패: {e}"


def _github_tree(args: dict) -> str:
    try:
        result = github_client.file_tree(args.get("owner", ""), args.get("repo", ""), args.get("branch", ""))
        paths = [f["path"] for f in result["files"]]
        return "\n".join(paths[:300]) if paths else "(파일 없음)"
    except github_client.GitHubAPIError as e:
        return f"조회 실패: {e}"


def _github_file(args: dict) -> str:
    try:
        result = github_client.file_content(
            args.get("owner", ""), args.get("repo", ""), args.get("path", ""), args.get("branch", "")
        )
        content = result["content"]
        if len(content) > 8000:
            content = content[:8000] + "\n... (내용이 길어 일부만 표시됨)"
        return content
    except github_client.GitHubAPIError as e:
        return f"조회 실패: {e}"


SEARCH_MAX_MATCHES = 30
SEARCH_CONTEXT_CHARS = 80


def _search_in_project(args: dict) -> str:
    """
    2026-08-13(신규): 큰 파일을 처음부터 순서대로 다 읽게 하지 않고, 원하는
    키워드/패턴이 있는 위치로 바로 점프할 수 있게 하는 도구. main.py처럼
    수천~십만자짜리 파일에서 특정 함수/엔드포인트를 찾을 때, read_file로
    8~10번 왕복하는 대신 이 도구 한 번으로 끝낼 수 있어 훨씬 빠르다.
    """
    import re as _re

    query = args.get("query", "").strip()
    if not query:
        return "검색어(query)를 지정해주세요."

    file_filter = args.get("file", "").strip()  # 특정 파일로 범위를 좁히고 싶을 때

    try:
        pattern = _re.compile(_re.escape(query), _re.IGNORECASE)
    except _re.error as e:
        return f"검색어 처리 오류: {e}"

    all_files = _build_file_tree_fn()
    if file_filter:
        all_files = [f for f in all_files if file_filter in f]

    results = []
    for rel_path in all_files:
        if len(results) >= SEARCH_MAX_MATCHES:
            break
        try:
            content = (_project_root / rel_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                snippet = line.strip()[:200]
                results.append(f"{rel_path}:{line_no}: {snippet}")
                if len(results) >= SEARCH_MAX_MATCHES:
                    break

    if not results:
        return f"'{query}'를 포함한 줄을 찾지 못했습니다."

    limit_note = " (최대치 도달 - 더 구체적으로 검색해보세요)" if len(results) >= SEARCH_MAX_MATCHES else ""
    header = f"'{query}' 검색 결과 ({len(results)}건{limit_note}):\n"
    return header + "\n".join(results)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lint_file",
            "description": (
                "Run a deterministic static analyzer (ruff for .py, eslint for "
                ".js/.jsx) on a file and get real syntax/style issues INSTANTLY, "
                "with no LLM reasoning needed. ALWAYS try this FIRST when the user "
                "asks about bugs, errors, or code quality in a specific file - it's "
                "free and far more reliable than eyeballing raw code. Only fall back "
                "to reading the file yourself for logic-level issues this can't catch."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path, e.g. main.py"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Search for code by MEANING/CONCEPT rather than exact text (e.g. "
                "'where is user authentication handled', 'translation logic'). Use "
                "this when you don't know the exact function/variable name to grep "
                "for - search_in_project is better when you DO know the exact term."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you're looking for, described naturally"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_project",
            "description": (
                "Search for a keyword or text across all project files (like grep). "
                "Returns matching file:line snippets. PREFER THIS over reading a whole "
                "large file sequentially when you're looking for something specific "
                "(e.g. a function name, endpoint path, variable) - it's much faster than "
                "paging through read_file offset by offset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"},
                    "file": {"type": "string", "description": "Optional: restrict search to files whose path contains this substring, e.g. 'main.py'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List every file that exists in the current local project.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file's content in the local project, starting at a CHARACTER "
                "offset (not a line number!). Large files are returned in chunks - if "
                "the response says there is more content remaining, call this again "
                "with the given next offset to continue reading. Do NOT assume a file "
                "ends just because one chunk stops - only trust the explicit "
                "'이 파일의 실제 끝' marker. IMPORTANT: if you got a line number from "
                "search_in_project, do NOT pass it as offset here - use read_lines "
                "instead, which takes actual line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path, e.g. main.py"},
                    "offset": {"type": "integer", "description": "Character offset to start reading from (0 for the beginning) - NOT a line number. Use this to continue reading a large file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": (
                "Read a file by LINE NUMBER range - use this to jump straight to a "
                "location reported by search_in_project (which shows results as "
                "'file:line: ...'). This is the correct tool for line numbers - "
                "read_file's offset is a character count and will land in the wrong "
                "place if you pass a line number to it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path, e.g. main.py"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to read (defaults to start_line + 50)"},
                },
                "required": ["path", "start_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_map",
            "description": "Get a compact summary of functions/classes across all code files in the project.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "See which files currently have uncommitted changes in the local project (read-only).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_repo_info",
            "description": "Look up basic info (description, default branch, language, stars) for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_tree",
            "description": "List files in a GitHub repository (not the local project - an external repo).",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                },
                "required": ["owner", "repo", "branch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_file",
            "description": "Read a file's content from a GitHub repository (not the local project - an external repo).",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["owner", "repo", "branch", "path"],
            },
        },
    },
]

_DISPATCH: dict[str, Callable[[dict], str]] = {
    "lint_file": _lint_file,
    "semantic_search": _semantic_search,
    "search_in_project": _search_in_project,
    "list_files": _list_files,
    "read_file": _read_file,
    "read_lines": _read_lines,
    "repo_map": _repo_map,
    "git_status": _git_status,
    "github_repo_info": _github_repo_info,
    "github_tree": _github_tree,
    "github_file": _github_file,
}


def call_tool(name: str, args: dict) -> str:
    fn = _DISPATCH.get(name)
    if not fn:
        return f"알 수 없는 도구입니다: {name}"
    try:
        return fn(args or {})
    except Exception as e:
        return f"도구 실행 중 오류: {e}"
