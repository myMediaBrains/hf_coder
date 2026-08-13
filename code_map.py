# code_map.py (hf_coder)
#
# Tree-sitter로 파일마다 함수/클래스 시그니처를 뽑아 "코드 지도"를 만든다.
# 목적: 파일 전체 텍스트를 매번 컨텍스트에 넣는 대신, 이 압축된 요약을
# 기본으로 주고 실제 내용이 필요한 파일만 사용자가 체크박스로 선택하게 해서
# 토큰 사용량(=메모리/속도 부하)을 줄인다.
#
# 2026-08-13(개정): Aider의 repo map 방식을 참고해서 단순 나열이 아니라
# "참조 그래프 + PageRank"로 파일 순서를 정한다. Aider는 심볼 단위 그래프에
# 토큰예산 이진탐색까지 하는 훨씬 정교한 구현인데, 여기서는 그 핵심 아이디어
# (많이 참조되는 파일이 더 중요하다)만 파일 단위로 단순화해서 가져왔다 -
# "A 파일이 B 파일에서 정의된 이름을 쓰면 A→B 엣지"로 그래프를 만들고,
# PageRank로 파일 중요도를 매겨서 중요한 파일이 목록 위쪽에 오게 한다.
# networkx가 없어도(또는 참조 관계가 하나도 없어도) 그냥 알파벳 순으로
# 조용히 폴백한다 - 순위가 없어도 목록 자체는 항상 나와야 함.
#
# 주의: tree_sitter_languages(구 버전 API 전용) 패키지는 최신 tree_sitter와
# 호환이 깨져 있어서 쓰지 않는다 - 언어별 개별 패키지(tree-sitter-python,
# tree-sitter-javascript)를 최신 Language/Parser API로 직접 조합한다.
#
# 안전장치: 이 모듈이 의존하는 패키지가 하나라도 없으면 import 자체가
# 실패하지 않고 "사용 불가"/"순위 없음" 상태로만 표시한다 - 이 기능이 없어도
# 나머지 서비스(코딩분석 채팅, 감시, GitHub 탐색)는 정상 동작해야 함.
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_TS_AVAILABLE = False
_PARSERS: dict[str, object] = {}

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as _tspython
    import tree_sitter_javascript as _tsjavascript

    _PY_LANG = Language(_tspython.language())
    _JS_LANG = Language(_tsjavascript.language())
    _PARSERS["python"] = Parser(_PY_LANG)
    _PARSERS["javascript"] = Parser(_JS_LANG)
    _TS_AVAILABLE = True
except Exception as e:
    logger.warning(
        f"[code_map] tree-sitter 파서 로드 실패, repo map 기능 비활성화됨: {e}. "
        f"쓰려면 pip install tree-sitter tree-sitter-python tree-sitter-javascript"
    )

_NX_AVAILABLE = False
try:
    import networkx as nx
    _NX_AVAILABLE = True
except Exception as e:
    logger.warning(
        f"[code_map] networkx 없음, repo map 순위 매기기 비활성화(목록은 정상 표시됨): {e}. "
        f"쓰려면 pip install networkx"
    )

_EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
}

_SYMBOL_NODE_TYPES = {
    "function_definition": "def",     # python
    "class_definition": "class",      # python
    "function_declaration": "def",    # js
    "class_declaration": "class",     # js
    "method_definition": "def",       # js (class 내부 메서드)
}

# 파이썬 키워드/내장함수처럼 어디서나 등장해서 "참조 그래프"에 노이즈만
# 더하는 흔한 이름들 - 그래프에서 제외해야 순위가 의미 있어짐.
_COMMON_IDENTIFIERS = {
    "self", "cls", "args", "kwargs", "request", "response", "session",
    "True", "False", "None", "return", "import", "from", "def", "class",
    "if", "else", "elif", "for", "while", "try", "except", "finally",
    "print", "len", "str", "int", "float", "list", "dict", "set", "tuple",
    "id", "type", "name", "value", "data", "result", "item", "index",
}

_CACHE_TTL = 30  # 초 - 저장할 때마다 다시 파싱하면 부하라 짧게 캐시
_cache = {"built_at": 0.0, "root": None, "map_text": ""}


def is_available() -> bool:
    return _TS_AVAILABLE


def _parse_file(abs_path: Path, lang: str):
    """한 번의 파싱으로 표시용 심볼 목록 + 정의된 이름들 + 등장하는 식별자들을 같이 뽑는다."""
    parser = _PARSERS.get(lang)
    if not parser:
        return [], set(), set()
    try:
        source = abs_path.read_bytes()
    except Exception:
        return [], set(), set()

    tree = parser.parse(source)
    symbols: list[str] = []
    def_names: set[str] = set()
    identifiers: set[str] = set()

    def walk(node, indent=1):
        if node.type in _SYMBOL_NODE_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                kind = _SYMBOL_NODE_TYPES[node.type]
                symbols.append(("  " * indent) + f"{kind} {name}")
                def_names.add(name)
                indent += 1  # 메서드는 클래스 안쪽으로 한 단계 들여쓰기
        if node.type == "identifier":
            name = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            if name and name not in _COMMON_IDENTIFIERS:
                identifiers.add(name)
        for child in node.children:
            walk(child, indent)

    walk(tree.root_node)
    return symbols, def_names, identifiers


def _extract_symbols(abs_path: Path, lang: str) -> list[str]:
    """하위호환용 - 심볼 목록만 필요할 때."""
    symbols, _, _ = _parse_file(abs_path, lang)
    return symbols


def _rank_files(file_defs: dict[str, set], file_idents: dict[str, set]) -> tuple[dict[str, float], set[str]]:
    """
    파일 참조 그래프를 만들고 PageRank로 중요도를 매긴다.
    "A 파일이 B 파일에서 정의된 이름을 쓴다" -> A에서 B로 가는 엣지.
    많은 파일에서 참조되는 B는 PageRank가 높아져서 목록 앞쪽에 온다.

    반환값: (파일별 점수, 그래프에 엣지가 하나라도 연결된 파일 집합).
    두 번째 값이 필요한 이유 - PageRank는 수학적으로 "완전히 고립된 파일"이
    "실제로 다른 파일과 연결됐지만 인바운드 엣지가 적은 파일"보다 점수가 높게
    나오는 부작용이 있다(연결된 파일들은 서로 점수를 나눠 갖으므로). 그래서
    호출부에서 "연결 여부"를 우선 기준으로 쓰고, 그 안에서만 점수로 정렬한다.
    networkx가 없거나 참조 관계가 하나도 없으면 빈 값들을 반환 (알파벳순 폴백).
    """
    if not _NX_AVAILABLE:
        return {}, set()

    graph = nx.DiGraph()
    for rel_path in file_defs:
        graph.add_node(rel_path)

    for user_path, idents in file_idents.items():
        for def_path, def_names in file_defs.items():
            if user_path == def_path or not def_names:
                continue
            shared = idents & def_names
            if shared:
                graph.add_edge(user_path, def_path, weight=len(shared))

    if graph.number_of_edges() == 0:
        return {}, set()

    connected = {n for n in graph.nodes if graph.degree(n) > 0}

    try:
        ranks = nx.pagerank(graph, weight="weight")
    except Exception as e:
        logger.warning(f"[code_map] PageRank 계산 실패, 알파벳순으로 폴백: {e}")
        return {}, connected

    return ranks, connected


def build_repo_map(root: Path, rel_paths: list[str]) -> str:
    """rel_paths(문서 확장자 포함 전체 목록)에서 지원 언어 파일만 골라 요약.

    2026-08-13(개정1): 함수/클래스가 하나도 없는 파일(짧은 스크립트 등)을 통째로
    목록에서 빼면, LLM이 "구조 요약에 없으니 존재 안 한다"고 착각해서 실제로
    존재하는 파일을 "없다"고 답하는 문제가 있었다. 심볼이 없어도 파일명 자체는
    반드시 나오게 해서 "이 파일은 있는데 함수/클래스는 없다"를 구분할 수 있게 한다.

    2026-08-13(개정2): 파일 순서를 알파벳순 대신 PageRank 중요도순으로 바꿨다 -
    다른 파일들에서 많이 참조하는(=코드베이스에서 중심적인) 파일이 먼저 나온다.
    """
    if not _TS_AVAILABLE:
        return ""

    code_paths = [p for p in rel_paths if Path(p).suffix in _EXT_TO_LANG]

    file_symbols: dict[str, list[str]] = {}
    file_defs: dict[str, set] = {}
    file_idents: dict[str, set] = {}

    for rel_path in code_paths:
        lang = _EXT_TO_LANG[Path(rel_path).suffix]
        symbols, def_names, identifiers = _parse_file(root / rel_path, lang)
        file_symbols[rel_path] = symbols
        file_defs[rel_path] = def_names
        file_idents[rel_path] = identifiers

    ranks, connected = _rank_files(file_defs, file_idents)
    # 정렬 기준: ① 그래프에 연결된 파일 우선(고립 파일이 수학적으로 더 높은
    # 점수를 받는 PageRank 부작용을 피하려고), ② 그 안에서는 점수 내림차순,
    # ③ 동점/무순위 구간은 알파벳순으로 안정적으로 정렬.
    def _sort_key(p):
        is_isolated = p not in connected
        return (is_isolated, -ranks.get(p, 0.0), p)

    ordered = sorted(code_paths, key=_sort_key)

    lines = []
    for rel_path in ordered:
        lines.append(f"{rel_path}:")
        symbols = file_symbols.get(rel_path) or []
        if symbols:
            lines.extend(symbols)
        else:
            lines.append("  (함수/클래스 없음 - 짧은 스크립트이거나 단순 실행 코드)")
    return "\n".join(lines)


def get_repo_map(root: Path, rel_paths: list[str], force: bool = False) -> str:
    """짧은 TTL 캐시를 둔 버전 - 매 질문/저장마다 프로젝트 전체를 다시 파싱하지 않게."""
    if not _TS_AVAILABLE:
        return ""

    now = time.time()
    if not force and _cache["root"] == str(root) and (now - _cache["built_at"]) < _CACHE_TTL:
        return _cache["map_text"]

    map_text = build_repo_map(root, rel_paths)
    _cache.update({"built_at": now, "root": str(root), "map_text": map_text})
    return map_text
