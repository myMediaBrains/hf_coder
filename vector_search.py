# vector_search.py (hf_coder)
#
# 로컬 임베딩(nomic-embed-text, Ollama) 기반 의미 검색.
#
# 설계 결정: sqlite-vec 같은 네이티브 SQLite 확장은 파이썬 빌드에 따라 로드가
# 안 될 수 있다(이 프로젝트에서 여러 번 겪은 "환경 호환성 문제" 패턴을 반복하고
# 싶지 않았음) - 대신 임베딩을 그냥 일반 테이블에 저장하고, 검색 시 파이썬으로
# 직접 코사인 유사도를 계산한다. 파일 수가 수백 개 수준인 개인 프로젝트에서는
# 이 브루트포스 방식으로도 충분히 빠르다(비교 자체가 수백 번의 768차원 내적일
# 뿐이라 밀리초 단위).
#
# 청크 전략: tree-sitter가 지원하는 언어(.py/.js/.jsx)는 함수/클래스 단위로
# 쪼갠다(임의로 글자수 자르는 것보다 의미 있는 단위). 그 외 파일(.md 등)은
# 고정 크기로 청크. 파일이 마지막으로 인덱싱된 이후 안 바뀌었으면(mtime 비교)
# 다시 임베딩하지 않는다 - 매번 전체를 다시 계산하는 부하를 피함.
#
# 안전장치: nomic-embed-text가 없거나 Ollama 호출이 실패해도 서비스가 안
# 죽는다 - is_available()이 False가 되고, 검색은 빈 결과를 반환할 뿐이다.
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import ollama

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
FALLBACK_CHUNK_SIZE = 1500  # tree-sitter 미지원 파일용 고정 크기 청크
MAX_CHUNK_CHARS = 4000      # 함수 하나가 너무 길면 임베딩 품질이 떨어져서 상한을 둠

_TS_AVAILABLE = False
_PARSERS: dict[str, object] = {}
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as _tspython
    import tree_sitter_javascript as _tsjavascript

    _PARSERS["python"] = Parser(Language(_tspython.language()))
    _PARSERS["javascript"] = Parser(Language(_tsjavascript.language()))
    _TS_AVAILABLE = True
except Exception as e:
    logger.warning(f"[vector_search] tree-sitter 없음, 청크가 고정크기 방식으로만 동작: {e}")

_EXT_TO_LANG = {".py": "python", ".js": "javascript", ".jsx": "javascript"}
_CHUNK_NODE_TYPES = {
    "function_definition", "class_definition",       # python
    "function_declaration", "class_declaration",      # js
}


class EmbedError(Exception):
    pass


def is_available() -> bool:
    """실제로 nomic-embed-text가 응답하는지 가벼운 호출로 확인 (모델 자체가 없으면 예외)."""
    try:
        ollama.embeddings(model=EMBED_MODEL, prompt="ping")
        return True
    except Exception as e:
        logger.warning(f"[vector_search] nomic-embed-text 사용 불가: {e}")
        return False


def embed_text(text: str) -> list[float]:
    try:
        res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return list(res["embedding"])
    except Exception as e:
        raise EmbedError(f"임베딩 실패: {e}")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_file(abs_path: Path) -> list[tuple[str, str]]:
    """(청크 라벨, 청크 텍스트) 튜플 리스트를 반환. 라벨은 검색 결과 표시용
    (예: 'def foo' 또는 '1~1500자')."""
    lang = _EXT_TO_LANG.get(abs_path.suffix)
    try:
        source_text = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not source_text.strip():
        return []

    if lang and _TS_AVAILABLE and lang in _PARSERS:
        chunks = _chunk_by_symbols(abs_path, lang, source_text)
        if chunks:
            return chunks
    # tree-sitter 미지원이거나 함수/클래스가 하나도 없으면 고정 크기로 폴백
    return _chunk_fixed_size(source_text)


def _chunk_by_symbols(abs_path: Path, lang: str, source_text: str) -> list[tuple[str, str]]:
    parser = _PARSERS[lang]
    source_bytes = source_text.encode("utf-8", errors="ignore")
    tree = parser.parse(source_bytes)
    chunks: list[tuple[str, str]] = []

    def walk(node):
        if node.type in _CHUNK_NODE_TYPES:
            name_node = node.child_by_field_name("name")
            name = (
                source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                if name_node else "(익명)"
            )
            body = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            if body.strip():
                chunks.append((name, body[:MAX_CHUNK_CHARS]))
            return  # 최상위 정의만 - 클래스 내부 메서드까지 더 쪼개면 너무 잘게 나뉨
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def _chunk_fixed_size(text: str) -> list[tuple[str, str]]:
    chunks = []
    for i in range(0, len(text), FALLBACK_CHUNK_SIZE):
        piece = text[i:i + FALLBACK_CHUNK_SIZE]
        if piece.strip():
            chunks.append((f"{i}~{i + len(piece)}자", piece))
    return chunks


# ============================================================
# 인덱스 저장/조회 - CodeEmbedding 테이블을 DB 세션으로 직접 다룬다
# (main.py에서 session을 주입받는 형태로 호출)
# ============================================================

def build_index(session, root: Path, rel_paths: list[str], force: bool = False) -> dict:
    """
    변경된 파일만(또는 force=True면 전부) 다시 청크+임베딩해서 DB에 반영한다.
    반환값: {"indexed": N개 파일, "skipped": M개 파일(안 바뀜), "chunks": 총 청크 수}
    """
    from models import CodeEmbedding  # 순환 임포트 방지를 위해 함수 내부에서 임포트
    from sqlmodel import select, delete

    indexed, skipped, total_chunks = 0, 0, 0

    for rel_path in rel_paths:
        abs_path = root / rel_path
        if not abs_path.is_file():
            continue
        try:
            mtime = abs_path.stat().st_mtime
        except Exception:
            continue

        if not force:
            existing = session.exec(
                select(CodeEmbedding).where(CodeEmbedding.path == rel_path).limit(1)
            ).first()
            if existing and abs(existing.file_mtime - mtime) < 1e-6:
                skipped += 1
                continue

        chunks = _chunk_file(abs_path)
        if not chunks:
            continue

        # 이 파일의 기존 청크 삭제 후 새로 삽입 (재인덱싱)
        session.exec(delete(CodeEmbedding).where(CodeEmbedding.path == rel_path))

        for i, (label, chunk_text) in enumerate(chunks):
            try:
                vector = embed_text(f"{rel_path} - {label}\n\n{chunk_text}")
            except EmbedError as e:
                logger.warning(f"[vector_search] {rel_path}#{label} 임베딩 실패: {e}")
                continue
            session.add(CodeEmbedding(
                path=rel_path,
                chunk_label=label,
                chunk_text=chunk_text,
                embedding=json.dumps(vector),
                file_mtime=mtime,
            ))
            total_chunks += 1

        indexed += 1

    session.commit()
    return {"indexed": indexed, "skipped": skipped, "chunks": total_chunks}


def search(session, query: str, top_k: int = 5) -> list[dict]:
    """의미 기반 검색 - 코사인 유사도 상위 top_k개 청크를 반환."""
    from models import CodeEmbedding
    from sqlmodel import select

    try:
        query_vector = embed_text(query)
    except EmbedError as e:
        return [{"error": str(e)}]

    rows = session.exec(select(CodeEmbedding)).all()
    if not rows:
        return []

    scored = []
    for row in rows:
        try:
            vector = json.loads(row.embedding)
        except Exception:
            continue
        score = _cosine_similarity(query_vector, vector)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, row in scored[:top_k]:
        preview = row.chunk_text[:400] + ("..." if len(row.chunk_text) > 400 else "")
        results.append({
            "path": row.path,
            "label": row.chunk_label,
            "score": round(score, 4),
            "preview": preview,
        })
    return results
