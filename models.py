# models.py (hf_coder)
# hf_crawler의 models.py에 있던 CodeChatMessage를 그대로 이식하되, user_id의
# foreign_key="users.user_id"는 제거했다. users 테이블은 hf_crawler의 DB에만
# 있고, 서로 다른 SQLite 파일 사이에는 외래키가 성립하지 않는다.
# user_id는 이제 그냥 문자열로만 저장한다 - 검증이 필요해지면 나중에
# hf_crawler의 GET /users/list API를 호출해서 확인하는 방식으로 가면 된다
# (append-only 원칙은 hf_crawler의 TextGeneration/CodeChatMessage와 동일하게 유지).
from typing import Optional
from sqlmodel import SQLModel, Field, Column, DateTime, func
from datetime import datetime


class CodeChatMessage(SQLModel, table=True):
    __tablename__ = "code_chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)   # 프론트에서 생성한 UUID, localStorage에 저장돼 새로고침해도 유지
    user_id: Optional[str] = Field(default=None, index=True)  # hf_crawler의 user_id 문자열을 그대로 보관 (FK 아님)

    role: str = Field(nullable=False)     # "user" | "assistant"
    content: str = Field(nullable=False)
    included_files: Optional[str] = None  # JSON 배열 문자열 - 이 메시지에 포함된 파일 경로들
    model_used: Optional[str] = None

    # 2026-08-13: VS Code 연동 루프 (저장감지 자동리뷰) 관련 필드
    source: str = Field(default="manual")  # "manual" (사람이 직접 질문) | "watcher" (저장 감지 자동리뷰)
    # 2026-08-13(개정): "파일 전체 재작성" 방식에서 Aider 스타일 SEARCH/REPLACE
    # 블록 방식으로 전환. proposed_edits는 JSON 배열 문자열:
    # [{"path": "...", "search": "...", "replace": "..."}, ...]
    proposed_edits: Optional[str] = None
    apply_results: Optional[str] = None     # 적용 시도 결과 JSON 배열 (성공/실패 사유 포함)
    applied: bool = Field(default=False)    # 이 제안이 (부분적으로라도) 적용된 적이 있는지

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now())
    )


class CodeEmbedding(SQLModel, table=True):
    """
    2026-08-13(신규): 로컬 벡터 검색용. nomic-embed-text로 임베딩한 코드 청크를
    저장한다. sqlite-vec 같은 네이티브 확장 대신 embedding을 그냥 JSON 문자열로
    저장하고, 검색 시 파이썬에서 코사인 유사도를 직접 계산한다(vector_search.py) -
    이 프로젝트 규모(수백 개 파일)에서는 이 방식으로도 충분히 빠르고, 환경별
    호환성 문제를 피할 수 있다.
    """
    __tablename__ = "code_embeddings"

    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True)          # 상대경로, 예: main.py
    chunk_label: str                        # 표시용 라벨, 예: "def foo" 또는 "0~1500자"
    chunk_text: str                         # 실제 청크 내용 (검색 결과 미리보기용)
    embedding: str                          # 768차원 벡터를 JSON 배열 문자열로 직렬화
    file_mtime: float                       # 마지막 인덱싱 시점의 파일 수정시각 - 변경 감지용

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, server_default=func.now())
    )
