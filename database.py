# database.py (hf_coder)
# hf_crawler의 database.py와 동일한 WAL/busy_timeout 패턴을 그대로 재사용하되,
# 완전히 별도의 DB 파일을 쓴다. hf_crawler와 테이블/스키마를 공유하지 않는다
# (CodeChatMessage.user_id는 문자열만 저장 - hf_crawler의 users 테이블과
# 외래키로 연결하지 않는다. 서로 다른 SQLite 파일 사이엔 FK가 성립하지 않는다).
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event
from typing import Generator
import os

DB_NAME = os.getenv("HF_CODER_DB_NAME", "hf_coder.db")
DATABASE_URL = f"sqlite:///./{DB_NAME}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    echo=False
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """hf_crawler와 동일한 이유로 WAL + busy_timeout 적용 (채팅 저장 중 읽기가
    막히지 않게, 여러 요청이 겹쳐도 'database is locked'를 덜 만나게)."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    print(f"✅ hf_coder 데이터베이스 테이블이 생성되었습니다. ({DB_NAME})")


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


if __name__ == "__main__":
    create_db_and_tables()
    print(f"📁 데이터베이스 파일: {DB_NAME}")
