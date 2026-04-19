from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_document_metadata_content_hash() -> None:
    """Add content_hash column/index on existing DBs (create_all does not alter tables)."""
    try:
        inspector = inspect(engine)
    except Exception:
        return
    if "document_metadata" not in inspector.get_table_names():
        return
    cols = {c["name"] for c in inspector.get_columns("document_metadata")}
    if "content_hash" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE document_metadata ADD COLUMN content_hash VARCHAR(64)"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_document_metadata_content_hash "
                "ON document_metadata (content_hash)"
            )
        )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
