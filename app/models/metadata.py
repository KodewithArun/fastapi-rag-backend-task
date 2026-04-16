import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from app.db.session import Base


def generate_uuid():
    return str(uuid.uuid4())

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    filename = Column(String, index=True, nullable=False)
    upload_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    file_type = Column(String, nullable=False)  # Example: .pdf, .txt
    chunk_strategy = Column(String, nullable=False)  # Strategy used to split the doc
    chunks_count = Column(Integer, nullable=False)
