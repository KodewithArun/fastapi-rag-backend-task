from sqlalchemy import Column, Integer, String, DateTime, Float
from datetime import datetime, timezone
import uuid

from app.db.session import Base

def generate_uuid():
    return str(uuid.uuid4())

class DocumentMetadata(Base):
    """
    SQLAlchemy Model for the Document Metadata table.
    Tracks ingested files so we aren't completely reliant on the Vector DB.
    """
    __tablename__ = "document_metadata"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    filename = Column(String, index=True, nullable=False)
    upload_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    file_type = Column(String, nullable=False)  # Example: .pdf, .txt
    chunk_strategy = Column(String, nullable=False)  # Strategy used to split the doc
    chunks_count = Column(Integer, nullable=False)
