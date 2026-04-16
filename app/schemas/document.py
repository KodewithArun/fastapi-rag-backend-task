from pydantic import BaseModel, ConfigDict
from datetime import datetime

class DocumentResponse(BaseModel):
    """Response schema for a successfully ingested document."""
    id: str
    filename: str
    upload_date: datetime
    file_type: str
    chunk_strategy: str
    chunks_count: int
    message: str

    model_config = ConfigDict(from_attributes=True)
