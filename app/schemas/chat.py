from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Chat session id (Redis history)")
    query: str = Field(..., description="User message")
    document_id: Optional[str] = Field(None, description="Optional: scope RAG to this document")
    model_config = ConfigDict(from_attributes=True)

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model_config = ConfigDict(from_attributes=True)