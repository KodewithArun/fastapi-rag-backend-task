from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Document Ingestion and Conversational RAG",
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Palm Mind AI Backend is running."}

