from fastapi import FastAPI
from app.core.config import settings
from app.db.session import engine, Base
from app.api.v1.ingestion import router as document_router
from app.api.v1.chat import router as chat_router

# Initialize SQL tables synchronously on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Document Ingestion and Conversational RAG",
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Palm Mind AI Backend is running."}

app.include_router(document_router, prefix=settings.API_V1_STR)
app.include_router(chat_router, prefix=settings.API_V1_STR)

