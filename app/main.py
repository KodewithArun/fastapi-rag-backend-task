import logging
import time

from fastapi import FastAPI, Request

from app.api.v1.chat import router as chat_router
from app.api.v1.ingestion import router as document_router
from app.core.config import settings
from app.core.logger import setup_logging
from app.db.session import Base, engine

setup_logging()
middleware_logger = logging.getLogger("system.middleware")

# Initialize SQL tables synchronously on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Document Ingestion and Conversational RAG",
)


@app.middleware("http")
async def log_request_metrics(request: Request, call_next):
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    middleware_logger.info(
        "Incoming Request: %s | Path: %s | Client IP: %s",
        request.method,
        request.url.path,
        client_ip,
    )

    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    middleware_logger.info(
        "Request Completed: %s %s | Status: %s | Execution Time: %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Palm Mind AI Backend is running."}

app.include_router(document_router, prefix=settings.API_V1_STR)
app.include_router(chat_router, prefix=settings.API_V1_STR)

