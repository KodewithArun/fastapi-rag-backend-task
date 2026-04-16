import io
import logging
import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.vector_store import QdrantService
from app.models.metadata import DocumentMetadata
from app.schemas.document import DocumentResponse
from app.services.chunker import get_chunker
from app.services.document_parser import get_document_parser
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Ingestion"])

# Initialize vector DB singleton
qdrant_service = QdrantService()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    embed_provider: str = Form("huggingface"),
    db: Session = Depends(get_db)
):
    # Validate File Format
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported document format. Only .txt and .pdf are allowed."
        )

    try:
        file_bytes = await file.read()
        file_stream = io.BytesIO(file_bytes)
        
        parser = get_document_parser(file.content_type)
        raw_text = parser.extract_text(file_stream)
        
        if not raw_text or not raw_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in document.")

        chunker = get_chunker(chunk_strategy)
        chunks = chunker.chunk(raw_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Chunking failed, resulting in zero chunks.")

        embedder = get_embedder(embed_provider)
        vectors = embedder.embed_documents(chunks)
        
        metadata_record = DocumentMetadata(
            filename=file.filename,
            file_type=file.content_type,
            chunk_strategy=chunk_strategy,
            chunks_count=len(chunks)
        )
        db.add(metadata_record)
        db.commit()
        db.refresh(metadata_record)
        
        await qdrant_service.upsert_chunks(
            document_id=metadata_record.id,
            chunks=chunks,
            vectors=vectors
        )
        
        return DocumentResponse(
            id=metadata_record.id,
            filename=metadata_record.filename,
            upload_date=metadata_record.upload_date,
            file_type=metadata_record.file_type,
            chunk_strategy=metadata_record.chunk_strategy,
            chunks_count=metadata_record.chunks_count,
            message="Document successfully processed, chunked, and embedded."
        )

    except ValueError as ve:
        logger.error(f"Value Error during ingestion: {str(ve)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during insertion: {str(e)}")
        # Ideally, we would run db.rollback() or handle transient failures gracefully
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server encountered an error processing the document.")
