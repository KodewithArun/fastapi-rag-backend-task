import hashlib
import io
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_qdrant_service
from app.db.session import get_db
from app.db.vector_store import QdrantService
from app.models.metadata import DocumentMetadata
from app.schemas.document import DocumentResponse
from app.services.chunker import get_chunker
from app.services.document_parser import load_documents
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


def _duplicate_file_conflict(row: DocumentMetadata) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "message": "This file was already ingested (identical content).",
            "existing_document_id": row.id,
            "filename": row.filename,
        },
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    db: Session = Depends(get_db),
    qdrant_service: QdrantService = Depends(get_qdrant_service),
):
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported document format. Only .txt and .pdf are allowed."
        )

    try:
        file_bytes = await file.read()
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = db.scalar(
            select(DocumentMetadata).where(DocumentMetadata.content_hash == content_hash)
        )
        if existing is not None:
            raise _duplicate_file_conflict(existing)

        file_stream = io.BytesIO(file_bytes)

        documents = load_documents(file.content_type, file_stream, source=file.filename)

        if not documents or not any((d.page_content or "").strip() for d in documents):
            raise HTTPException(status_code=400, detail="No readable text found in document.")

        chunker = get_chunker(chunk_strategy)
        chunked_docs = chunker.chunk_documents(documents)
        chunks = [c.page_content for c in chunked_docs]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Chunking failed, resulting in zero chunks.")

        embedder = get_embedder()
        vectors = embedder.embed_documents(chunks)
        
        metadata_record = DocumentMetadata(
            filename=file.filename,
            file_type=file.content_type,
            chunk_strategy=chunk_strategy,
            chunks_count=len(chunks),
            content_hash=content_hash,
        )
        db.add(metadata_record)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raced = db.scalar(
                select(DocumentMetadata).where(DocumentMetadata.content_hash == content_hash)
            )
            if raced is not None:
                raise _duplicate_file_conflict(raced) from None
            raise
        db.refresh(metadata_record)

        chunk_metadata = [
            {k: v for k, v in doc.metadata.items() if v is not None}
            for doc in chunked_docs
        ]
        await qdrant_service.upsert_chunks(
            document_id=metadata_record.id,
            chunks=chunks,
            vectors=vectors,
            additional_metadata=chunk_metadata if any(chunk_metadata) else None,
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

    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Value Error during ingestion: {str(ve)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during insertion: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server encountered an error processing the document.")
