import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self._client: Optional[AsyncQdrantClient] = None
        
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            url = getattr(settings, "QDRANT_URL", f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            self._client = AsyncQdrantClient(
                url=url,
                api_key=settings.QDRANT_API_KEY
            )
        return self._client

    async def initialize_collection(self, vector_size: int = 384) -> bool:
        """
        Creates the Qdrant vector database collection if it does not exist.
        """
        try:
            exists = await self.client.collection_exists(collection_name=self.collection_name)
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            raise

    async def upsert_chunks(
        self, 
        document_id: str, 
        chunks: List[str], 
        vectors: List[List[float]],
        additional_metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> bool:
        if len(chunks) != len(vectors):
            raise ValueError("Mismatched chunks and vectors count.")
            
        if additional_metadata and len(additional_metadata) != len(chunks):
            raise ValueError("Mismatched metadata and chunks count.")

        points = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            
            payload = {
                "document_id": document_id,
                "text": chunk,
                "chunk_index": idx,
                "timestamp": timestamp
            }
            
            # Merge in any additional metadata provided (like page numbers)
            if additional_metadata:
                payload.update(additional_metadata[idx])
                
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        try:
            # Execute batch upsert
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Upserted batch of {len(batch)} points for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert chunks to Qdrant: {str(e)}")
            raise

    async def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        try:
            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "document_id": hit.payload.get("document_id"),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "text": hit.payload.get("text"),
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ["document_id", "text", "chunk_index"]}
                }
                for hit in search_result.points
            ]
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {str(e)}")
            raise
