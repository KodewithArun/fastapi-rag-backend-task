import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
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
        self._collection_ready = False
        
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            url = getattr(settings, "QDRANT_URL", f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            api_key = settings.QDRANT_API_KEY if str(url).startswith("https://") else None
            self._client = AsyncQdrantClient(
                url=url,
                api_key=api_key
            )
        return self._client

    async def initialize_collection(self, vector_size: int = 384) -> bool:
        """Ensure collection exists (list + create; handles races)."""
        try:
            collections = await self.client.get_collections()
            names = {c.name for c in collections.collections}
            if self.collection_name not in names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                try:
                    await self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                except UnexpectedResponse:
                    collections = await self.client.get_collections()
                    names = {c.name for c in collections.collections}
                    if self.collection_name not in names:
                        raise
            self._collection_ready = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            raise

    async def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        await self.initialize_collection(vector_size=vector_size)

    async def upsert_chunks(
        self,
        document_id: str,
        chunks: List[str],
        vectors: List[List[float]],
        additional_metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> bool:
        if len(chunks) != len(vectors):
            raise ValueError("Mismatched chunks and vectors count.")
        if not vectors:
            raise ValueError("No vectors provided for upsert.")
            
        if additional_metadata and len(additional_metadata) != len(chunks):
            raise ValueError("Mismatched metadata and chunks count.")

        vector_size = len(vectors[0])
        await self._ensure_collection(vector_size=vector_size)

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
            for attempt in range(2):
                try:
                    for i in range(0, len(points), batch_size):
                        batch = points[i:i + batch_size]
                        await self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch
                        )
                        logger.info(f"Upserted batch of {len(batch)} points for document {document_id}")
                    return True
                except UnexpectedResponse as e:
                    if getattr(e, "status_code", None) == 404 and attempt == 0:
                        logger.warning(
                            "Qdrant collection missing on upsert; recreating and retrying once."
                        )
                        self._collection_ready = False
                        await self.initialize_collection(vector_size=vector_size)
                        continue
                    raise
        except Exception as e:
            logger.error(f"Failed to upsert chunks to Qdrant: {str(e)}")
            raise

    async def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not query_vector:
            return []

        await self._ensure_collection(vector_size=len(query_vector))

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

        qdim = len(query_vector)
        try:
            for attempt in range(2):
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
                except UnexpectedResponse as e:
                    if getattr(e, "status_code", None) == 404 and attempt == 0:
                        self._collection_ready = False
                        await self.initialize_collection(vector_size=qdim)
                        continue
                    raise
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {str(e)}")
            raise
