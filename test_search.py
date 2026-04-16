import asyncio
from app.services.embeddings import get_embedder
from app.db.vector_store import QdrantService
async def main():
    try:
        embedder = get_embedder('huggingface')
        vector = embedder.embed_query('What is the tech stack used by the engineering team?')
        qdrant = QdrantService()
        res = await qdrant.search_similar(vector, limit=8)
        for r in res:
            print('Score:', r.get('score'), 'Text:', r.get('text')[:50])
    except Exception as e:
        print('Error:', e)
asyncio.run(main())
