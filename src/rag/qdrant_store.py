from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from .documents import KnowledgeChunk
from .embeddings import GeminiEmbeddings


COLLECTION = "zentromart_kb"


class LocalQdrantStore:
    def __init__(self, path: Path, embeddings: GeminiEmbeddings) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install requirements.txt first."
            ) from exc
        self.client = QdrantClient(path=str(path))
        self.embeddings = embeddings

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        if not vectors:
            raise ValueError("Cannot index an empty knowledge base")
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{chunk.source_file}:{chunk.section_id}")),
                vector=vector,
                payload={"text": chunk.text, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search(self, query: str, limit: int = 16) -> list[dict]:
        if not self.client.collection_exists(COLLECTION):
            raise RuntimeError(
                "Qdrant collection is missing. Build it with --rebuild-index."
            )
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=self.embeddings.embed_query(query),
            limit=limit,
            with_payload=True,
        ).points
        return [
            {"score": hit.score, **(hit.payload or {})}
            for hit in hits
        ]
