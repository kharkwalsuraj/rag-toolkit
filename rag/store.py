from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from .core import Chunk, SearchResult

log = logging.getLogger("rag.store")

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-toolkit")
BATCH_SIZE = 128


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def in_document(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


class QdrantStore:
    def __init__(self, collection: str, path: str, url: str | None = None, api_key: str | None = None) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(path=path)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        try:
            self._ensure_collection(len(vectors[0]))
            for start in range(0, len(chunks), BATCH_SIZE):
                points = [
                    models.PointStruct(id=point_id(c.chunk_id), vector=list(v), payload=c.to_dict())
                    for c, v in zip(chunks[start:start + BATCH_SIZE], vectors[start:start + BATCH_SIZE])
                ]
                self.client.upsert(collection_name=self.collection, points=points)
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Upsert failed: {exc}") from exc

    def dense_search(
        self, vector: Sequence[float], top_k: int = 5, doc_ids: Sequence[str] | None = None
    ) -> list[SearchResult]:
        only = None
        if doc_ids:
            only = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))])
        try:
            if not self.client.collection_exists(self.collection):
                return []
            response = self.client.query_points(
                collection_name=self.collection, query=list(vector), limit=top_k, query_filter=only, with_payload=True
            )
        except Exception as exc:
            raise RuntimeError(f"Search failed: {exc}") from exc
        return [
            SearchResult(Chunk.from_dict(p.payload), float(p.score), vector_score=float(p.score))
            for p in response.points
            if p.payload
        ]

    def delete_document(self, doc_id: str) -> None:
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete(
                    collection_name=self.collection, points_selector=models.FilterSelector(filter=in_document(doc_id))
                )
        except Exception as exc:
            raise RuntimeError(f"Delete failed: {exc}") from exc

    def count_document(self, doc_id: str) -> int:
        try:
            if not self.client.collection_exists(self.collection):
                return 0
            return int(self.client.count(collection_name=self.collection, count_filter=in_document(doc_id), exact=True).count)
        except Exception as exc:
            raise RuntimeError(f"Count failed: {exc}") from exc

    def document_chunks(self, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        try:
            if not self.client.collection_exists(self.collection):
                return chunks
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection, scroll_filter=in_document(doc_id),
                    limit=256, offset=offset, with_payload=True, with_vectors=False,
                )
                chunks += [Chunk.from_dict(p.payload) for p in points if p.payload]
                if offset is None:
                    break
        except Exception as exc:
            raise RuntimeError(f"Reading chunks failed: {exc}") from exc
        return sorted(chunks, key=lambda c: int(c.chunk_id.rsplit(":", 1)[1]))

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def _ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            try:
                self.client.create_payload_index(
                    self.collection, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception:
                log.debug("payload index not created")
            return
        existing = self.client.get_collection(self.collection).config.params.vectors
        size = getattr(existing, "size", None)
        if size is not None and size != dim:
            raise ValueError(
                f"Collection '{self.collection}' has dimension {size}, but the embedding model "
                f"produces {dim}. Use a different collection name or overwrite the documents."
            )
