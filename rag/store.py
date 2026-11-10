# Qdrant. Local path mode allows one process at a time; use a server url to share a collection.

from __future__ import annotations

import logging
import uuid
from typing import List, Optional, Sequence

from qdrant_client import QdrantClient, models

from .exceptions import RAGError
from .schemas import Chunk, SearchResult

log = logging.getLogger("rag.vectorstore")

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-toolkit")
_BATCH = 128


# the same chunk id always maps to the same point, so ingesting again overwrites instead of duplicating
def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantStore:
    def __init__(
        self,
        *,
        collection: str = "documents",
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:

        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

    def ensure_collection(self, dim: int) -> None:

        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
                for field in ("doc_id", "content_type"):
                    try:
                        self.client.create_payload_index(
                            self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                        )
                    except Exception:  # local mode ignores payload indexes
                        log.debug("payload index on %s not created", field)
                return
            existing = self.client.get_collection(self.collection).config.params.vectors
            size = getattr(existing, "size", None)
            if size is not None and size != dim:
                raise RAGError(
                    f"Collection '{self.collection}' has dimension {size}, but the embedding model "
                    f"produces {dim}. Use a different collection name or re-ingest."
                )
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"Could not prepare collection: {exc}") from exc

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:

        if len(chunks) != len(vectors):
            raise RAGError("chunks and vectors must have the same length")
        try:
            for start in range(0, len(chunks), _BATCH):
                batch = [
                    models.PointStruct(id=point_id(c.chunk_id), vector=list(v), payload=c.to_dict())
                    for c, v in zip(chunks[start:start + _BATCH], vectors[start:start + _BATCH])
                ]
                self.client.upsert(collection_name=self.collection, points=batch)
        except Exception as exc:
            raise RAGError(f"Upsert failed: {exc}") from exc

    def dense_search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        content_types: Optional[Sequence[str]] = None,
    ) -> List[SearchResult]:

        must = []
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
        if content_types:
            must.append(models.FieldCondition(key="content_type", match=models.MatchAny(any=list(content_types))))
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                limit=top_k,
                query_filter=models.Filter(must=must) if must else None,
                with_payload=True,
            )
        except Exception as exc:
            raise RAGError(f"Search failed: {exc}") from exc
        return [
            SearchResult(Chunk.from_dict(p.payload), float(p.score), vector_score=float(p.score))
            for p in response.points
            if p.payload
        ]

    def delete_document(self, doc_id: str) -> None:

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                    )
                ),
            )
        except Exception as exc:
            raise RAGError(f"Delete failed: {exc}") from exc

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)
