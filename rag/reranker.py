# Cross-encoder that rescores the dense retrieval hits against the query.

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional, Sequence

from .exceptions import RAGError
from .schemas import SearchResult

log = logging.getLogger("rag.rerank")

DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"


class LocalReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # slow import, so done on first use

            log.info("Loading reranker model %s", self.model_name)
            try:
                self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            except Exception as exc:
                raise RAGError(f"Could not load reranker model {self.model_name!r}: {exc}") from exc
        return self._model

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.chunk.embedding_text) for r in results]
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:
            raise RAGError(f"Reranking failed: {exc}") from exc
        ranked = sorted(zip(results, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [replace(r, score=float(s), rerank_score=float(s)) for r, s in ranked[:top_k]]
