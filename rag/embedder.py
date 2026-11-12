from __future__ import annotations

import logging
from collections.abc import Sequence


log = logging.getLogger("rag.embeddings")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def default_query_prefix(model_name: str) -> str:
    name = model_name.lower()
    return BGE_QUERY_PREFIX if "bge" in name and "m3" not in name else ""


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = default_query_prefix(model_name) if query_prefix is None else query_prefix
        self._model = None
        self._dim: int | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(f"Could not load embedding model {self.model_name!r}: {exc}") from exc
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_documents(["dimension probe"])[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {exc}") from exc
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.query_prefix + text])[0]
