# Settings for RAGPipeline. Every field has a default; pass only what you want to change.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    # MinerU (cloud parser). The token is only needed to parse new PDFs.
    mineru_token: Optional[str] = None
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_model: str = "vlm"
    mineru_language: str = "en"  # "ch" for Chinese documents
    mineru_ocr: bool = False  # True for scanned PDFs
    mineru_poll_interval: float = 5.0
    mineru_timeout: float = 1800.0

    # Where parsed MinerU results are cached
    data_dir: Path = Path(".rag_data")

    # Embeddings (local)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: Optional[str] = None  # None = pick from the model name
    embedding_device: Optional[str] = None  # "cpu", "cuda", "mps" or None for auto
    embedding_batch_size: int = 32

    # Reranker (local)
    reranker_model: str = "mixedbread-ai/mxbai-rerank-base-v1"
    reranker_device: Optional[str] = None
    reranker_batch_size: int = 16
    rerank_candidates: int = 30  # vector hits that are rescored before cutting to top_k

    # Qdrant. If qdrant_url is set it is used; otherwise qdrant_path (on disk);
    # set both to None for an in-memory database.
    qdrant_path: Optional[str] = ".rag_data/qdrant"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "documents"

    # Gemini (cloud). The key is only needed for ask() and describe_figures().
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2

    # Chunking (the embedding and reranker models read at most about 512 tokens)
    chunk_text_tokens: int = 400
    chunk_table_tokens: int = 300

    @property
    def parsed_dir(self) -> Path:
        return Path(self.data_dir) / "parsed"
