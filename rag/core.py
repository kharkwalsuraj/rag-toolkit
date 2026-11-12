from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    workspace: Path = Path("rag_workspace")

    mineru_token: str | None = None
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_model: str = "vlm"
    mineru_language: str = "en"
    mineru_ocr: bool = False
    mineru_poll_interval: float = 5.0
    mineru_timeout: float = 1800.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str | None = None
    embedding_device: str | None = None
    embedding_batch_size: int = 32

    reranker_model: str = "mixedbread-ai/mxbai-rerank-base-v1"
    reranker_device: str | None = None
    reranker_batch_size: int = 16
    rerank_candidates: int = 30

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    describe_figures: bool = False

    chunk_text_tokens: int = 400
    chunk_table_tokens: int = 300

    @property
    def mineru_dir(self) -> Path:
        return Path(self.workspace) / "mineru"

    @property
    def qdrant_dir(self) -> Path:
        return Path(self.workspace) / "qdrant"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int
    content_type: str
    section_title: str
    text: str
    image_path: str | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.section_title}\n\n{self.text}" if self.section_title else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data.get("doc_id", ""),
            page_no=int(data.get("page_no", 0)),
            content_type=data.get("content_type", "text"),
            section_title=data.get("section_title", ""),
            text=data.get("text", ""),
            image_path=data.get("image_path") or None,
        )


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Answer:
    text: str
    sources: list[SearchResult] = field(default_factory=list)


@dataclass
class IngestResult:
    pdf: str
    doc_id: str
    num_chunks: int
    skipped: bool = False
    mineru_dir: Path | None = None


@dataclass
class ParsedDocument:
    doc_id: str
    root: Path
    content_list: list
    markdown_path: Path | None = None
