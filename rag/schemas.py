from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int
    content_type: str  # text, table, figure or formula
    section_title: str
    text: str
    image_path: Optional[str] = None

    # the section title gives the passage some context
    @property
    def embedding_text(self) -> str:
        return f"{self.section_title}\n\n{self.text}" if self.section_title else self.text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data.get("doc_id", ""),
            page_no=int(data.get("page_no", 0)),
            content_type=data.get("content_type", "text"),
            section_title=data.get("section_title", ""),
            text=data.get("text", ""),
            image_path=data.get("image_path") or None,
        )


# score is what results are ordered by (the rerank score); vector_score and rerank_score keep both parts
@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Answer:
    text: str
    sources: List[SearchResult] = field(default_factory=list)


@dataclass
class IngestResult:
    doc_id: str
    num_chunks: int
    by_type: Dict[str, int] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    doc_id: str
    root: Path  # directory holding content_list_v2.json and images/
    content_list: list
    markdown_path: Optional[Path] = None
