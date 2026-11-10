from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from ..exceptions import RAGError
from ..schemas import Chunk, ParsedDocument
from . import blocks as B
from .tables import html_to_text, table_to_chunks
from .text_utils import clean, split_text

log = logging.getLogger("rag.chunking")


@dataclass(frozen=True)
class ChunkerConfig:
    text_max_tokens: int = 400  # merge small paragraphs up to this size
    table_max_tokens: int = 300  # long tables are split by rows beyond this (models read ~512 tokens)
    context_chars: int = 300  # nearby text attached to figures/formulas (0 = off)
    section_path: bool = True  # section titles as a path: "Chapter > Section"
    skip_sections: Tuple[str, ...] = (
        r"^(table of )?contents$",
        r"^(revision|document|change) history$",
    )

    @property
    def text_max_chars(self) -> int:
        return self.text_max_tokens * 4

    @property
    def table_max_chars(self) -> int:
        return self.table_max_tokens * 4


class Chunker:
    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(
        self,
        content_list: list,
        *,
        doc_id: str,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> List[Chunk]:
        if not isinstance(content_list, list):
            raise RAGError("content_list must be a list (expected MinerU content_list_v2.json)")
        if content_list and not all(isinstance(page, list) for page in content_list):
            raise RAGError(
                "Unexpected structure: content_list_v2 is a list of pages, each a list of blocks."
            )
        run = _Run(doc_id, self.config, Path(asset_root) if asset_root else None)
        for page_index, page in enumerate(content_list):
            run.set_page(page_index + 1)
            for block in page:
                run.handle(block)
        run.flush()
        return run.chunks


def chunk_content_list(
    content_list: list,
    doc_id: str,
    asset_root: Optional[Union[str, Path]] = None,
    config: Optional[ChunkerConfig] = None,
) -> List[Chunk]:
    return Chunker(config).chunk(content_list, doc_id=doc_id, asset_root=asset_root)


def chunk_parsed(parsed: ParsedDocument, config: Optional[ChunkerConfig] = None) -> List[Chunk]:
    return Chunker(config).chunk(parsed.content_list, doc_id=parsed.doc_id, asset_root=parsed.root)


# state for one chunking pass
class _Run:
    def __init__(self, doc_id: str, cfg: ChunkerConfig, asset_root: Optional[Path]) -> None:
        self.doc_id = doc_id
        self.cfg = cfg
        self.asset_root = asset_root
        self.chunks: List[Chunk] = []
        self.page = 1
        self._counter = 0
        self._stack: List[Tuple[int, str]] = []
        self._skipping = False
        self._buffer: List[str] = []
        self._buffer_chars = 0
        self._recent = ""  # last paragraph, used as context for figures/formulas
        self._skip_patterns = [re.compile(p, re.I) for p in cfg.skip_sections]
        self._warned_types: set = set()
        self._handlers = {
            "title": self._title,
            "paragraph": self._paragraph,
            "text": self._paragraph,
            "list": self._list,
            "table": self._table,
            "image": self._figure,
            "figure": self._figure,
            "chart": self._figure,
            "equation_interline": self._formula,
            "equation": self._formula,
            "code": self._code,
            "algorithm": self._code,
            "page_footnote": self._footnote,
        }

    # driver
    def set_page(self, page: int) -> None:
        self.flush()  # text is merged only within one page and section
        self.page = page

    def handle(self, block: Any) -> None:
        btype = B.block_type(block)
        if not btype or btype in B.SKIP_TYPES:
            return
        content = block.get("content")
        if btype == "title":
            self._title(content)
            return
        if self._skipping:
            return
        handler = self._handlers.get(btype)
        if handler is None:
            if btype not in self._warned_types:
                log.warning("Unknown block type %r: falling back to plain text", btype)
                self._warned_types.add(btype)
            handler = self._paragraph
        handler(content)

    # section tracking
    @property
    def section(self) -> str:
        if not self._stack:
            return ""
        if self.cfg.section_path:
            return " > ".join(title for _, title in self._stack)
        return self._stack[-1][1]

    def _title(self, content: Any) -> None:
        self.flush()
        text, level = B.title_info(content)
        if not text:
            return
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))
        self._skipping = any(p.search(text) for p in self._skip_patterns)
        self._recent = ""

    # text-like blocks
    def _paragraph(self, content: Any) -> None:
        text = B.paragraph_text(content)
        if text:
            self._add_text(text)
            self._recent = text

    def _footnote(self, content: Any) -> None:
        text = B.paragraph_text(content)
        if text:
            self._add_text(text)

    def _list(self, content: Any) -> None:
        text = B.list_text(content)
        if text:
            self._add_text(text)
            self._recent = text

    def _code(self, content: Any) -> None:
        body, language = B.code_text(content)
        if not body.strip():
            return
        self.flush()
        caption = B.caption_text(content)
        text = f"{caption}\n```{language}\n{body}\n```" if caption else f"```{language}\n{body}\n```"
        for piece in split_text(text, self.cfg.text_max_chars):
            self._emit("text", piece)

    def _add_text(self, text: str) -> None:
        limit = self.cfg.text_max_chars
        if len(text) > limit:
            self.flush()
            for piece in split_text(text, limit):
                self._emit("text", piece)
            return
        if self._buffer and self._buffer_chars + len(text) + 2 > limit:
            self.flush()
        self._buffer.append(text)
        self._buffer_chars += len(text) + 2

    def flush(self) -> None:
        if not self._buffer:
            return
        text = "\n\n".join(self._buffer)
        self._buffer, self._buffer_chars = [], 0
        self._emit("text", text)

    # structured blocks
    def _table(self, content: Any) -> None:
        self.flush()
        caption = B.caption_text(content)
        footnote = B.footnote_text(content)
        html = B.table_html(content)
        pieces = (
            table_to_chunks(html, caption=caption, footnote=footnote, max_chars=self.cfg.table_max_chars)
            if html
            else []
        )
        if not pieces:  # not parseable as a table: keep whatever text is available
            fallback = "\n\n".join(p for p in (caption, html_to_text(html) if html else "", footnote) if p)
            pieces = [fallback] if fallback else []
        image = B.image_rel_path(content)
        for piece in pieces:
            self._emit("table", piece, image)

    def _figure(self, content: Any) -> None:
        self.flush()
        lines = [t for t in (B.caption_text(content), B.footnote_text(content)) if t]
        context = self._recent[-self.cfg.context_chars:] if self.cfg.context_chars else ""
        if context:
            lines.append(f"Nearby text: {context}")
        if not lines:  # never drop a figure, even with no caption
            where = f" in section '{self.section}'" if self.section else ""
            lines.append(f"Figure on page {self.page}{where}")
        self._emit("figure", "\n".join(lines), B.image_rel_path(content))

    def _formula(self, content: Any) -> None:
        latex = B.formula_latex(content)
        if not latex:
            return
        self.flush()
        text = f"$$ {latex} $$"
        context = self._recent[-self.cfg.context_chars:] if self.cfg.context_chars else ""
        self._emit("formula", f"{context}\n\n{text}" if context else text, B.image_rel_path(content))

    # output
    def _emit(self, content_type: str, text: str, image: Optional[str] = None) -> None:
        text = clean(text) if content_type in ("figure", "formula") else text.strip()
        if not text:
            return
        self._counter += 1
        image_path = None
        if image:
            image_path = str(self.asset_root / image) if self.asset_root else image
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.doc_id}:p{self.page}:{content_type}:{self._counter}",
                doc_id=self.doc_id,
                page_no=self.page,
                content_type=content_type,
                section_title=self.section,
                text=text,
                image_path=image_path,
            )
        )
