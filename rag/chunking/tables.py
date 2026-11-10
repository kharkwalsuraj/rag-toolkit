from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

from .text_utils import clean


@dataclass
class _Cell:
    text: str
    rowspan: int
    colspan: int
    header: bool


def _span(attrs: Dict[str, Optional[str]], name: str) -> int:
    try:
        return max(1, min(int(str(attrs.get(name) or 1).strip()), 100))
    except ValueError:
        return 1


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[_Cell]] = []
        self._row: Optional[List[_Cell]] = None
        self._cell: Optional[Tuple[bool, int, int]] = None  # (header, rowspan, colspan)
        self._buf: List[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth == 1:
            self._end_row()
            self._row = []
        elif tag in ("td", "th") and self._depth == 1:
            self._end_cell()
            if self._row is None:
                self._row = []
            self._cell = (tag == "th", _span(attrs, "rowspan"), _span(attrs, "colspan"))
            self._buf = []
        elif tag == "br" and self._cell is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._depth == 1:
            self._end_cell()
        elif tag == "tr" and self._depth == 1:
            self._end_row()
        elif tag == "table":
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._cell is not None:
            self._buf.append(data)

    def _end_cell(self):
        if self._cell is not None and self._row is not None:
            header, rowspan, colspan = self._cell
            self._row.append(_Cell(re.sub(r"\s+", " ", "".join(self._buf)).strip(), rowspan, colspan, header))
        self._cell, self._buf = None, []

    def _end_row(self):
        self._end_cell()
        if self._row:
            self.rows.append(self._row)
        self._row = None

    def finish(self):
        self._end_row()


# fill rowspan and colspan into a rectangular grid (spanned cells repeat their text)
def _expand(rows: List[List[_Cell]]) -> List[List[str]]:
    carry: Dict[int, List] = {}  # column -> [text, remaining_rows]
    grid: List[List[str]] = []
    for cells in rows:
        out: List[str] = []
        col = 0

        def drain() -> None:
            nonlocal col
            while col in carry:
                text, remaining = carry[col]
                out.append(text)
                if remaining <= 1:
                    del carry[col]
                else:
                    carry[col][1] = remaining - 1
                col += 1

        for cell in cells:
            drain()
            for _ in range(cell.colspan):
                out.append(cell.text)
                if cell.rowspan > 1:
                    carry[col] = [cell.text, cell.rowspan - 1]
                col += 1
        drain()
        grid.append(out)
    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]


def parse_html_table(html: str) -> Tuple[List[List[str]], int]:
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    parser.finish()
    if not parser.rows:
        return [], 0
    header_rows = 0
    for row in parser.rows:
        if row and all(c.header for c in row):
            header_rows += 1
        else:
            break
    return _expand(parser.rows), max(1, header_rows)


def html_to_text(html: str) -> str:
    return clean(html_lib.unescape(re.sub(r"<[^>]+>", " ", html)))


def _merge_headers(header_rows: List[List[str]]) -> List[str]:
    merged: List[str] = []
    for col in range(len(header_rows[0])):
        seen: List[str] = []
        for row in header_rows:
            value = row[col]
            if value and value not in seen:
                seen.append(value)
        merged.append(" / ".join(seen))
    return merged


def _md_row(cells: List[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


# render as Markdown; a long table is split by rows and every piece repeats the caption and header
def table_to_chunks(
    html: str,
    *,
    caption: str = "",
    footnote: str = "",
    max_chars: int = 3200,
) -> List[str]:
    grid, n_header = parse_html_table(html)
    if not grid:
        return []
    header = _merge_headers(grid[:n_header])
    body = grid[n_header:]
    head_lines = [_md_row(header), _md_row(["---"] * len(header))]
    prefix = f"{caption}\n\n" if caption else ""
    suffix = f"\n\n{footnote}" if footnote else ""

    def assemble(lines: List[str]) -> str:
        return prefix + "\n".join(head_lines + lines)

    if not body:
        return [assemble([]) + suffix]

    base = len(prefix) + sum(len(line) + 1 for line in head_lines)
    pieces: List[str] = []
    current: List[str] = []
    size = base
    for row in body:
        line = _md_row(row)
        if current and size + len(line) + 1 > max_chars:
            pieces.append(assemble(current))
            current, size = [], base
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append(assemble(current))
    pieces[-1] += suffix
    return pieces
