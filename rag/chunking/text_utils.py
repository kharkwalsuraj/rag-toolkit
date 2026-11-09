"""Text helpers: span flattening, whitespace cleanup, size-bounded splitting."""

from __future__ import annotations

import math
import re
from typing import Any, List

_IGNORED_KEYS = {"bbox", "path", "image_source", "img_path", "image_path", "sub_type", "anchor"}
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ceil(chars / 4)."""
    return math.ceil(len(text) / 4)


def clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def spans_to_text(value: Any) -> str:
    """Flatten MinerU 'spans' (lists of {type, content} dicts) into a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "".join(spans_to_text(v) for v in value)
    if isinstance(value, dict):
        if "content" in value and isinstance(value.get("type"), str):
            inner = spans_to_text(value["content"])
            if value["type"] == "equation_inline" and inner:
                return f"${inner}$"
            return inner
        parts = [
            spans_to_text(v)
            for k, v in value.items()
            if k not in _IGNORED_KEYS
        ]
        return "\n".join(p for p in parts if p.strip())
    return ""


def _hard_split(text: str, max_chars: int) -> List[str]:
    pieces: List[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def split_text(text: str, max_chars: int) -> List[str]:
    """Split on line/sentence boundaries into pieces of at most ``max_chars`` (best effort)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    by_lines = "\n" in text
    joiner = "\n" if by_lines else " "
    units = text.split("\n") if by_lines else _SENTENCE_BREAK.split(text)

    expanded: List[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            expanded.extend(split_text(unit, max_chars) if by_lines else _hard_split(unit, max_chars))
        else:
            expanded.append(unit)

    pieces: List[str] = []
    current = ""
    for unit in expanded:
        if current and len(current) + len(joiner) + len(unit) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}{joiner}{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces
