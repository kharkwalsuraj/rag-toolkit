"""Readers for individual MinerU ``content_list_v2`` blocks.

All readers are deliberately tolerant: MinerU's schema has shifted between versions, so
they look up several key names and fall back to generic text extraction.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .text_utils import clean, spans_to_text

# Structural page furniture MinerU labels explicitly; never indexed.
SKIP_TYPES = frozenset({"page_header", "page_footer", "page_number", "page_aside_text"})


def block_type(block: Any) -> str:
    return str(block.get("type", "")).lower() if isinstance(block, dict) else ""


def get(content: Any, *keys: str) -> Any:
    if not isinstance(content, dict):
        return None
    for key in keys:
        value = content.get(key)
        if value not in (None, "", []):
            return value
    return None


def any_text(content: Any) -> str:
    return clean(spans_to_text(content))


def _join_suffix(content: Any, suffix: str) -> str:
    if not isinstance(content, dict):
        return ""
    parts = [spans_to_text(v) for k, v in content.items() if k.endswith(suffix)]
    return clean("\n".join(p for p in parts if p.strip()))


def caption_text(content: Any) -> str:
    return _join_suffix(content, "_caption")


def footnote_text(content: Any) -> str:
    return _join_suffix(content, "_footnote")


def image_rel_path(content: Any) -> Optional[str]:
    src = get(content, "image_source", "img_path", "image_path")
    if isinstance(src, dict):
        src = src.get("path")
    return str(src) if src else None


def paragraph_text(content: Any) -> str:
    return clean(spans_to_text(get(content, "paragraph_content")) or any_text(content))


def title_info(content: Any) -> Tuple[str, int]:
    text = clean(spans_to_text(get(content, "title_content")) or any_text(content))
    try:
        level = int(get(content, "level") or 1)
    except (TypeError, ValueError):
        level = 1
    return text, max(1, level)


def list_text(content: Any) -> str:
    items = get(content, "list_items") or []
    ordered = str(get(content, "list_type") or "").startswith("ordered")
    lines = []
    for i, item in enumerate(items, 1):
        raw = get(item, "item_content") if isinstance(item, dict) else item
        text = clean(spans_to_text(raw if raw is not None else item))
        if text:
            lines.append(f"{i}. {text}" if ordered else f"- {text}")
    return "\n".join(lines)


def table_html(content: Any) -> str:
    return str(get(content, "html", "table_body") or "")


def code_text(content: Any) -> Tuple[str, str]:
    body = spans_to_text(get(content, "code_content", "algorithm_content"))
    language = str(get(content, "code_language", "language") or "")
    return body.strip("\n"), language


def formula_latex(content: Any) -> str:
    return clean(str(get(content, "math_content", "latex") or ""))
