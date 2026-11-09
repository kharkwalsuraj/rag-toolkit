"""Load an existing MinerU result directory (no network, no token needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from ..exceptions import ParsingError
from ..models import ParsedDocument
from ..utils import slugify


def load_mineru_output(path: Union[str, Path], doc_id: Optional[str] = None) -> ParsedDocument:
    """Load a directory (or the ``content_list_v2.json`` file itself) produced by MinerU."""
    path = Path(path)
    if path.is_file():
        content_file = path
    else:
        candidates = sorted(path.rglob("*content_list_v2.json"))
        if len(candidates) != 1:
            raise ParsingError(
                f"Expected exactly one *content_list_v2.json under {path}, found {len(candidates)}"
            )
        content_file = candidates[0]

    try:
        content_list = json.loads(content_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParsingError(f"Cannot read {content_file}: {exc}") from exc

    root = content_file.parent
    markdown_files = sorted(root.glob("full.md")) or sorted(root.glob("*.md"))
    markdown = markdown_files[0] if markdown_files else None
    return ParsedDocument(
        doc_id=doc_id or slugify(root.name if path.is_file() else path.name),
        root=root,
        content_list=content_list,
        markdown_path=markdown,
    )
