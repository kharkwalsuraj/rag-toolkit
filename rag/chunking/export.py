"""JSONL import/export for chunks (handy for debugging and for non-Qdrant pipelines)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union

from ..models import Chunk


def write_jsonl(chunks: Iterable[Chunk], path: Union[str, Path]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Union[str, Path]) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as fh:
        return [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
