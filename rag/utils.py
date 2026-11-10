from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Union


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:max_len].strip("-")
    return slug or "doc"


def file_sha256(path: Union[str, Path], block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


# file name plus content hash, so the same PDF always gets the same id
def make_doc_id(path: Union[str, Path]) -> str:
    p = Path(path)
    return f"{slugify(p.stem)}_{file_sha256(p)[:8]}"
