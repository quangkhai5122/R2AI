"""Shared deterministic I/O and hashing helpers for G3B."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..g3a.common import (
    canonical_sha256,
    file_sha256,
    normalize_question,
    read_jsonl,
    write_json,
    write_jsonl,
)


def tree_rows(root: Path | str) -> list[dict]:
    root = Path(root)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def tree_sha256(root: Path | str) -> str:
    return canonical_sha256(tree_rows(root))


def atomic_text(path: Path | str, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
