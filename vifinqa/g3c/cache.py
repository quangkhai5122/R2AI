"""Content-addressed, crash-safe caches for G3C GPU work."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .common import (
    canonical_json_sha256,
    read_json,
    write_json,
)


def vector_cache_key(
    *, contract_fingerprint: str, backend: str,
    model_revision: str, kind: str,
    instruction: str, content: str,
) -> str:
    return canonical_json_sha256({
        "contract_fingerprint": contract_fingerprint,
        "backend": backend,
        "model_revision": model_revision,
        "kind": kind,
        "instruction": instruction,
        "content": content,
    })


def score_cache_key(
    *, contract_fingerprint: str, backend: str,
    model_revision: str, kind: str,
    instruction: str, query: str, document: str,
) -> str:
    return canonical_json_sha256({
        "contract_fingerprint": contract_fingerprint,
        "backend": backend,
        "model_revision": model_revision,
        "kind": kind,
        "instruction": instruction,
        "query": query,
        "document": document,
    })


class VectorCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.values: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as archive:
                self.values = {
                    key: archive[key] for key in archive.files
                }

    def get(self, key: str) -> np.ndarray | None:
        return self.values.get(key)

    def put(self, key: str, value: np.ndarray) -> None:
        vector = np.asarray(value)
        if vector.ndim != 1:
            raise ValueError("vector cache accepts one-dimensional values")
        if not np.isfinite(vector).all():
            raise ValueError("vector cache rejects non-finite values")
        self.values[key] = vector.astype(np.float16)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=f".{self.path.stem}.",
            suffix=".npz",
        )
        os.close(descriptor)
        try:
            np.savez_compressed(temp_name, **self.values)
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


class ScoreCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.values: dict[str, float] = {}
        if self.path.exists():
            payload = read_json(self.path)
            if payload.get("schema_version") != "g3c_score_cache_v1":
                raise ValueError("unknown G3C score-cache schema")
            self.values = {
                str(key): float(value)
                for key, value in payload.get("scores", {}).items()
            }

    def get(self, key: str) -> float | None:
        return self.values.get(key)

    def put(self, key: str, value: float) -> None:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("score cache rejects non-finite values")
        self.values[key] = number

    def save(self) -> None:
        write_json(self.path, {
            "schema_version": "g3c_score_cache_v1",
            "scores": self.values,
        })
