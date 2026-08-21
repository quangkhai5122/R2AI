"""Optional BGE-M3 matching between metric phrases and financial row labels.

The deduplicated label vocabulary is embedded once and stored inside the table
store. Retrieval then encodes only the metric variants for each question. All
callers must retain a lexical fallback when the optional model is unavailable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"
INDEX_SCHEMA_VERSION = 1
_METADATA_NAME = "index-metadata.json"


class LabelEncoder:
    """Cosine similarity between metric phrases and a cached label vocabulary."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None,
                 device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 64
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.vocab: list[str] = []
        self.matrix: np.ndarray | None = None
        self._index: dict[str, int] = {}
        self.metadata: dict = {}
        if self.cache_dir and (self.cache_dir / "labels.json").exists():
            self.load_cache(self.cache_dir)

    def build_cache(self, labels, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vocab = sorted({str(label).strip() for label in labels if str(label).strip()})
        emb = self.model.encode(
            vocab,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        matrix = np.asarray(emb, dtype=np.float32)
        labels_text = json.dumps(vocab, ensure_ascii=False)
        np.save(cache_dir / "labels.npy", matrix.astype(np.float16))
        (cache_dir / "labels.json").write_text(labels_text, encoding="utf-8")
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "model_name": self.model_name,
            "vocab_size": len(vocab),
            "dimension": int(matrix.shape[1]) if matrix.ndim == 2 and len(matrix) else 0,
            "normalized": True,
            "storage_dtype": "float16",
            "labels_sha256": hashlib.sha256(labels_text.encode("utf-8")).hexdigest(),
        }
        (cache_dir / _METADATA_NAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.cache_dir = cache_dir
        self.vocab, self.matrix, self.metadata = vocab, matrix, metadata
        self._index = {label: i for i, label in enumerate(vocab)}
        print(f"[dense] cached {len(vocab)} labels -> {cache_dir}")

    def load_cache(self, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        labels_path = cache_dir / "labels.json"
        labels_text = labels_path.read_text(encoding="utf-8")
        vocab = json.loads(labels_text)
        matrix = np.load(cache_dir / "labels.npy").astype(np.float32)
        metadata_path = cache_dir / _METADATA_NAME
        metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.exists() else {})
        if matrix.ndim != 2 or matrix.shape[0] != len(vocab):
            raise ValueError("dense label cache shape does not match labels.json")
        cached_model = str(metadata.get("model_name", ""))
        if cached_model and cached_model != self.model_name:
            raise ValueError(
                f"dense cache model mismatch: cache={cached_model}, runtime={self.model_name}"
            )
        expected_hash = str(metadata.get("labels_sha256", ""))
        actual_hash = hashlib.sha256(labels_text.encode("utf-8")).hexdigest()
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("dense labels cache hash mismatch")
        self.cache_dir = cache_dir
        self.vocab, self.matrix, self.metadata = vocab, matrix, metadata
        self._index = {label: i for i, label in enumerate(vocab)}
        print(f"[dense] loaded {len(self.vocab)} cached label vectors")

    def describe(self) -> dict:
        return {
            "enabled": True,
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir) if self.cache_dir else "",
            "vocab_size": len(self.vocab),
            "index_schema": self.metadata.get("schema_version"),
            "labels_sha256": self.metadata.get("labels_sha256", ""),
        }

    def similarity(self, queries: list[str], labels: list[str]) -> dict[str, float]:
        """Return each label's best cosine similarity against any query."""
        if not queries or not labels:
            return {}
        q = self.model.encode(
            list(queries), batch_size=self.batch_size, normalize_embeddings=True
        )
        q = np.asarray(q, dtype=np.float32)

        known = [label for label in labels if label in self._index]
        unknown = [label for label in labels if label not in self._index]
        out: dict[str, float] = {}
        if known and self.matrix is not None:
            m = self.matrix[[self._index[label] for label in known]]
            sims = (m @ q.T).max(axis=1)
            out.update({label: float(score) for label, score in zip(known, sims)})
        if unknown:
            e = np.asarray(
                self.model.encode(
                    unknown,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            sims = (e @ q.T).max(axis=1)
            out.update({label: float(score) for label, score in zip(unknown, sims)})
        return out


def load_encoder(model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None,
                 device: str | None = None, quiet: bool = False):
    """Return a LabelEncoder, or None when the optional dependency is unavailable."""
    try:
        return LabelEncoder(model_name, cache_dir, device)
    except Exception as exc:  # noqa: BLE001 - optional dependency by design
        if not quiet:
            print(f"[dense] disabled ({type(exc).__name__}: {exc}); using lexical matching")
        return None


def collect_labels(store, tickers=None, max_per_ticker: int = 0) -> list[str]:
    """Collect the distinct row-label vocabulary across the store."""
    from .serialize import grid_of

    tickers = tickers or sorted({ticker for (ticker, _year, _doc) in store.report_index})
    seen: set[str] = set()
    for ticker in tickers:
        df = store._load("tables", ticker)
        if not len(df):
            continue
        rows = df.to_dict("records")
        if max_per_ticker:
            rows = rows[:max_per_ticker]
        for meta in rows:
            for row in grid_of(meta)[1:]:
                for cell in row[:3]:
                    cell = (cell or "").strip()
                    if len(cell) > 3 and not cell.replace(".", "").replace(",", "").isdigit():
                        seen.add(cell[:160])
                        break
    return sorted(seen)
