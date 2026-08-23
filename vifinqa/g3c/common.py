"""Shared G3C configuration, hashing, and atomic I/O helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Iterable

G3C_SCHEMA = "g3c_qwen_retrieval_v1"
GPU_PAYLOAD_SCHEMA = "g3c_gpu_payload_v2"
GPU_RESULT_SCHEMA = "g3c_gpu_result_v1"
CANDIDATE_FREEZE_SCHEMA = "g3c_candidate_freeze_v1"
EXPECTED_G3_FREEZE = (
    "242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877"
)
STAGES = ("R0", "R0L", "R1", "R2", "R3", "R4")
EXPECTED_MODELS = {
    "embedding": "Qwen/Qwen3-Embedding-4B",
    "reranker": "Qwen/Qwen3-Reranker-4B",
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path | str, value: object) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_replace(Path(path), payload)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        .encode("utf-8") + b"\n"
        for row in rows
    )
    _atomic_replace(Path(path), payload)


def load_config(path: Path | str) -> dict:
    config = read_json(path)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("schema_version") != G3C_SCHEMA:
        raise ValueError(f"expected schema_version={G3C_SCHEMA}")
    if config.get("g3_evaluation_freeze_sha256") != EXPECTED_G3_FREEZE:
        raise ValueError("G3C config is not bound to the frozen G3 evaluator")
    if tuple(config.get("retrieval", {}).get("stages", ())) != STAGES:
        raise ValueError(f"G3C stage order must be {STAGES}")
    if int(config["retrieval"].get("submission_k", 0)) != 5:
        raise ValueError("G3C contract requires submission_k=5")
    runtime = config.get("runtime", {})
    if runtime.get("precision") != "float16":
        raise ValueError("initial G3C ablation requires float16")
    if runtime.get("quantization") != "none":
        raise ValueError("quantization must be a separate ablation")
    if runtime.get("attention_implementation") != "sdpa":
        raise ValueError("portable G3C v1 runtime requires SDPA")
    if runtime.get("sequential_model_loading") is not True:
        raise ValueError("embedding and reranker must load sequentially")
    length_keys = (
        "query_max_length", "table_max_length",
        "reranker_max_length", "row_max_length",
    )
    if any(int(runtime.get(key, 0)) < 64 for key in length_keys):
        raise ValueError("G3C runtime token limits must be at least 64")
    if int(runtime["row_max_length"]) > int(
        runtime["reranker_max_length"]
    ):
        raise ValueError(
            "row max length cannot exceed table reranker length"
        )
    if any(
        int(runtime.get(key, 0)) < 1
        for key in ("embedding_batch_size", "reranker_batch_size")
    ):
        raise ValueError("G3C runtime batch sizes must be positive")
    for name in ("embedding", "reranker"):
        model = config.get("models", {}).get(name, {})
        if model.get("model_id") != EXPECTED_MODELS[name]:
            raise ValueError(f"{name} model ID drifted from G3C v1")
        for field in ("revision", "tokenizer_revision"):
            if not _HEX40.fullmatch(str(model.get(field, ""))):
                raise ValueError(f"{name}.{field} must be an immutable commit SHA")
        if model["tokenizer_revision"] != model["revision"]:
            raise ValueError(
                f"{name} tokenizer/model revisions must match"
            )
        if str(model.get("license", "")).lower() != "apache-2.0":
            raise ValueError(f"{name} license was not recorded as apache-2.0")
        if model.get("eligibility_verified") is not True:
            raise ValueError(f"{name} revision eligibility is not verified")
        try:
            revision_date = date.fromisoformat(str(model["revision_date"]))
            cutoff = date.fromisoformat(str(config["eligibility_cutoff"]))
        except (KeyError, ValueError) as error:
            raise ValueError("model revision/cutoff dates must be ISO dates") from error
        if revision_date >= cutoff:
            raise ValueError(
                f"{name} revision date {revision_date} is not before {cutoff}"
            )
    if int(
        config["models"]["embedding"].get("embedding_dimension", 0)
    ) != 2560:
        raise ValueError(
            "initial G3C embedding must use the full 2560 dimensions"
        )
    policy = config.get("policy", {})
    if policy.get("dev_payload_forbids_gold") is not True:
        raise ValueError("dev payload gold exclusion must stay enabled")
    if int(policy.get("promotion_runs", 0)) != 1:
        raise ValueError("G3C promotion is exactly one locked run")
    gates = config.get("gates", {})
    required = (
        "minimum_leaf_recall_at_5_delta_vs_r0",
        "minimum_full_plan_coverage_delta_vs_r0",
        "maximum_docs_f2_regression_vs_r0",
        "maximum_tables_f2_regression_vs_r0",
        "maximum_hard_constraint_violations",
    )
    if any(key not in gates for key in required):
        raise ValueError("all numerical G3C promotion gates must be pre-registered")


def config_fingerprint(config: dict) -> str:
    validate_config(config)
    return canonical_json_sha256(config)


def file_manifest(root: Path | str) -> list[dict]:
    root = Path(root)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def verify_file_manifest(root: Path | str, rows: Iterable[dict]) -> None:
    root = Path(root)
    expected = {str(row["path"]): row for row in rows}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    missing = sorted(set(expected) - actual_paths)
    extra = sorted(actual_paths - set(expected))
    bad = []
    for rel, row in expected.items():
        path = root / rel
        if path.is_file() and (
            path.stat().st_size != int(row["size"])
            or sha256_file(path) != row["sha256"]
        ):
            bad.append(rel)
    if missing or extra or bad:
        raise ValueError(
            f"payload manifest mismatch missing={missing} extra={extra} bad={bad}"
        )
