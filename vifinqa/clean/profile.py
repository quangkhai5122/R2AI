"""Fail-closed contract for clean/private-safe ViFinQA experiments."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..finance.metrics import METRICS
from ..finance.operators import operator_registry_fingerprint

CLEAN_PROFILE = "clean"
HISTORICAL_PROFILE = "historical"
PAYLOAD_SCHEMA_VERSION = 9

FORBIDDEN_CLEAN_OPTION_NAMES = (
    "llm_ids_file",
    "target_dir",
    "expect_selected_ids_file",
    "allow_ids_file",
)


def canonical_json_sha256(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def metric_registry_fingerprint() -> str:
    return canonical_json_sha256({
        key: metric.to_dict() for key, metric in sorted(METRICS.items())
    })


def source_tree_fingerprint(root: Path, paths: list[Path]) -> str:
    rows = []
    for path in sorted(paths, key=lambda p: p.as_posix()):
        resolved = path if path.is_absolute() else root / path
        rows.append({
            "path": resolved.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        })
    return canonical_json_sha256(rows)


def validate_clean_runtime(*, llm_mode: str, llm_ids_file=None,
                           expect_selected_ids_file=None,
                           allow_ids_file=None, target_dir=None,
                           skip_payload_verification: bool = False) -> None:
    if llm_mode != "select_v2":
        raise ValueError("clean profile requires llm_mode=select_v2")
    forbidden = {
        "llm_ids_file": llm_ids_file,
        "expect_selected_ids_file": expect_selected_ids_file,
        "allow_ids_file": allow_ids_file,
        "target_dir": target_dir,
    }
    active = sorted(name for name, value in forbidden.items()
                    if value not in (None, "", False))
    if active:
        raise ValueError("clean profile forbids public/question-ID controls: " + ", ".join(active))
    if skip_payload_verification:
        raise ValueError("clean profile forbids --skip-payload-verification")


def advertised_parameter_billions(model_name: str) -> float | None:
    match = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", model_name or "")
    return float(match.group(1)) if match else None


def validate_model_limit(model_name: str, maximum_billions: float = 15.0) -> None:
    size = advertised_parameter_billions(model_name)
    if size is not None and size > maximum_billions:
        raise ValueError(
            f"model advertises {size:g}B parameters, above the {maximum_billions:g}B limit"
        )


def contract_fingerprints() -> dict[str, str]:
    return {
        "metric_registry_sha256": metric_registry_fingerprint(),
        "operator_registry_sha256": operator_registry_fingerprint(),
    }
