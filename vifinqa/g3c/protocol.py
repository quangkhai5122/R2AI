"""Machine-verifiable pre-run freeze for the G3C dev protocol."""
from __future__ import annotations

from pathlib import Path

from .common import (
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    read_json,
    sha256_file,
    write_json,
)

PROTOCOL_SCHEMA = "g3c_dev_protocol_freeze_v1"
BEHAVIOR_FILES = (
    "configs/g3c_qwen_retrieval_v1.json",
    "vifinqa/config.py",
    "scripts/60_run_clean_b0_v2.py",
    "scripts/71_g3b_freeze_candidate.py",
    "scripts/72_g3b_evaluate.py",
    "scripts/74_g3b_build_submission.py",
    "scripts/75_g3c_audit_leaves.py",
    "scripts/76_g3c_build_payload.py",
    "scripts/77_g3c_validate_gpu_results.py",
    "scripts/78_g3c_evaluate_stages.py",
    "scripts/79_g3c_select_freeze.py",
    "scripts/80_g3c_freeze_protocol.py",
    "kaggle/kaggle_g3c_qwen_retrieval.py",
    "kaggle/requirements-g3c.txt",
    "kaggle/vifinqa-g3c-dev-qwen-retrieval.ipynb",
    "kaggle/vifinqa-g3c-promotion-qwen-retrieval.ipynb",
)
BEHAVIOR_GLOBS = (
    "vifinqa/g3c/**/*.py",
    "vifinqa/clean/**/*.py",
    "vifinqa/codegen/**/*.py",
    "vifinqa/extraction/**/*.py",
    "vifinqa/finance/**/*.py",
    "vifinqa/retrieval/**/*.py",
    "vifinqa/router/**/*.py",
    "vifinqa/utils/**/*.py",
)


def build_protocol_freeze(
    *,
    repo_root: Path | str,
    config_path: Path | str,
    output_path: Path | str,
) -> dict:
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    sources = _source_hashes(repo_root)
    body = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "frozen_before_first_qwen_dev_run",
        "config_sha256": config_fingerprint(config),
        "g3_evaluation_freeze_sha256": (
            config["g3_evaluation_freeze_sha256"]
        ),
        "g3_evaluation_freeze_file_sha256": sha256_file(
            repo_root
            / "experiments/g3_evaluation_v1/g3_evaluation_freeze.json"
        ),
        "model_revisions": {
            name: {
                "model_id": value["model_id"],
                "revision": value["revision"],
                "tokenizer_revision": value["tokenizer_revision"],
                "revision_date": value["revision_date"],
            }
            for name, value in config["models"].items()
        },
        "instructions_sha256": canonical_json_sha256(
            config["instructions"]
        ),
        "retrieval_contract_sha256": canonical_json_sha256(
            config["retrieval"]
        ),
        "runtime_contract_sha256": canonical_json_sha256(
            config["runtime"]
        ),
        "gate_contract_sha256": canonical_json_sha256(
            config["gates"]
        ),
        "behavior_files": sources,
        "behavior_tree_sha256": canonical_json_sha256(sources),
        "scientific_boundary": {
            "gold_in_gpu_payload": False,
            "development_split": "primary_tune",
            "promotion_splits": ["primary_locked", "hard"],
            "promotion_runs": 1,
            "reasoning_stack_frozen": True,
        },
    }
    body["protocol_fingerprint"] = canonical_json_sha256(body)
    write_json(output_path, body)
    return body


def load_protocol_freeze(path: Path | str) -> dict:
    freeze = read_json(path)
    if freeze.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unknown G3C protocol-freeze schema")
    expected = canonical_json_sha256({
        key: value for key, value in freeze.items()
        if key != "protocol_fingerprint"
    })
    if freeze.get("protocol_fingerprint") != expected:
        raise ValueError("G3C protocol fingerprint mismatch")
    return freeze


def validate_protocol_freeze(
    *,
    repo_root: Path | str,
    config_path: Path | str,
    freeze_path: Path | str,
    verify_worktree: bool = True,
) -> dict:
    repo_root = Path(repo_root).resolve()
    config = load_config(config_path)
    freeze = load_protocol_freeze(freeze_path)
    if freeze.get("config_sha256") != config_fingerprint(config):
        raise ValueError("G3C protocol freeze/config mismatch")
    if (
        freeze.get("g3_evaluation_freeze_sha256")
        != config["g3_evaluation_freeze_sha256"]
    ):
        raise ValueError("G3 evaluation freeze drift")
    if verify_worktree:
        current = _source_hashes(repo_root)
        if current != freeze.get("behavior_files"):
            raise ValueError("G3C behavior files drifted after protocol freeze")
        current_g3 = sha256_file(
            repo_root
            / "experiments/g3_evaluation_v1/g3_evaluation_freeze.json"
        )
        if current_g3 != freeze["g3_evaluation_freeze_file_sha256"]:
            raise ValueError("G3 evaluation freeze file drift")
    return freeze


def _source_hashes(repo_root: Path) -> list[dict]:
    relative_paths = _behavior_paths(repo_root)
    output = []
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"G3C behavior file missing: {relative}")
        output.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return output


def _behavior_paths(repo_root: Path) -> list[str]:
    paths = set(BEHAVIOR_FILES)
    for pattern in BEHAVIOR_GLOBS:
        for path in repo_root.glob(pattern):
            if path.is_file():
                paths.add(path.relative_to(repo_root).as_posix())
    missing = [
        relative for relative in paths
        if not (repo_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"G3C behavior files missing: {sorted(missing)}"
        )
    return sorted(paths)
