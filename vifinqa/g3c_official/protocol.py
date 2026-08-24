"""Machine-verifiable freeze for exact official R4 execution."""
from __future__ import annotations

from pathlib import Path

from ..g3c.common import (
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    sha256_file,
    write_json,
)
from ..g3c.freeze import load_candidate_freeze
from ..g3c.protocol import validate_protocol_freeze
from .canary import load_numeric_canary
from .closeout import validate_pb_closeout
from .common import (
    OFFICIAL_PROTOCOL_SCHEMA,
    load_execution_config,
    load_fingerprinted,
)

BEHAVIOR_FILES = (
    "configs/g3c_qwen_retrieval_v1.json",
    "configs/g3c_official_1012_v1.json",
    "scripts/81_g3c_closeout_pb.py",
    "scripts/82_g3c_build_exact_canary.py",
    "scripts/83_g3c_freeze_official_protocol.py",
    "scripts/84_g3c_build_official_payload.py",
    "scripts/85_g3c_finalize_official.py",
    "kaggle/kaggle_g3c_official.py",
    "kaggle/requirements-g3c.txt",
    "kaggle/vifinqa-g3c-official-embedding.ipynb",
    "kaggle/vifinqa-g3c-official-rerank-a.ipynb",
    "kaggle/vifinqa-g3c-official-rerank-b.ipynb",
)
BEHAVIOR_GLOBS = ("vifinqa/g3c_official/**/*.py",)


def build_official_protocol_freeze(
    *,
    repo_root: Path | str,
    output_path: Path | str,
    source_config_path: Path | str,
    execution_config_path: Path | str,
    source_protocol_path: Path | str,
    candidate_path: Path | str,
    closeout_path: Path | str,
    registry_path: Path | str,
    canary_manifest_path: Path | str,
    canary_vectors_path: Path | str,
) -> dict:
    repo_root = Path(repo_root).resolve()
    output_path = Path(output_path).resolve()
    source_config_path = Path(source_config_path).resolve()
    execution_config_path = Path(execution_config_path).resolve()
    source_protocol_path = Path(source_protocol_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    closeout_path = Path(closeout_path).resolve()
    registry_path = Path(registry_path).resolve()
    canary_manifest_path = Path(canary_manifest_path).resolve()
    canary_vectors_path = Path(canary_vectors_path).resolve()

    source_config = load_config(source_config_path)
    execution = load_execution_config(execution_config_path)
    source_protocol = validate_protocol_freeze(
        repo_root=repo_root,
        config_path=source_config_path,
        freeze_path=source_protocol_path,
        verify_worktree=True,
    )
    candidate = load_candidate_freeze(candidate_path)
    closeout, registry = validate_pb_closeout(
        repo_root=repo_root,
        closeout_path=closeout_path,
        registry_path=registry_path,
    )
    canary = load_numeric_canary(canary_manifest_path, canary_vectors_path)
    if execution["source_config_sha256"] != config_fingerprint(source_config):
        raise ValueError("official execution/source config mismatch")
    if execution["candidate_fingerprint"] != candidate["candidate_fingerprint"]:
        raise ValueError("official execution/candidate mismatch")
    if closeout["candidate_fingerprint"] != candidate["candidate_fingerprint"]:
        raise ValueError("official closeout/candidate mismatch")
    if candidate["protocol_fingerprint"] != source_protocol["protocol_fingerprint"]:
        raise ValueError("official candidate/source protocol mismatch")
    if canary["protocol_fingerprint"] != source_protocol["protocol_fingerprint"]:
        raise ValueError("official canary/source protocol mismatch")

    sources = _source_hashes(repo_root)
    body = {
        "schema_version": OFFICIAL_PROTOCOL_SCHEMA,
        "status": "frozen_before_official_1012_gpu_execution",
        "frozen_on": "2026-08-24",
        "purpose": "post_freeze_crash_schema_finiteness_audit_only",
        "source_g3c_protocol_fingerprint": source_protocol[
            "protocol_fingerprint"
        ],
        "source_g3c_behavior_tree_sha256": source_protocol[
            "behavior_tree_sha256"
        ],
        "source_config_sha256": config_fingerprint(source_config),
        "execution_config_sha256": sha256_file(execution_config_path),
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "candidate_freeze_sha256": sha256_file(candidate_path),
        "pb_closeout_fingerprint": closeout["closeout_fingerprint"],
        "pb_closeout_sha256": sha256_file(closeout_path),
        "registry_fingerprint": registry["registry_fingerprint"],
        "numeric_canary_fingerprint": canary["canary_fingerprint"],
        "numeric_canary_manifest_sha256": sha256_file(canary_manifest_path),
        "numeric_canary_vectors_sha256": sha256_file(canary_vectors_path),
        "behavior_files": sources,
        "behavior_tree_sha256": canonical_json_sha256(sources),
        "equivalence_contract": {
            "selected_stage": "R4",
            "model_tokenizer_instruction_config_unchanged": True,
            "precision_attention_lengths_batches_unchanged": True,
            "embedding_partition_unit": "whole_frozen_batch",
            "reranker_partition_unit": "whole_question",
            "prior_qwen_vector_or_score_cache_seeded": False,
            "approximate_search_or_quantization": False,
            "exact_numeric_canary_required_on_each_gpu_and_model": True,
            "rank_and_submission_projection_must_match_frozen_replay": True,
            "supported_questions_use_exact_frozen_r4": True,
            "undefined_missing_report_questions_use_r0_passthrough": True,
            "fallback_report_search_or_fuzzy_match": False,
        },
        "public_bias_guard": {
            "official_gold_in_payload": False,
            "public_score_used": False,
            "selection_or_threshold_tuning": False,
            "question_specific_repairs": False,
        },
    }
    body["official_protocol_fingerprint"] = canonical_json_sha256(body)
    write_json(output_path, body)
    return body


def load_official_protocol(path: Path | str) -> dict:
    return load_fingerprinted(
        path,
        schema=OFFICIAL_PROTOCOL_SCHEMA,
        fingerprint_field="official_protocol_fingerprint",
    )


def validate_official_protocol(
    *,
    repo_root: Path | str,
    freeze_path: Path | str,
    verify_worktree: bool = True,
) -> dict:
    repo_root = Path(repo_root).resolve()
    freeze = load_official_protocol(freeze_path)
    if verify_worktree:
        current = _source_hashes(repo_root)
        if current != freeze["behavior_files"]:
            raise ValueError("official G3C behavior files drifted after freeze")
        if canonical_json_sha256(current) != freeze["behavior_tree_sha256"]:
            raise ValueError("official G3C behavior tree mismatch")
        source_protocol = validate_protocol_freeze(
            repo_root=repo_root,
            config_path=repo_root / "configs/g3c_qwen_retrieval_v1.json",
            freeze_path=(
                repo_root / "experiments/g3c_qwen_retrieval_v1/"
                "dev_protocol_freeze_v2.json"
            ),
            verify_worktree=True,
        )
        if source_protocol["protocol_fingerprint"] != freeze[
            "source_g3c_protocol_fingerprint"
        ]:
            raise ValueError("official freeze/source G3C protocol mismatch")
    return freeze


def _source_hashes(repo_root: Path) -> list[dict]:
    rows = []
    for relative in _behavior_paths(repo_root):
        path = repo_root / relative
        rows.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _behavior_paths(repo_root: Path) -> list[str]:
    paths = set(BEHAVIOR_FILES)
    for pattern in BEHAVIOR_GLOBS:
        for path in repo_root.glob(pattern):
            if path.is_file():
                paths.add(path.relative_to(repo_root).as_posix())
    missing = sorted(
        relative for relative in paths if not (repo_root / relative).is_file()
    )
    if missing:
        raise FileNotFoundError(f"official G3C behavior files missing: {missing}")
    return sorted(paths)
