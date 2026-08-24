"""Schemas and validation shared by the official G3C execution path."""
from __future__ import annotations

from pathlib import Path

from ..g3c.common import (
    canonical_json_sha256,
    read_json,
    sha256_file,
)

OFFICIAL_EXECUTION_SCHEMA = "g3c_official_execution_v1"
OFFICIAL_CLOSEOUT_SCHEMA = "g3c_pb_r4_closeout_v1"
OFFICIAL_PROTOCOL_SCHEMA = "g3c_official_protocol_freeze_v1"
OFFICIAL_PAYLOAD_SCHEMA = "g3c_official_payload_v1"
OFFICIAL_EMBEDDING_SCHEMA = "g3c_official_embedding_result_v1"
OFFICIAL_SHARD_SCHEMA = "g3c_official_rerank_shard_v1"
OFFICIAL_RESULT_SCHEMA = "g3c_official_result_v1"
OFFICIAL_AUDIT_SCHEMA = "g3c_official_engineering_audit_v1"
WORKLOAD_SCHEMA = "g3c_official_workload_v1"
CANARY_SCHEMA = "g3c_official_numeric_canary_v1"


def load_fingerprinted(
    path: Path | str,
    *,
    schema: str,
    fingerprint_field: str,
) -> dict:
    value = read_json(path)
    if value.get("schema_version") != schema:
        raise ValueError(f"unknown schema in {path}: {value.get('schema_version')}")
    expected = canonical_json_sha256({
        key: item for key, item in value.items()
        if key != fingerprint_field
    })
    if value.get(fingerprint_field) != expected:
        raise ValueError(f"fingerprint mismatch: {path}")
    return value


def file_record(root: Path | str, path: Path | str) -> dict:
    root = Path(root).resolve()
    path = Path(path).resolve()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_execution_config(value: dict) -> None:
    if value.get("schema_version") != OFFICIAL_EXECUTION_SCHEMA:
        raise ValueError("unknown official execution config schema")
    if int(value.get("question_count", 0)) != 1012:
        raise ValueError("official execution must bind exactly 1,012 questions")
    if int(value.get("gpu_workers", 0)) != 2:
        raise ValueError("official exact runner is frozen for two GPU workers")
    if int(value.get("rerank_shards", 0)) != 4:
        raise ValueError("official exact runner is frozen for four rerank shards")
    if value.get("selected_stage") != "R4":
        raise ValueError("official execution must bind frozen R4")
    if value.get("cache_seed_policy") != (
        "empty_official_cache_no_dev_or_promotion_vectors"
    ):
        raise ValueError("prior-run vector seeding is forbidden")
    unsupported = value.get("unsupported_question_policy", {})
    if unsupported != {
        "trigger": "any_atomic_leaf_has_no_exact_report",
        "action": "r0_passthrough_with_explicit_provenance",
        "allow_report_fallback_search": False,
        "allow_fuzzy_report_match": False,
        "preserve_r0_candidate_order": True,
    }:
        raise ValueError("unsupported-question totalization policy drifted")
    required_true = {
        "preserve_model_revisions",
        "preserve_tokenizer_revisions",
        "preserve_instructions",
        "preserve_precision",
        "preserve_attention_implementation",
        "preserve_max_lengths",
        "preserve_batch_sizes",
        "require_exact_gpu_canary",
        "require_rank_and_submission_projection_equivalence",
    }
    equivalence = value.get("equivalence_policy", {})
    if any(equivalence.get(key) is not True for key in required_true):
        raise ValueError("official equivalence safeguards drifted")
    required_false = {
        "split_embedding_batches",
        "split_reranker_question_calls",
        "allow_approximate_search",
        "allow_quantization",
        "allow_cache_seed_from_prior_runs",
    }
    if any(equivalence.get(key) is not False for key in required_false):
        raise ValueError("official execution enables a semantic approximation")
    public = value.get("public_question_policy", {})
    if public.get("purpose") != "post_freeze_engineering_audit_only":
        raise ValueError("official questions cannot become a selection set")
    if any(public.get(key) is not False for key in (
        "gold_available_to_pipeline",
        "score_based_selection_allowed",
        "question_specific_patches_allowed",
        "threshold_changes_allowed",
    )):
        raise ValueError("public-bias guard drifted")


def load_execution_config(path: Path | str) -> dict:
    value = read_json(path)
    validate_execution_config(value)
    return value
