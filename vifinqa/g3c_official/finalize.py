"""Merge four exact R4 shards and freeze the official engineering artifact."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from ..g3c.common import (
    canonical_json_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from ..g3c.serialize import table_key
from .common import (
    OFFICIAL_AUDIT_SCHEMA,
    OFFICIAL_RESULT_SCHEMA,
)
from .execution import (
    validate_embedding_result,
    validate_rerank_pair_result,
)
from .payload import validate_official_payload


def finalize_official_result(
    *,
    payload_dir: Path | str,
    embedding_result_dir: Path | str,
    rerank_pair_dirs: list[Path | str],
    output_dir: Path | str,
) -> tuple[dict, dict, dict]:
    payload_dir = Path(payload_dir).resolve()
    embedding_result_dir = Path(embedding_result_dir).resolve()
    pair_dirs = [Path(path).resolve() for path in rerank_pair_dirs]
    output_dir = Path(output_dir).resolve()
    payload = validate_official_payload(payload_dir)
    embedding = validate_embedding_result(
        payload_dir=payload_dir,
        result_dir=embedding_result_dir,
        require_qwen=True,
    )
    workload = read_json(payload_dir / payload["paths"]["workload"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty official result: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_manifests = []
    shards = {}
    for pair_dir in pair_dirs:
        raw = read_json(
            pair_dir / "g3c_official_rerank_pair_manifest.json"
        )
        expected = tuple(int(index) for index in raw["shard_indices"])
        pair = validate_rerank_pair_result(
            payload_dir=payload_dir,
            embedding_result_dir=embedding_result_dir,
            result_dir=pair_dir,
            expected_shards=expected,
            require_qwen=True,
        )
        pair_manifests.append({
            "path": str(pair_dir),
            "sha256": sha256_file(
                pair_dir / "g3c_official_rerank_pair_manifest.json"
            ),
            "run_signature": pair["run_signature"],
            "shard_indices": pair["shard_indices"],
        })
        for record in pair["shards"]:
            index = int(record["shard_index"])
            if index in shards:
                raise ValueError(f"official rerank shard repeated: {index}")
            shard_dir = pair_dir / record["path"]
            shard_manifest = read_json(shard_dir / "shard_manifest.json")
            shards[index] = {
                "manifest": shard_manifest,
                "rows": read_jsonl(
                    shard_dir / shard_manifest["retrieval"]["path"]
                ),
            }
    if set(shards) != {0, 1, 2, 3}:
        raise ValueError(f"official rerank shards incomplete: {sorted(shards)}")

    questions = read_jsonl(payload_dir / payload["paths"]["questions"])
    baseline_path = payload_dir / payload["paths"]["baseline_retrieval"]
    baseline = read_jsonl(baseline_path)
    row_by_id = {}
    for shard in shards.values():
        for row in shard["rows"]:
            qid = str(row["id"])
            if qid in row_by_id:
                raise ValueError(f"official R4 duplicate ID across shards: {qid}")
            row_by_id[qid] = row
    expected_ids = [str(row["id"]) for row in questions]
    if set(row_by_id) != set(expected_ids):
        raise ValueError("official R4 shard union does not cover 1,012 IDs")
    r4_rows = [row_by_id[qid] for qid in expected_ids]
    r0_path = output_dir / "r0_retrieval.jsonl"
    r4_path = output_dir / "r4_retrieval.jsonl"
    shutil.copyfile(baseline_path, r0_path)
    write_jsonl(r4_path, r4_rows)
    audit = audit_official_rows(
        questions=questions,
        baseline=baseline,
        r4_rows=r4_rows,
        payload=payload,
        expected_fallback_ids=set(
            workload["unsupported_r0_passthrough_ids"]
        ),
    )
    audit_path = output_dir / "g3c_official_engineering_audit.json"
    write_json(audit_path, audit)
    result = {
        "schema_version": OFFICIAL_RESULT_SCHEMA,
        "status": "official_1012_engineering_audit_passed",
        "scientific_or_selection_evidence": False,
        "public_score_used": False,
        "payload_fingerprint": payload["payload_fingerprint"],
        "official_protocol_fingerprint": payload[
            "official_protocol_fingerprint"
        ],
        "candidate_fingerprint": payload["candidate_fingerprint"],
        "selected_stage": "R4",
        "question_count": 1012,
        "embedding_run_signature": embedding["run_signature"],
        "embedding_manifest_sha256": sha256_file(
            embedding_result_dir / "g3c_official_embedding_manifest.json"
        ),
        "rerank_pairs": pair_manifests,
        "r0": _artifact(r0_path),
        "r4": _artifact(r4_path),
        "engineering_audit": _artifact(audit_path),
        "rank_and_submission_projection_equivalence_guard_passed": True,
        "exact_numeric_canary_passed": True,
    }
    result["result_fingerprint"] = canonical_json_sha256(result)
    result_path = output_dir / "g3c_official_result_manifest.json"
    write_json(result_path, result)
    freeze = {
        "schema_version": "g3c_official_artifact_freeze_v1",
        "candidate_fingerprint": payload["candidate_fingerprint"],
        "selected_stage": "R4",
        "payload_fingerprint": payload["payload_fingerprint"],
        "official_protocol_fingerprint": payload[
            "official_protocol_fingerprint"
        ],
        "result_fingerprint": result["result_fingerprint"],
        "result_manifest_sha256": sha256_file(result_path),
        "r4_sha256": sha256_file(r4_path),
        "question_count": 1012,
        "purpose": "post_freeze_engineering_audit_only",
        "selection_or_tuning_allowed": False,
    }
    freeze["artifact_fingerprint"] = canonical_json_sha256(freeze)
    write_json(output_dir / "g3c_official_artifact_freeze.json", freeze)
    return result, audit, freeze


def audit_official_rows(
    *,
    questions: list[dict],
    baseline: list[dict],
    r4_rows: list[dict],
    payload: dict,
    expected_fallback_ids: set[str] | None = None,
) -> dict:
    if not (len(questions) == len(baseline) == len(r4_rows) == 1012):
        raise ValueError("official engineering audit requires 1,012 aligned rows")
    ids = [str(row["id"]) for row in questions]
    if len(set(ids)) != 1012:
        raise ValueError("official engineering audit duplicate question IDs")
    candidate_counts = []
    empty_candidate_ids = []
    hard_violations = []
    nonfinite_paths = []
    duplicate_candidate_ids = []
    projection = []
    fallback_ids = []
    for index, (question, r0, r4) in enumerate(zip(questions, baseline, r4_rows)):
        qid = str(question["id"])
        if str(r0.get("id")) != qid or str(r4.get("id")) != qid:
            raise ValueError(f"official row order mismatch at {index}")
        if question["question"] != r0.get("question") or (
            question["question"] != r4.get("question")
        ):
            raise ValueError(f"official question mismatch: {qid}")
        if r4.get("g3c", {}).get("stage") != "R4":
            raise ValueError(f"official R4 provenance missing: {qid}")
        candidates = r4.get("candidates", [])
        execution_mode = r4.get("g3c", {}).get("execution")
        if execution_mode == "r0_passthrough_unsupported":
            fallback_ids.append(qid)
            if r4.get("route") != r0.get("route") or candidates != r0.get(
                "candidates", []
            ):
                raise ValueError(f"official R0 fallback rank drift: {qid}")
            if r4["g3c"].get("fallback_reason") != (
                "missing_exact_report_for_atomic_leaf"
            ):
                raise ValueError(f"official R0 fallback reason drift: {qid}")
        candidate_counts.append(len(candidates))
        if not candidates:
            empty_candidate_ids.append(qid)
        if len(candidates) > 20:
            raise ValueError(f"official R4 depth overflow: {qid}")
        keys = [table_key(candidate) for candidate in candidates]
        if len(keys) != len(set(keys)):
            duplicate_candidate_ids.append(qid)
        violations = r4["g3c"].get("hard_constraint_violations", [])
        hard_violations.extend({"id": qid, **value} for value in violations)
        _collect_nonfinite(r4, f"$[{index}]", nonfinite_paths)
        top5 = candidates[:5]
        projection.append({
            "id": question["id"],
            "relevant_docs": list(dict.fromkeys(
                str(candidate["report_id"]) for candidate in top5
            )),
            "ranked_internal_tables": [
                f"{candidate['report_id']}|{int(candidate['table_pos'])}"
                for candidate in top5
            ],
        })
    if duplicate_candidate_ids:
        raise ValueError(
            f"official R4 duplicate candidates: {duplicate_candidate_ids[:10]}"
        )
    if hard_violations:
        raise ValueError(
            f"official R4 hard-constraint violations: {hard_violations[:10]}"
        )
    if nonfinite_paths:
        raise ValueError(f"official R4 non-finite values: {nonfinite_paths[:10]}")
    if expected_fallback_ids is not None and set(fallback_ids) != set(
        expected_fallback_ids
    ):
        raise ValueError("official R0 fallback ID partition drift")
    body = {
        "schema_version": OFFICIAL_AUDIT_SCHEMA,
        "passed": True,
        "purpose": "post_freeze_crash_schema_finiteness_audit_only",
        "selection_or_metric_evidence": False,
        "payload_fingerprint": payload["payload_fingerprint"],
        "candidate_fingerprint": payload["candidate_fingerprint"],
        "question_count": 1012,
        "unique_id_count": len(set(ids)),
        "ordered_id_match": True,
        "question_text_match": True,
        "stage": "R4",
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_mean": round(
            sum(candidate_counts) / len(candidate_counts), 6
        ),
        "empty_candidate_count": len(empty_candidate_ids),
        "empty_candidate_ids": empty_candidate_ids,
        "supported_r4_count": 1012 - len(fallback_ids),
        "unsupported_r0_passthrough_count": len(fallback_ids),
        "unsupported_r0_passthrough_ids": fallback_ids,
        "unsupported_r0_candidate_order_exact": True,
        "duplicate_candidate_count": 0,
        "hard_constraint_violation_count": 0,
        "nonfinite_value_count": 0,
        "ranked_top5_projection_sha256": canonical_json_sha256(projection),
        "public_score_read": False,
        "gold_read": False,
    }
    body["audit_fingerprint"] = canonical_json_sha256(body)
    return body


def _collect_nonfinite(value, path: str, output: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_nonfinite(item, f"{path}.{key}", output)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_nonfinite(item, f"{path}[{index}]", output)
    elif isinstance(value, float) and not math.isfinite(value):
        output.append(path)


def _artifact(path: Path) -> dict:
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
