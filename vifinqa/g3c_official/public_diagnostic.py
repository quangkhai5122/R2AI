"""Build and audit the frozen R4/B1-fixed public retrieval diagnostic."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ..g3c.common import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json,
)
from ..submission.build import build_submission

PUBLIC_DIAGNOSTIC_SCHEMA = "g3c_public_retrieval_diagnostic_v1"

EXPECTED = {
    "candidate_fingerprint": (
        "1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a"
    ),
    "official_protocol_fingerprint": (
        "958ba9d2dd4c979d54635f251415211b75316bc927b0e145a39e7c4270faf51f"
    ),
    "payload_fingerprint": (
        "4428637d718fddbd99db7d034337af66743a7dca23026b2da3a04af9dc55f227"
    ),
    "r0_sha256": (
        "76bedf827712a7f78a1db6b34a1da89f58b842621c01e42f89329592478f93d0"
    ),
    "r4_sha256": (
        "1281af9a737fd235e61b275c4ffebe34624d4790be6c043c81ca88d8552132d5"
    ),
    "b1_codegen_sha256": (
        "a8c2b93279daa7099ce0fdcead123cf9df134687fefeb48b484dd66267f9371c"
    ),
    "b1_submission_sha256": (
        "c98f1859e41a924458abfc7f5b2f2673e028136e7a73b2cb04c6cb84467cb75c"
    ),
}

BASELINE_PUBLIC_METRICS = {
    "tables_f2_macro": 0.4518,
    "docs_f2_macro": 0.9125,
    "tables_precision": 0.3037,
    "tables_recall": 0.6313,
    "tables_mrr5": 0.6152,
    "docs_precision": 0.9510,
    "docs_recall": 0.9109,
    "docs_mrr5": 0.9723,
    "answer_accuracy": 0.1897,
    "execution_accuracy": 0.1897,
}

_RETRIEVAL_FIELDS = {"relevant_docs", "relevant_tables"}


def _read_json_list(path: Path | str, label: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(
        not isinstance(row, dict) for row in value
    ):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def _artifact(path: Path | str) -> dict:
    path = Path(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def audit_retrieval_only_rows(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    expected_count: int = 1012,
) -> dict:
    """Prove that only relevant_docs/relevant_tables changed from B1."""
    if len(baseline_rows) != expected_count or len(candidate_rows) != expected_count:
        raise ValueError(
            f"retrieval diagnostic requires {expected_count} rows, got "
            f"baseline={len(baseline_rows)} candidate={len(candidate_rows)}"
        )
    baseline_ids = [int(row["id"]) for row in baseline_rows]
    candidate_ids = [int(row["id"]) for row in candidate_rows]
    if len(set(baseline_ids)) != expected_count:
        raise ValueError("baseline submission IDs are not unique")
    if baseline_ids != candidate_ids:
        raise ValueError("candidate submission ID/order drifted from B1")

    stats = {
        "record_count": expected_count,
        "table_order_changed_count": 0,
        "table_set_changed_count": 0,
        "doc_order_changed_count": 0,
        "doc_set_changed_count": 0,
        "table_addition_count": 0,
        "table_removal_count": 0,
        "doc_addition_count": 0,
        "doc_removal_count": 0,
    }
    baseline_table_count = candidate_table_count = 0
    baseline_doc_count = candidate_doc_count = 0
    changed_fields: dict[str, int] = {}
    for baseline, candidate in zip(baseline_rows, candidate_rows):
        qid = int(baseline["id"])
        if set(baseline) != set(candidate):
            raise ValueError(f"submission field-set drift at question {qid}")
        differences = {
            field for field in baseline
            if baseline[field] != candidate[field]
        }
        forbidden = differences - _RETRIEVAL_FIELDS
        if forbidden:
            raise ValueError(
                f"non-retrieval field drift at question {qid}: "
                f"{sorted(forbidden)}"
            )
        for field in differences:
            changed_fields[field] = changed_fields.get(field, 0) + 1

        base_tables = list(baseline["relevant_tables"])
        cand_tables = list(candidate["relevant_tables"])
        base_docs = list(baseline["relevant_docs"])
        cand_docs = list(candidate["relevant_docs"])
        if len(base_tables) != len(set(base_tables)) or len(cand_tables) != len(
            set(cand_tables)
        ):
            raise ValueError(f"duplicate relevant table at question {qid}")
        if len(base_docs) != len(set(base_docs)) or len(cand_docs) != len(
            set(cand_docs)
        ):
            raise ValueError(f"duplicate relevant document at question {qid}")

        base_table_set, cand_table_set = set(base_tables), set(cand_tables)
        base_doc_set, cand_doc_set = set(base_docs), set(cand_docs)
        stats["table_order_changed_count"] += int(base_tables != cand_tables)
        stats["table_set_changed_count"] += int(
            base_table_set != cand_table_set
        )
        stats["doc_order_changed_count"] += int(base_docs != cand_docs)
        stats["doc_set_changed_count"] += int(base_doc_set != cand_doc_set)
        stats["table_addition_count"] += len(cand_table_set - base_table_set)
        stats["table_removal_count"] += len(base_table_set - cand_table_set)
        stats["doc_addition_count"] += len(cand_doc_set - base_doc_set)
        stats["doc_removal_count"] += len(base_doc_set - cand_doc_set)
        baseline_table_count += len(base_tables)
        candidate_table_count += len(cand_tables)
        baseline_doc_count += len(base_docs)
        candidate_doc_count += len(cand_docs)

    if stats["table_set_changed_count"] == 0:
        raise ValueError("retrieval diagnostic did not change any table sets")
    if stats["doc_set_changed_count"] == 0:
        raise ValueError("retrieval diagnostic did not change any document sets")
    stats.update({
        "changed_field_counts": dict(sorted(changed_fields.items())),
        "allowed_changed_fields": sorted(_RETRIEVAL_FIELDS),
        "non_retrieval_fields_exact": True,
        "baseline_tables_per_question_mean": round(
            baseline_table_count / expected_count, 6
        ),
        "candidate_tables_per_question_mean": round(
            candidate_table_count / expected_count, 6
        ),
        "baseline_docs_per_question_mean": round(
            baseline_doc_count / expected_count, 6
        ),
        "candidate_docs_per_question_mean": round(
            candidate_doc_count / expected_count, 6
        ),
    })
    return stats


def audit_zip_delta(
    baseline_zip: Path | str,
    candidate_zip: Path | str,
    candidate_results_path: Path | str,
) -> dict:
    """Require the candidate archive to differ from B1 only in results.json."""
    with zipfile.ZipFile(baseline_zip) as baseline, zipfile.ZipFile(
        candidate_zip
    ) as candidate:
        baseline_names = baseline.namelist()
        candidate_names = candidate.namelist()
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("candidate ZIP contains duplicate member paths")
        unsafe = [
            name for name in candidate_names
            if Path(name).is_absolute() or ".." in Path(name).parts
        ]
        if unsafe:
            raise ValueError(f"candidate ZIP contains unsafe paths: {unsafe[:5]}")
        if set(baseline_names) != set(candidate_names):
            raise ValueError("candidate ZIP member set drifted from B1")
        differing = [
            name for name in sorted(candidate_names)
            if baseline.read(name) != candidate.read(name)
        ]
        if differing != ["results.json"]:
            raise ValueError(
                "candidate ZIP must differ from B1 only in results.json; "
                f"got {differing[:10]}"
            )
        archived_results = json.loads(candidate.read("results.json"))
    local_results = _read_json_list(candidate_results_path, "candidate results")
    if archived_results != local_results:
        raise ValueError("candidate ZIP results.json differs from local results.json")
    return {
        "archive_member_count": len(candidate_names),
        "data_member_count": sum(
            name.startswith("data/") and name.endswith(".csv")
            for name in candidate_names
        ),
        "member_set_exact_vs_b1": True,
        "data_members_byte_exact_vs_b1": True,
        "differing_members_vs_b1": differing,
        "results_json_matches_local": True,
        "safe_relative_paths": True,
    }


def validate_frozen_inputs(
    *,
    r0_path: Path | str,
    r4_path: Path | str,
    codegen_path: Path | str,
    baseline_zip: Path | str,
    official_result_manifest_path: Path | str,
    official_freeze_path: Path | str,
) -> dict:
    actual = {
        "r0_sha256": sha256_file(r0_path),
        "r4_sha256": sha256_file(r4_path),
        "b1_codegen_sha256": sha256_file(codegen_path),
        "b1_submission_sha256": sha256_file(baseline_zip),
    }
    for field, value in actual.items():
        if value != EXPECTED[field]:
            raise ValueError(
                f"frozen input drift: {field} expected={EXPECTED[field]} got={value}"
            )
    result = read_json(official_result_manifest_path)
    freeze = read_json(official_freeze_path)
    required = {
        "candidate_fingerprint": EXPECTED["candidate_fingerprint"],
        "official_protocol_fingerprint": EXPECTED[
            "official_protocol_fingerprint"
        ],
        "payload_fingerprint": EXPECTED["payload_fingerprint"],
    }
    for field, expected in required.items():
        if result.get(field) != expected or freeze.get(field) != expected:
            raise ValueError(f"official result/freeze {field} mismatch")
    if result.get("status") != "official_1012_engineering_audit_passed":
        raise ValueError("official R4 engineering audit is not complete")
    if result.get("selected_stage") != "R4" or freeze.get(
        "selected_stage"
    ) != "R4":
        raise ValueError("official selected stage is not R4")
    if result.get("public_score_used") is not False:
        raise ValueError("official R4 artifact unexpectedly used public score")
    if freeze.get("selection_or_tuning_allowed") is not False:
        raise ValueError("official R4 freeze unexpectedly permits tuning")
    if result.get("r0", {}).get("sha256") != actual["r0_sha256"]:
        raise ValueError("official R0 hash mismatch")
    if result.get("r4", {}).get("sha256") != actual["r4_sha256"]:
        raise ValueError("official R4 hash mismatch")
    if freeze.get("r4_sha256") != actual["r4_sha256"]:
        raise ValueError("official freeze R4 hash mismatch")
    return {
        **actual,
        **required,
        "result_fingerprint": result["result_fingerprint"],
        "artifact_fingerprint": freeze["artifact_fingerprint"],
        "question_count": int(result["question_count"]),
        "public_score_used_for_r4_selection": False,
    }


def _build_audit(
    *,
    baseline_results_path: Path | str,
    candidate_results_path: Path | str,
    baseline_zip: Path | str,
    candidate_zip: Path | str,
) -> dict:
    baseline_rows = _read_json_list(baseline_results_path, "B1 baseline results")
    candidate_rows = _read_json_list(
        candidate_results_path, "R4 candidate results"
    )
    audit = {
        "schema_version": PUBLIC_DIAGNOSTIC_SCHEMA,
        "purpose": "public_aggregate_retrieval_metrics_only",
        "row_delta": audit_retrieval_only_rows(baseline_rows, candidate_rows),
        "zip_delta": audit_zip_delta(
            baseline_zip, candidate_zip, candidate_results_path
        ),
    }
    audit["audit_fingerprint"] = canonical_json_sha256(audit)
    return audit


def build_public_retrieval_diagnostic(
    *,
    r0_path: Path | str,
    r4_path: Path | str,
    codegen_path: Path | str,
    store_dir: Path | str,
    baseline_results_path: Path | str,
    baseline_zip: Path | str,
    official_result_manifest_path: Path | str,
    official_freeze_path: Path | str,
    output_dir: Path | str,
) -> tuple[dict, dict]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty diagnostic output: {output_dir}"
        )
    inputs = validate_frozen_inputs(
        r0_path=r0_path,
        r4_path=r4_path,
        codegen_path=codegen_path,
        baseline_zip=baseline_zip,
        official_result_manifest_path=official_result_manifest_path,
        official_freeze_path=official_freeze_path,
    )
    candidate_zip = build_submission(
        Path(r4_path),
        Path(codegen_path),
        Path(store_dir),
        output_dir,
        sub_k=5,
        pos_mode="line",
        expand_docs=False,
        offline_eval=False,
    )
    marker = output_dir / "DO_NOT_UPLOAD.txt"
    if marker.exists():
        raise ValueError("builder marked the candidate as non-official")
    candidate_results = output_dir / "results.json"
    audit = _build_audit(
        baseline_results_path=baseline_results_path,
        candidate_results_path=candidate_results,
        baseline_zip=baseline_zip,
        candidate_zip=candidate_zip,
    )
    audit_path = output_dir / "g3c_public_retrieval_diagnostic_audit.json"
    write_json(audit_path, audit)
    manifest = {
        "schema_version": PUBLIC_DIAGNOSTIC_SCHEMA,
        "status": "ready_for_public_dashboard_upload",
        "candidate_name": "G3C-R4-B1-fixed-public-retrieval-diagnostic-v1",
        "purpose": "measure_public_aggregate_retrieval_metrics_only",
        "canonical_private_pb_artifact": False,
        "inputs": inputs,
        "build": {
            "retrieval_stage": "R4",
            "sub_k": 5,
            "position_mode": "line",
            "expand_docs": False,
            "answer_codegen_stack": "B1_14B_NF4_fixed",
            "answer_evidence_query_mutation_allowed": False,
        },
        "artifacts": {
            "results": _artifact(candidate_results),
            "submission_zip": _artifact(candidate_zip),
            "audit": _artifact(audit_path),
        },
        "baseline_public_metrics": BASELINE_PUBLIC_METRICS,
        "expected_dashboard_invariants": {
            "answer_accuracy": BASELINE_PUBLIC_METRICS["answer_accuracy"],
            "execution_accuracy": BASELINE_PUBLIC_METRICS[
                "execution_accuracy"
            ],
            "failure_meaning": (
                "If answer/execution changes, reject the comparison as a "
                "packaging or grader-drift failure."
            ),
        },
        "interpretation_policy": {
            "primary_metrics": ["tables_f2_macro", "docs_f2_macro"],
            "secondary_metrics": [
                "tables_precision", "tables_recall", "tables_mrr5",
                "docs_precision", "docs_recall", "docs_mrr5",
            ],
            "read_aggregate_metrics_only": True,
            "question_level_public_analysis_allowed": False,
            "r4_retuning_from_dashboard_allowed": False,
            "private_candidate_selection_from_dashboard_allowed": False,
            "g3d_plan_changes_from_dashboard_allowed": False,
        },
        "audit_fingerprint": audit["audit_fingerprint"],
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    write_json(
        output_dir / "g3c_public_retrieval_diagnostic_manifest.json",
        manifest,
    )
    return manifest, audit


def validate_public_retrieval_diagnostic(
    *,
    r0_path: Path | str,
    r4_path: Path | str,
    codegen_path: Path | str,
    baseline_results_path: Path | str,
    baseline_zip: Path | str,
    official_result_manifest_path: Path | str,
    official_freeze_path: Path | str,
    output_dir: Path | str,
) -> dict:
    output_dir = Path(output_dir)
    inputs = validate_frozen_inputs(
        r0_path=r0_path,
        r4_path=r4_path,
        codegen_path=codegen_path,
        baseline_zip=baseline_zip,
        official_result_manifest_path=official_result_manifest_path,
        official_freeze_path=official_freeze_path,
    )
    audit = _build_audit(
        baseline_results_path=baseline_results_path,
        candidate_results_path=output_dir / "results.json",
        baseline_zip=baseline_zip,
        candidate_zip=output_dir / "submission.zip",
    )
    stored_audit = read_json(
        output_dir / "g3c_public_retrieval_diagnostic_audit.json"
    )
    if audit != stored_audit:
        raise ValueError("stored public diagnostic audit drifted")
    manifest_path = output_dir / "g3c_public_retrieval_diagnostic_manifest.json"
    manifest = read_json(manifest_path)
    fingerprint = manifest.pop("manifest_fingerprint", None)
    if fingerprint != canonical_json_sha256(manifest):
        raise ValueError("public diagnostic manifest fingerprint mismatch")
    manifest["manifest_fingerprint"] = fingerprint
    if manifest.get("inputs") != inputs:
        raise ValueError("public diagnostic manifest input binding drifted")
    if manifest.get("audit_fingerprint") != audit["audit_fingerprint"]:
        raise ValueError("public diagnostic manifest audit binding drifted")
    artifacts = manifest.get("artifacts", {})
    expected_artifacts = {
        "results": _artifact(output_dir / "results.json"),
        "submission_zip": _artifact(output_dir / "submission.zip"),
        "audit": _artifact(
            output_dir / "g3c_public_retrieval_diagnostic_audit.json"
        ),
    }
    if artifacts != expected_artifacts:
        raise ValueError("public diagnostic artifact hash drifted")
    return {
        "status": "public_retrieval_diagnostic_valid",
        "manifest_fingerprint": fingerprint,
        "submission_sha256": artifacts["submission_zip"]["sha256"],
        "audit_fingerprint": audit["audit_fingerprint"],
        "record_count": audit["row_delta"]["record_count"],
        "non_retrieval_fields_exact": audit["row_delta"][
            "non_retrieval_fields_exact"
        ],
        "data_members_byte_exact_vs_b1": audit["zip_delta"][
            "data_members_byte_exact_vs_b1"
        ],
    }
