"""Close and register the promoted P-B/R4 retrieval hypothesis."""
from __future__ import annotations

from pathlib import Path

from ..g3c.common import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json,
)
from ..g3c.freeze import load_candidate_freeze
from ..g3c.payload import validate_gpu_payload
from ..g3c.protocol import validate_protocol_freeze
from ..g3c.validate import validate_gpu_results
from .common import OFFICIAL_CLOSEOUT_SCHEMA, load_fingerprinted


def build_pb_closeout(
    *,
    repo_root: Path | str,
    output_path: Path | str,
    registry_path: Path | str,
    config_path: Path | str,
    protocol_path: Path | str,
    candidate_path: Path | str,
    promotion_payload_dir: Path | str,
    promotion_result_dir: Path | str,
    promotion_import_path: Path | str,
    promotion_marker_path: Path | str,
    baseline_evaluation_path: Path | str,
    candidate_evaluation_path: Path | str,
    paired_path: Path | str,
) -> tuple[dict, dict]:
    repo_root = Path(repo_root).resolve()
    output_path = Path(output_path).resolve()
    registry_path = Path(registry_path).resolve()
    config_path = Path(config_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    promotion_payload_dir = Path(promotion_payload_dir).resolve()
    promotion_result_dir = Path(promotion_result_dir).resolve()
    promotion_import_path = Path(promotion_import_path).resolve()
    promotion_marker_path = Path(promotion_marker_path).resolve()
    baseline_evaluation_path = Path(baseline_evaluation_path).resolve()
    candidate_evaluation_path = Path(candidate_evaluation_path).resolve()
    paired_path = Path(paired_path).resolve()

    protocol = validate_protocol_freeze(
        repo_root=repo_root,
        config_path=config_path,
        freeze_path=protocol_path,
        verify_worktree=True,
    )
    candidate = load_candidate_freeze(candidate_path)
    payload = validate_gpu_payload(promotion_payload_dir)
    validation = validate_gpu_results(
        payload_dir=promotion_payload_dir,
        result_dir=promotion_result_dir,
        require_scientific=True,
        candidate_freeze_path=candidate_path,
    )
    imported = read_json(promotion_import_path)
    marker = read_json(promotion_marker_path)
    baseline = read_json(baseline_evaluation_path)
    promoted = read_json(candidate_evaluation_path)
    paired = read_json(paired_path)

    if candidate.get("selected_stage") != "R4" or not candidate.get("gate_passed"):
        raise ValueError("P-B closeout requires the passing frozen R4 candidate")
    if candidate.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("candidate/protocol mismatch")
    if payload.get("selected_stage") != "R4":
        raise ValueError("promotion payload is not the frozen R4")
    if validation.get("candidate_fingerprint") != candidate["candidate_fingerprint"]:
        raise ValueError("promotion result/candidate mismatch")
    if imported.get("run_signature") != validation.get("run_signature"):
        raise ValueError("preserved import report/run mismatch")
    if marker.get("candidate_fingerprint") != candidate["candidate_fingerprint"]:
        raise ValueError("promotion-open marker/candidate mismatch")
    if marker.get("submission_freeze_sha256") != sha256_file(
        candidate_evaluation_path.parent / "g3b_submission_freeze.json"
    ):
        raise ValueError("promotion-open marker/submission freeze mismatch")
    if promoted.get("integrity", {}).get("passed") is not True:
        raise ValueError("promotion evaluation integrity did not pass")
    if baseline.get("integrity", {}).get("passed") is not True:
        raise ValueError("promotion baseline integrity did not pass")
    if paired.get("policy_mode") != "promotion":
        raise ValueError("paired diagnostics are not promotion evidence")

    metric_keys = (
        "docs_f2_macro",
        "tables_f2_macro",
        "leaf_recall_at_k",
        "full_plan_coverage_rate",
        "answer_accuracy",
        "execution_accuracy",
    )
    base_metrics = baseline["metrics"]
    candidate_metrics = promoted["metrics"]
    metrics = {
        key: {
            "r0": round(float(base_metrics[key]), 6),
            "r4": round(float(candidate_metrics[key]), 6),
            "delta": round(
                float(candidate_metrics[key]) - float(base_metrics[key]), 6
            ),
        }
        for key in metric_keys
    }
    for key in (
        "docs_f2_macro", "tables_f2_macro",
        "leaf_recall_at_k", "full_plan_coverage_rate",
    ):
        if metrics[key]["delta"] <= 0:
            raise ValueError(f"R4 did not replicate a positive {key} delta")
    for key in ("answer_accuracy", "execution_accuracy"):
        if metrics[key]["delta"] != 0:
            raise ValueError(f"unexpected promoted {key} change")

    evidence = {
        "config": _evidence(repo_root, config_path),
        "dev_protocol_freeze": _evidence(repo_root, protocol_path),
        "candidate_freeze": _evidence(repo_root, candidate_path),
        "promotion_payload_manifest": _evidence(
            repo_root,
            promotion_payload_dir / "g3c_gpu_payload_manifest.json",
        ),
        "promotion_result_manifest": _evidence(
            repo_root,
            promotion_result_dir / "g3c_gpu_result_manifest.json",
        ),
        "promotion_import_validation": _evidence(
            repo_root, promotion_import_path
        ),
        "promotion_open_marker": _evidence(repo_root, promotion_marker_path),
        "baseline_evaluation": _evidence(repo_root, baseline_evaluation_path),
        "candidate_evaluation": _evidence(repo_root, candidate_evaluation_path),
        "paired_diagnostics": _evidence(repo_root, paired_path),
    }
    body = {
        "schema_version": OFFICIAL_CLOSEOUT_SCHEMA,
        "status": "g3c_complete_pb_r4_frozen",
        "closed_on": "2026-08-24",
        "candidate_name": candidate["candidate_name"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "selected_stage": "R4",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "promotion_payload_fingerprint": payload["payload_fingerprint"],
        "promotion_run_signature": validation["run_signature"],
        "promotion_metrics": metrics,
        "bounded_claim": (
            "R4 improved frozen retrieval metrics on dev and one-shot "
            "same-corpus/different-question promotion; answer accuracy did not improve."
        ),
        "official_1012_policy": {
            "purpose": "post_freeze_crash_schema_finiteness_audit_only",
            "selection_or_tuning_allowed": False,
            "gold_or_public_score_dependency_allowed": False,
            "question_specific_changes_allowed": False,
        },
        "promotion_evaluator_consumed": True,
        "promotion_rerun_allowed": False,
        "evidence": evidence,
    }
    body["closeout_fingerprint"] = canonical_json_sha256(body)
    write_json(output_path, body)

    registry = {
        "schema_version": "g3c_experiment_registry_v1",
        "updated_on": "2026-08-24",
        "workstream": "clean-canonical-baseline-v1",
        "active_candidate": {
            "hypothesis_class": "P-B retrieval",
            "candidate_name": candidate["candidate_name"],
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "selected_stage": "R4",
            "status": "promotion_passed_frozen",
            "closeout": output_path.relative_to(repo_root).as_posix(),
            "closeout_sha256": sha256_file(output_path),
        },
        "promotion": {
            "maximum_runs": 1,
            "runs_consumed": 1,
            "rerun_allowed": False,
        },
        "official_1012": {
            "status": "engineering_audit_pending",
            "selection_role": "none",
            "expected_records": 1012,
        },
        "next_scientific_hypothesis": "G3D typed planning and row/cell grounding",
    }
    registry["registry_fingerprint"] = canonical_json_sha256(registry)
    write_json(registry_path, registry)
    return body, registry


def load_pb_closeout(path: Path | str) -> dict:
    return load_fingerprinted(
        path,
        schema=OFFICIAL_CLOSEOUT_SCHEMA,
        fingerprint_field="closeout_fingerprint",
    )


def validate_pb_closeout(
    *, repo_root: Path | str, closeout_path: Path | str,
    registry_path: Path | str,
) -> tuple[dict, dict]:
    repo_root = Path(repo_root).resolve()
    closeout_path = Path(closeout_path).resolve()
    registry_path = Path(registry_path).resolve()
    closeout = load_pb_closeout(closeout_path)
    for record in closeout["evidence"].values():
        path = repo_root / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"P-B closeout evidence drift: {record['path']}")
    registry = load_fingerprinted(
        registry_path,
        schema="g3c_experiment_registry_v1",
        fingerprint_field="registry_fingerprint",
    )
    active = registry["active_candidate"]
    if active["candidate_fingerprint"] != closeout["candidate_fingerprint"]:
        raise ValueError("G3C registry/closeout candidate mismatch")
    if active["closeout_sha256"] != sha256_file(closeout_path):
        raise ValueError("G3C registry/closeout hash mismatch")
    return closeout, registry


def _evidence(repo_root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
