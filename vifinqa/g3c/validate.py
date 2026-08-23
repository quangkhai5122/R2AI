"""Strict local import validation for Kaggle G3C result bundles."""
from __future__ import annotations

from pathlib import Path

from .common import (
    GPU_RESULT_SCHEMA,
    config_fingerprint,
    load_config,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from .freeze import load_candidate_freeze
from .payload import validate_gpu_payload
from .serialize import table_key


def validate_gpu_results(
    *,
    payload_dir: Path | str,
    result_dir: Path | str,
    output_path: Path | str | None = None,
    require_scientific: bool = True,
    candidate_freeze_path: Path | str | None = None,
) -> dict:
    payload_dir = Path(payload_dir)
    result_dir = Path(result_dir)
    payload = validate_gpu_payload(payload_dir)
    config = load_config(payload_dir / payload["paths"]["config"])
    manifest_path = result_dir / "g3c_gpu_result_manifest.json"
    result = read_json(manifest_path)
    if result.get("schema_version") != GPU_RESULT_SCHEMA:
        raise ValueError("unknown G3C GPU-result schema")
    if result.get("payload_fingerprint") != payload["payload_fingerprint"]:
        raise ValueError("GPU result/payload fingerprint mismatch")
    if result.get("config_sha256") != config_fingerprint(config):
        raise ValueError("GPU result/config mismatch")
    if result.get("protocol_fingerprint") != (
        payload.get("protocol_fingerprint")
    ):
        raise ValueError("GPU result/payload protocol mismatch")
    if result.get("mode") != payload["mode"]:
        raise ValueError("GPU result/payload mode mismatch")
    if require_scientific and (
        result.get("backend") != "qwen"
        or result.get("scientific_evidence_valid") is not True
        or int(result.get("smoke_limit", -1)) != 0
    ):
        raise ValueError("result is fake, limited, or not scientific evidence")

    expected_stages = (
        ["R0", "R0L", "R1", "R2", "R3", "R4"]
        if payload["mode"] == "dev"
        else ["R0", payload["selected_stage"]]
    )
    if result.get("stages_written") != expected_stages:
        raise ValueError("GPU result stage set violates payload policy")
    questions = read_jsonl(payload_dir / payload["paths"]["questions"])
    smoke_limit = int(result.get("smoke_limit", 0))
    if not require_scientific and smoke_limit:
        questions = questions[:smoke_limit]
    question_map = {str(row["id"]): row["question"] for row in questions}
    if int(result.get("question_count", -1)) != len(questions):
        raise ValueError("GPU result question count mismatch")

    expected_files = {"g3c_gpu_result_manifest.json"}
    stage_validation = {}
    for stage in expected_stages:
        artifact = result["stage_artifacts"].get(stage)
        if not artifact:
            raise ValueError(f"missing stage artifact: {stage}")
        path = result_dir / artifact["path"]
        expected_files.add(path.relative_to(result_dir).as_posix())
        _verify_file(path, artifact)
        rows = read_jsonl(path)
        if len(rows) != len(questions):
            raise ValueError(f"{stage} record count mismatch")
        seen = set()
        hard_violations = []
        duplicate_candidates = []
        for row in rows:
            qid = str(row.get("id"))
            if qid in seen:
                raise ValueError(f"{stage} duplicate question ID: {qid}")
            seen.add(qid)
            if question_map.get(qid) != row.get("question"):
                raise ValueError(f"{stage} question mismatch: {qid}")
            candidate_keys = [
                table_key(candidate) for candidate in row.get("candidates", [])
            ]
            if len(candidate_keys) != len(set(candidate_keys)):
                duplicate_candidates.append(qid)
            if len(candidate_keys) > int(config["retrieval"]["depth"]):
                raise ValueError(f"{stage} exceeds retrieval depth: {qid}")
            if stage != "R0":
                g3c = row.get("g3c", {})
                if g3c.get("stage") != stage:
                    raise ValueError(f"{stage} provenance missing: {qid}")
                hard_violations.extend(
                    {"id": qid, **item}
                    for item in g3c.get("hard_constraint_violations", [])
                )
                allowed_reports = {
                    report_id
                    for leaf in g3c.get("leaves", [])
                    for report_id in leaf.get("report_ids", [])
                }
                for candidate in row.get("candidates", []):
                    if candidate.get("report_id") not in allowed_reports:
                        hard_violations.append({
                            "id": qid,
                            "report_id": candidate.get("report_id"),
                            "table_pos": candidate.get("table_pos"),
                            "reason": "report_outside_leaf_guard",
                        })
        if seen != set(question_map):
            raise ValueError(f"{stage} result ID set mismatch")
        if duplicate_candidates:
            raise ValueError(
                f"{stage} duplicate candidates: {duplicate_candidates}"
            )
        if hard_violations:
            raise ValueError(
                f"{stage} hard-constraint violations: {hard_violations[:10]}"
            )
        if int(artifact["hard_constraint_violation_count"]) != 0:
            raise ValueError(f"{stage} manifest reports hard violations")
        stage_validation[stage] = {
            "record_count": len(rows),
            "sha256": artifact["sha256"],
            "hard_constraint_violations": 0,
        }

    for name, artifact in result.get("cache_artifacts", {}).items():
        path = result_dir / "cache" / name
        expected_files.add(path.relative_to(result_dir).as_posix())
        _verify_file(path, artifact)
    actual_files = {
        path.relative_to(result_dir).as_posix()
        for path in result_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            f"result file-set mismatch extra={sorted(actual_files - expected_files)} "
            f"missing={sorted(expected_files - actual_files)}"
        )

    baseline_payload = payload_dir / payload["paths"]["baseline_retrieval"]
    if (
        int(result.get("smoke_limit", 0)) == 0
        and result["stage_artifacts"]["R0"]["sha256"]
        != sha256_file(baseline_payload)
    ):
        raise ValueError("R0 GPU artifact is not the frozen payload control")

    freeze_fingerprint = None
    if payload["mode"] == "promotion":
        freeze_path = (
            Path(candidate_freeze_path)
            if candidate_freeze_path is not None
            else payload_dir / payload["paths"]["candidate_freeze"]
        )
        freeze = load_candidate_freeze(freeze_path)
        if freeze["selected_stage"] != payload["selected_stage"]:
            raise ValueError("promotion result/candidate-freeze stage mismatch")
        if freeze["config_sha256"] != result["config_sha256"]:
            raise ValueError("promotion result/candidate-freeze config mismatch")
        if freeze["model_revisions"] != result["model_revisions"]:
            raise ValueError("promotion model revisions drifted from dev freeze")
        if freeze["instructions_sha256"] != result["instructions_sha256"]:
            raise ValueError("promotion instructions drifted from dev freeze")
        if freeze["protocol_fingerprint"] != result["protocol_fingerprint"]:
            raise ValueError("promotion protocol drifted from dev freeze")
        freeze_fingerprint = freeze["candidate_fingerprint"]

    report = {
        "schema_version": "g3c_gpu_import_validation_v1",
        "passed": True,
        "mode": payload["mode"],
        "backend": result["backend"],
        "scientific_evidence_valid": result["scientific_evidence_valid"],
        "payload_fingerprint": payload["payload_fingerprint"],
        "run_signature": result["run_signature"],
        "config_sha256": result["config_sha256"],
        "protocol_fingerprint": result["protocol_fingerprint"],
        "candidate_fingerprint": freeze_fingerprint,
        "stage_validation": stage_validation,
        "runtime": result["runtime"],
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def _verify_file(path: Path, artifact: dict) -> None:
    if not path.is_file():
        raise ValueError(f"result file missing: {path}")
    if (
        path.stat().st_size != int(artifact["size"])
        or sha256_file(path) != artifact["sha256"]
    ):
        raise ValueError(f"result artifact hash mismatch: {path}")
