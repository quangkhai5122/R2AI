"""Numerical dev gate and machine-verifiable G3C candidate freeze."""
from __future__ import annotations

from pathlib import Path

from .common import (
    CANDIDATE_FREEZE_SCHEMA,
    GPU_RESULT_SCHEMA,
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    read_json,
    sha256_file,
    write_json,
)


def build_dev_selection(
    *,
    config_path: Path | str,
    gpu_result_manifest_path: Path | str,
    evaluations: dict[str, Path | str],
    output_path: Path | str,
) -> dict:
    config = load_config(config_path)
    gpu_path = Path(gpu_result_manifest_path)
    gpu = read_json(gpu_path)
    if gpu.get("schema_version") != GPU_RESULT_SCHEMA:
        raise ValueError("unknown GPU result manifest schema")
    if gpu.get("mode") != "dev":
        raise ValueError("candidate selection accepts dev results only")
    if gpu.get("backend") != "qwen" or not gpu.get("scientific_evidence_valid"):
        raise ValueError("fake/smoke GPU results cannot select a candidate")
    if gpu.get("config_sha256") != config_fingerprint(config):
        raise ValueError("GPU result/config mismatch")
    protocol_fingerprint = str(gpu.get("protocol_fingerprint", ""))
    if (
        len(protocol_fingerprint) != 64
        or any(
            char not in "0123456789abcdef"
            for char in protocol_fingerprint
        )
    ):
        raise ValueError("GPU result has no valid protocol fingerprint")
    if set(evaluations) != set(gpu["stages_written"]):
        raise ValueError("one dev evaluation is required for every GPU stage")
    reports = {stage: read_json(path) for stage, path in evaluations.items()}
    for stage, report in reports.items():
        if report.get("policy_mode") != "dev":
            raise ValueError(f"{stage} evaluation is not dev policy")
        if report.get("evidence_mode") != "end_to_end":
            raise ValueError(f"{stage} is not end-to-end retrieval evidence")
        if not report.get("integrity", {}).get("passed"):
            raise ValueError(f"{stage} evaluation integrity failed")
    baseline = reports["R0"]["metrics"]
    gates = config["gates"]
    candidates = []
    total_runtime = float(
        gpu.get("runtime", {}).get("timings", {}).get("total_seconds", 0.0)
    )
    stage_runtimes = gpu.get("runtime", {}).get(
        "stage_ready_seconds", {}
    )
    for stage in gpu["stages_written"]:
        if stage == "R0":
            continue
        metrics = reports[stage]["metrics"]
        delta = {
            "leaf_recall_at_5_delta_vs_r0": round(
                float(metrics["leaf_recall_at_k"])
                - float(baseline["leaf_recall_at_k"]), 6
            ),
            "full_plan_coverage_delta_vs_r0": round(
                float(metrics["full_plan_coverage_rate"])
                - float(baseline["full_plan_coverage_rate"]), 6
            ),
            "docs_f2_delta_vs_r0": round(
                float(metrics["docs_f2_macro"])
                - float(baseline["docs_f2_macro"]), 6
            ),
            "tables_f2_delta_vs_r0": round(
                float(metrics["tables_f2_macro"])
                - float(baseline["tables_f2_macro"]), 6
            ),
        }
        hard = int(
            gpu["stage_artifacts"][stage][
                "hard_constraint_violation_count"
            ]
        )
        checks = {
            "leaf_recall_gate": (
                delta["leaf_recall_at_5_delta_vs_r0"]
                >= float(gates[
                    "minimum_leaf_recall_at_5_delta_vs_r0"
                ])
            ),
            "full_plan_gate": (
                delta["full_plan_coverage_delta_vs_r0"]
                >= float(gates[
                    "minimum_full_plan_coverage_delta_vs_r0"
                ])
            ),
            "docs_f2_gate": (
                delta["docs_f2_delta_vs_r0"]
                >= -float(gates[
                    "maximum_docs_f2_regression_vs_r0"
                ])
            ),
            "tables_f2_gate": (
                delta["tables_f2_delta_vs_r0"]
                >= -float(gates[
                    "maximum_tables_f2_regression_vs_r0"
                ])
            ),
            "hard_constraint_gate": (
                hard <= int(gates[
                    "maximum_hard_constraint_violations"
                ])
            ),
        }
        candidates.append({
            "stage": stage,
            "gate_passed": all(checks.values()),
            "checks": checks,
            "deltas": delta,
            "metrics": {
                key: metrics[key] for key in (
                    "docs_f2_macro",
                    "tables_f2_macro",
                    "leaf_recall_at_k",
                    "full_plan_coverage_rate",
                    "answer_accuracy",
                    "execution_accuracy",
                )
            },
            "hard_constraint_violation_count": hard,
            "runtime_seconds": float(
                stage_runtimes.get(stage, total_runtime)
            ),
            "evaluation_sha256": sha256_file(evaluations[stage]),
            "retrieval_sha256": gpu["stage_artifacts"][stage]["sha256"],
        })
    passed = [item for item in candidates if item["gate_passed"]]
    passed.sort(key=lambda item: (
        -item["deltas"]["full_plan_coverage_delta_vs_r0"],
        -item["deltas"]["leaf_recall_at_5_delta_vs_r0"],
        -float(item["metrics"]["tables_f2_macro"]),
        float(item["runtime_seconds"]),
        item["stage"],
    ))
    selected = passed[0]["stage"] if passed else None
    report = {
        "schema_version": "g3c_dev_selection_v1",
        "config_sha256": config_fingerprint(config),
        "protocol_fingerprint": protocol_fingerprint,
        "gpu_result_manifest_sha256": sha256_file(gpu_path),
        "g3_evaluation_freeze_sha256": (
            config["g3_evaluation_freeze_sha256"]
        ),
        "baseline_evaluation_sha256": sha256_file(evaluations["R0"]),
        "baseline_metrics": {
            key: baseline[key] for key in (
                "docs_f2_macro",
                "tables_f2_macro",
                "leaf_recall_at_k",
                "full_plan_coverage_rate",
                "answer_accuracy",
                "execution_accuracy",
            )
        },
        "gates": gates,
        "candidates": candidates,
        "selected_stage": selected,
        "gate_passed": selected is not None,
    }
    report["selection_fingerprint"] = canonical_json_sha256(report)
    write_json(output_path, report)
    return report


def freeze_selected_candidate(
    *,
    config_path: Path | str,
    gpu_result_manifest_path: Path | str,
    selection_path: Path | str,
    output_path: Path | str,
) -> dict:
    config = load_config(config_path)
    gpu_path = Path(gpu_result_manifest_path)
    selection_path = Path(selection_path)
    gpu = read_json(gpu_path)
    selection = read_json(selection_path)
    _validate_selection(selection)
    if not selection.get("gate_passed") or not selection.get("selected_stage"):
        raise ValueError("no G3C candidate passed the pre-registered dev gate")
    if selection["config_sha256"] != config_fingerprint(config):
        raise ValueError("selection/config mismatch")
    if selection["gpu_result_manifest_sha256"] != sha256_file(gpu_path):
        raise ValueError("selection/GPU manifest mismatch")
    if selection.get("protocol_fingerprint") != gpu.get(
        "protocol_fingerprint"
    ):
        raise ValueError("selection/GPU protocol mismatch")
    selected_stage = selection["selected_stage"]
    selected_row = next(
        item for item in selection["candidates"]
        if item["stage"] == selected_stage
    )
    freeze = {
        "schema_version": CANDIDATE_FREEZE_SCHEMA,
        "candidate_name": f"g3c-qwen-retrieval-v1-{selected_stage.lower()}",
        "selected_stage": selected_stage,
        "gate_passed": True,
        "config_sha256": config_fingerprint(config),
        "protocol_fingerprint": gpu["protocol_fingerprint"],
        "g3_evaluation_freeze_sha256": (
            config["g3_evaluation_freeze_sha256"]
        ),
        "model_revisions": gpu["model_revisions"],
        "instructions_sha256": gpu["instructions_sha256"],
        "dev_payload_fingerprint": gpu["payload_fingerprint"],
        "dev_gpu_run_signature": gpu["run_signature"],
        "dev_gpu_result_manifest_sha256": sha256_file(gpu_path),
        "dev_selection_sha256": sha256_file(selection_path),
        "selected_dev_retrieval_sha256": (
            gpu["stage_artifacts"][selected_stage]["sha256"]
        ),
        "selected_dev_evaluation_sha256": (
            selected_row["evaluation_sha256"]
        ),
        "selected_dev_deltas": selected_row["deltas"],
        "promotion_policy": {
            "splits": ["primary_locked", "hard"],
            "allowed_stage": selected_stage,
            "maximum_runs": 1,
            "threshold_changes_allowed": False,
            "instruction_changes_allowed": False,
            "model_changes_allowed": False,
        },
    }
    freeze["candidate_fingerprint"] = canonical_json_sha256(freeze)
    write_json(output_path, freeze)
    return freeze


def load_candidate_freeze(path: Path | str) -> dict:
    freeze = read_json(path)
    if freeze.get("schema_version") != CANDIDATE_FREEZE_SCHEMA:
        raise ValueError("unknown G3C candidate-freeze schema")
    expected = canonical_json_sha256({
        key: value for key, value in freeze.items()
        if key != "candidate_fingerprint"
    })
    if freeze.get("candidate_fingerprint") != expected:
        raise ValueError("G3C candidate-freeze fingerprint mismatch")
    protocol_fingerprint = str(freeze.get("protocol_fingerprint", ""))
    if (
        len(protocol_fingerprint) != 64
        or any(
            char not in "0123456789abcdef"
            for char in protocol_fingerprint
        )
    ):
        raise ValueError(
            "candidate freeze has no valid protocol fingerprint"
        )
    policy = freeze.get("promotion_policy", {})
    if int(policy.get("maximum_runs", 0)) != 1:
        raise ValueError("candidate freeze does not bind one promotion run")
    if policy.get("allowed_stage") != freeze.get("selected_stage"):
        raise ValueError("candidate freeze selected-stage mismatch")
    return freeze


def _validate_selection(selection: dict) -> None:
    if selection.get("schema_version") != "g3c_dev_selection_v1":
        raise ValueError("unknown G3C selection schema")
    expected = canonical_json_sha256({
        key: value for key, value in selection.items()
        if key != "selection_fingerprint"
    })
    if selection.get("selection_fingerprint") != expected:
        raise ValueError("G3C selection fingerprint mismatch")
