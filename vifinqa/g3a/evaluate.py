"""Competition-shaped G3A evaluator and weight-robust promotion gate."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..codegen.executor import run_code
from .builder import GOLD_NAME, MANIFEST_NAME, validate_bundle
from .common import file_sha256, read_jsonl, write_json

EVALUATOR_VERSION = "g3a_competition_vector_v1"


def _ratio(value: float, count: int) -> float:
    return round(float(value) / count, 6) if count else 0.0


def _retrieval_metrics(predicted: list[str], gold: list[str]) -> dict:
    predicted = [str(value) for value in predicted]
    gold_set = {str(value) for value in gold}
    predicted_set = set(predicted)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f2 = (
        5.0 * precision * recall / (4.0 * precision + recall)
        if precision + recall else 0.0
    )
    reciprocal_rank = 0.0
    for rank, value in enumerate(predicted[:5], 1):
        if value in gold_set:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "mrr5": reciprocal_rank,
    }


def _load_predictions(path: Path) -> tuple[list[dict], Path, Path]:
    if path.is_dir():
        results_path = path / "results.json"
        root = path
    else:
        results_path = path
        root = path.parent
    value = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("submission results must be a JSON list")
    return value, root, results_path


def _replay(prediction: dict, root: Path) -> dict:
    query = str(prediction.get("pandas_query") or "").strip()
    if not query:
        return {"ok": False, "value": None, "error": "empty pandas_query"}
    dfs = {}
    try:
        for evidence in prediction.get("evidence", []):
            variable = str(evidence["variable"])
            csv_path = root / str(evidence["csv_path"])
            if variable in dfs:
                return {
                    "ok": False,
                    "value": None,
                    "error": f"duplicate variable {variable}",
                }
            dfs[variable] = pd.read_csv(csv_path)
        result = run_code(query, dfs, timeout=10)
    except Exception as exc:
        return {
            "ok": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    value = result.get("value")
    ok = result.get("status") == "ok" and value is not None
    return {
        "ok": ok,
        "value": value,
        "error": str(result.get("error") or "")[:500],
    }


def _summarize(records: list[dict]) -> dict:
    count = len(records)
    if not count:
        return {"count": 0}
    output: dict[str, float | int] = {"count": count}
    for prefix in ("docs", "tables"):
        for metric in ("precision", "recall", "f2", "mrr5"):
            name = (
                f"{prefix}_{metric}_macro"
                if metric != "mrr5" else f"{prefix}_mrr5"
            )
            output[name] = _ratio(
                sum(float(row[f"{prefix}_{metric}"]) for row in records), count
            )
    for item_key, count_key, rate_key in (
        ("answer_correct", "answer_correct", "answer_accuracy"),
        ("execution_correct", "execution_correct", "execution_accuracy"),
        ("execution_ran", "execution_ran", "execution_run_rate"),
    ):
        total = sum(bool(row[item_key]) for row in records)
        output[count_key] = total
        output[rate_key] = _ratio(total, count)
    return output


def _breakdown(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: _summarize(groups[name]) for name in sorted(groups)}


def _scenario_scores(metrics: dict, scenarios: dict) -> dict[str, dict]:
    output = {}
    for name, weights in scenarios.items():
        total_weight = sum(float(value) for value in weights.values())
        if total_weight <= 0:
            raise ValueError(
                f"scenario {name}: weights must sum to a positive value"
            )
        score = sum(
            float(metrics[key]) * float(weight)
            for key, weight in weights.items()
        ) / total_weight
        output[name] = {
            "score": round(score, 6),
            "weights": {
                key: float(value) for key, value in weights.items()
            },
        }
    return output


def evaluate_submission(
    submission: Path | str,
    bundle_dir: Path | str,
    config_path: Path | str,
    *,
    output_path: Path | str | None = None,
    require_hard_approved: bool = True,
) -> dict:
    bundle_dir = Path(bundle_dir)
    config_path = Path(config_path)
    validate_bundle(
        bundle_dir, require_hard_approved=require_hard_approved
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gold_rows = read_jsonl(bundle_dir / GOLD_NAME)
    predictions, submission_root, results_path = _load_predictions(
        Path(submission)
    )

    prediction_map: dict[int, dict] = {}
    duplicate_ids: list[int] = []
    for prediction in predictions:
        qid = int(prediction["id"])
        if qid in prediction_map:
            duplicate_ids.append(qid)
        prediction_map[qid] = prediction
    gold_ids = {int(row["id"]) for row in gold_rows}
    prediction_ids = set(prediction_map)
    missing_ids = sorted(gold_ids - prediction_ids)
    extra_ids = sorted(prediction_ids - gold_ids)

    records: list[dict] = []
    question_mismatch: list[int] = []
    nonfinite_answer: list[int] = []
    duplicate_retrieval: list[int] = []
    for gold in gold_rows:
        qid = int(gold["id"])
        prediction = prediction_map.get(qid, {})
        if (
            prediction
            and str(prediction.get("question", "")) != gold["question"]
        ):
            question_mismatch.append(qid)
        predicted_docs = list(prediction.get("relevant_docs") or [])
        predicted_tables = list(prediction.get("relevant_tables") or [])
        if (
            len(predicted_docs) != len(set(map(str, predicted_docs)))
            or len(predicted_tables) != len(set(map(str, predicted_tables)))
        ):
            duplicate_retrieval.append(qid)
        docs = _retrieval_metrics(
            predicted_docs, gold["relevant_docs"]
        )
        tables = _retrieval_metrics(
            predicted_tables, gold["relevant_tables"]
        )
        try:
            predicted_answer = float(prediction.get("answer"))
            finite = math.isfinite(predicted_answer)
        except (TypeError, ValueError):
            predicted_answer, finite = 0.0, False
        if not finite:
            nonfinite_answer.append(qid)
        tolerance = float(gold["tolerance"])
        answer_correct = bool(
            finite
            and math.isclose(
                predicted_answer,
                float(gold["answer"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
        replay = (
            _replay(prediction, submission_root)
            if prediction
            else {
                "ok": False,
                "value": None,
                "error": "missing prediction",
            }
        )
        execution_value = replay["value"]
        execution_correct = bool(
            replay["ok"]
            and execution_value is not None
            and math.isclose(
                float(execution_value),
                float(gold["answer"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
        records.append({
            "id": qid,
            "split": gold["split"],
            "set": gold["set"],
            "stratum": gold["stratum"],
            "operator": gold["operator"],
            "difficulty": gold["difficulty"],
            "review_status": gold["review"]["status"],
            **{f"docs_{key}": value for key, value in docs.items()},
            **{f"tables_{key}": value for key, value in tables.items()},
            "gold_answer": float(gold["answer"]),
            "predicted_answer": predicted_answer if finite else None,
            "answer_correct": answer_correct,
            "execution_ran": bool(replay["ok"]),
            "execution_value": execution_value,
            "execution_correct": execution_correct,
            "execution_error": replay["error"],
        })

    integrity = {
        "passed": not (
            duplicate_ids
            or missing_ids
            or extra_ids
            or question_mismatch
            or nonfinite_answer
            or duplicate_retrieval
        ),
        "prediction_count": len(predictions),
        "gold_count": len(gold_rows),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "question_mismatch_ids": question_mismatch,
        "nonfinite_answer_ids": nonfinite_answer,
        "duplicate_retrieval_ids": duplicate_retrieval,
    }
    metrics = _summarize(records)
    report = {
        "schema_version": "g3a_evaluation_report_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "provenance": {
            "bundle_manifest_sha256": file_sha256(
                bundle_dir / MANIFEST_NAME
            ),
            "gold_sha256": file_sha256(bundle_dir / GOLD_NAME),
            "submission_results_sha256": file_sha256(results_path),
            "config_sha256": file_sha256(config_path),
        },
        "metric_definitions": {
            "retrieval": (
                "per-query macro precision/recall/F2; MRR@5 is reciprocal "
                "rank of first relevant item"
            ),
            "answer_accuracy": (
                "submitted answer equals gold within per-record absolute "
                "tolerance"
            ),
            "execution_accuracy": (
                "submitted pandas_query replays on submitted CSV evidence "
                "and equals gold within tolerance"
            ),
            "weight_policy": (
                "private weights unknown; raw metric vector is primary, "
                "scenarios are sensitivity checks only"
            ),
        },
        "integrity": integrity,
        "metrics": metrics,
        "weight_scenarios": _scenario_scores(
            metrics, config["weight_scenarios"]
        ),
        "breakdown": {
            "split": _breakdown(records, "split"),
            "set": _breakdown(records, "set"),
            "operator": _breakdown(records, "operator"),
            "difficulty": _breakdown(records, "difficulty"),
        },
        "records": records,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def _metric_get(report: dict, path: str) -> float:
    current: object = report
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"report has no metric path: {path}")
        current = current[part]
    return float(current)


def compare_reports(
    baseline_path: Path | str,
    candidate_path: Path | str,
    config_path: Path | str,
    *,
    output_path: Path | str | None = None,
) -> dict:
    baseline = json.loads(
        Path(baseline_path).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        Path(candidate_path).read_text(encoding="utf-8")
    )
    config = json.loads(
        Path(config_path).read_text(encoding="utf-8")
    )
    promotion = config["promotion_gate"]
    paths = list(promotion["metric_paths"])
    deltas = {
        path: round(
            _metric_get(candidate, path) - _metric_get(baseline, path), 6
        )
        for path in paths
    }
    regressions = {
        path: delta
        for path, delta in deltas.items()
        if delta < -float(promotion["max_regression"][path])
    }
    material = {
        path: delta
        for path, delta in deltas.items()
        if delta >= float(promotion["min_material_gain"][path])
    }
    scenario_deltas = {
        name: round(
            float(candidate["weight_scenarios"][name]["score"])
            - float(baseline["weight_scenarios"][name]["score"]),
            6,
        )
        for name in baseline["weight_scenarios"]
    }
    hard_answer_delta = round(
        float(candidate["breakdown"]["set"]["hard"]["answer_accuracy"])
        - float(baseline["breakdown"]["set"]["hard"]["answer_accuracy"]),
        6,
    )
    blockers = []
    if not candidate.get("integrity", {}).get("passed", False):
        blockers.append("candidate_integrity_failed")
    if regressions:
        blockers.append("metric_regression_guard_failed")
    if any(delta < 0.0 for delta in scenario_deltas.values()):
        blockers.append("unknown_weight_sensitivity_failed")
    if (
        hard_answer_delta
        < -float(promotion["hard_answer_max_regression"])
    ):
        blockers.append("hard_set_answer_guard_failed")
    if not material:
        blockers.append("no_material_gain")
    report = {
        "schema_version": "g3a_promotion_report_v1",
        "decision": "promote" if not blockers else "block",
        "policy": (
            "Pareto-style guardrails plus unknown-weight scenario sensitivity"
        ),
        "deltas": deltas,
        "material_gains": material,
        "regressions": regressions,
        "scenario_deltas": scenario_deltas,
        "hard_answer_delta": hard_answer_delta,
        "blockers": blockers,
        "provenance": {
            "baseline_sha256": file_sha256(baseline_path),
            "candidate_sha256": file_sha256(candidate_path),
            "config_sha256": file_sha256(config_path),
        },
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
