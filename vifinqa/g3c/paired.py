"""Local-only paired retrieval diagnostics using the frozen G3B gold corpus."""
from __future__ import annotations

from pathlib import Path

from .common import read_json, read_jsonl, write_json


def build_paired_diagnostics(
    *,
    baseline_evaluation_path: Path | str,
    candidate_evaluation_path: Path | str,
    baseline_submission_dir: Path | str,
    candidate_submission_dir: Path | str,
    corpus_path: Path | str,
    policy_mode: str,
    output_path: Path | str,
) -> dict:
    if policy_mode not in {"dev", "promotion"}:
        raise ValueError("policy_mode must be dev or promotion")
    baseline_eval = read_json(baseline_evaluation_path)
    candidate_eval = read_json(candidate_evaluation_path)
    for name, report in (
        ("baseline", baseline_eval), ("candidate", candidate_eval)
    ):
        if report.get("policy_mode") != policy_mode:
            raise ValueError(f"{name} evaluation policy mismatch")
        if not report.get("integrity", {}).get("passed"):
            raise ValueError(f"{name} evaluation integrity failed")
    baseline_records = {
        str(row["id"]): row for row in baseline_eval["records"]
    }
    candidate_records = {
        str(row["id"]): row for row in candidate_eval["records"]
    }
    if set(baseline_records) != set(candidate_records):
        raise ValueError("paired evaluation ID sets differ")

    baseline_results = _load_submission_results(baseline_submission_dir)
    candidate_results = _load_submission_results(candidate_submission_dir)
    corpus = {
        str(row["id"]): row for row in read_jsonl(corpus_path)
        if (
            row["split"] == "primary_tune"
            if policy_mode == "dev"
            else row["split"] in {"primary_locked", "hard"}
        )
    }
    if set(corpus) != set(baseline_records):
        raise ValueError("paired corpus/evaluation ID sets differ")
    if set(baseline_results) != set(corpus):
        raise ValueError("baseline submission/corpus ID sets differ")
    if set(candidate_results) != set(corpus):
        raise ValueError("candidate submission/corpus ID sets differ")

    gained_leaves = []
    lost_leaves = []
    gained_full_plans = []
    lost_full_plans = []
    false_positive_removed = []
    false_positive_introduced = []
    per_question = []
    for qid in sorted(corpus, key=int):
        gold = set(corpus[qid]["relevant_tables"])
        baseline_tables = set(
            baseline_results[qid].get("relevant_tables", [])[:5]
        )
        candidate_tables = set(
            candidate_results[qid].get("relevant_tables", [])[:5]
        )
        baseline_false = baseline_tables - gold
        candidate_false = candidate_tables - gold
        removed = sorted(baseline_false - candidate_false)
        introduced = sorted(candidate_false - baseline_false)
        base_record = baseline_records[qid]
        cand_record = candidate_records[qid]
        leaf_delta = round(
            float(cand_record["leaf_recall_at_k"])
            - float(base_record["leaf_recall_at_k"]), 6
        )
        if leaf_delta > 0:
            gained_leaves.append(int(qid))
        elif leaf_delta < 0:
            lost_leaves.append(int(qid))
        base_full = bool(base_record["full_plan_coverage"])
        cand_full = bool(cand_record["full_plan_coverage"])
        if cand_full and not base_full:
            gained_full_plans.append(int(qid))
        elif base_full and not cand_full:
            lost_full_plans.append(int(qid))
        for table in removed:
            false_positive_removed.append({
                "id": int(qid), "table_ref": table
            })
        for table in introduced:
            false_positive_introduced.append({
                "id": int(qid), "table_ref": table
            })
        per_question.append({
            "id": int(qid),
            "family": corpus[qid]["family"],
            "leaf_recall_delta": leaf_delta,
            "full_plan_before": base_full,
            "full_plan_after": cand_full,
            "gold_tables_gained": sorted(
                (candidate_tables & gold) - (baseline_tables & gold)
            ),
            "gold_tables_lost": sorted(
                (baseline_tables & gold) - (candidate_tables & gold)
            ),
            "false_positives_removed": removed,
            "false_positives_introduced": introduced,
        })
    report = {
        "schema_version": "g3c_paired_diagnostics_v1",
        "policy_mode": policy_mode,
        "question_count": len(corpus),
        "gained_leaf_question_ids": gained_leaves,
        "lost_leaf_question_ids": lost_leaves,
        "gained_full_plan_question_ids": gained_full_plans,
        "lost_full_plan_question_ids": lost_full_plans,
        "false_positive_tables_removed": false_positive_removed,
        "false_positive_tables_introduced": false_positive_introduced,
        "counts": {
            "gained_leaf_questions": len(gained_leaves),
            "lost_leaf_questions": len(lost_leaves),
            "gained_full_plans": len(gained_full_plans),
            "lost_full_plans": len(lost_full_plans),
            "false_positive_tables_removed": len(false_positive_removed),
            "false_positive_tables_introduced": len(
                false_positive_introduced
            ),
        },
        "per_question": per_question,
    }
    write_json(output_path, report)
    return report


def _load_submission_results(directory: Path | str) -> dict[str, dict]:
    path = Path(directory) / "results.json"
    rows = read_json_array(path)
    return {str(row["id"]): row for row in rows}


def read_json_array(path: Path | str) -> list[dict]:
    import json
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(
        not isinstance(row, dict) for row in value
    ):
        raise ValueError(f"expected JSON array of objects: {path}")
    return value
