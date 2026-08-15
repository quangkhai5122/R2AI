"""Read-only quality audit summaries for completed P2.4 tune gold."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from .p24 import canonical_sha256


def _walk_ops(node: dict[str, Any], counter: Counter[str]) -> None:
    if not isinstance(node, dict) or node.get("kind") != "op":
        return
    counter[str(node["op"])] += 1
    for arg in node.get("args", []):
        _walk_ops(arg, counter)


def build_tune_gold_audit(
    gold: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize coverage, provenance, operations and review flags.

    Strict cell/AST/replay validation is intentionally a separate gate.  This
    function assumes that gate has passed and adds descriptive QA only.
    """
    qids = {int(row["id"]) for row in questions}
    gids = {int(row["id"]) for row in gold}
    sids = {int(row["id"]) for row in specs}
    if len(gids) != len(gold) or gids != qids or sids != qids:
        raise ValueError("gold/spec IDs must uniquely and exactly cover tune questions")
    spec_map = {int(row["id"]): row for row in specs}
    output_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    exact_cells: set[tuple[str, int, int, int]] = set()
    exact_tables: set[tuple[str, int]] = set()
    reports: set[str] = set()
    evidence_counts: list[int] = []
    answers: defaultdict[str, list[float]] = defaultdict(list)
    flags: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    expression_groups: defaultdict[str, list[int]] = defaultdict(list)
    for record in gold:
        qid = int(record["id"])
        output_type = str(record["output"]["type"])
        value = float(record["output"]["value"])
        output_counts[output_type] += 1
        stratum_counts[str(record["stratum"])] += 1
        answers[output_type].append(value)
        evidence_counts.append(len(record["evidence"]))
        _walk_ops(record["ast"], op_counts)
        local_cells = []
        for evidence in record["evidence"]:
            key = (str(evidence["report_id"]), int(evidence["table_pos"]),
                   int(evidence["row"]), int(evidence["col"]))
            local_cells.append(key)
            exact_cells.add(key)
            exact_tables.add((key[0], key[1]))
            reports.add(key[0])
        expression = str(spec_map[qid]["expression"])
        expression_groups[expression].append(qid)
        reasons = []
        if not math.isfinite(value):
            reasons.append("non_finite")
        if value == 0:
            reasons.append("zero_answer_review")
        if output_type in {"percent", "percentage_point"} and abs(value) > 100:
            reasons.append("percent_magnitude_gt_100")
        if output_type == "ratio" and abs(value) > 100:
            reasons.append("ratio_magnitude_gt_100")
        if abs(value) >= 1e20:
            reasons.append("sentinel_or_scale_risk")
        if len(record["evidence"]) >= 25:
            reasons.append("large_evidence_graph")
        if len(local_cells) != len(set(local_cells)):
            reasons.append("duplicate_exact_cell")
        if reasons:
            flags.append({"id": qid, "answer": value, "output_type": output_type,
                          "reasons": reasons})
        rows.append({
            "id": qid, "stratum": str(record["stratum"]),
            "output_type": output_type, "answer": value,
            "unit": str(record["output"]["unit"]),
            "evidence_count": len(record["evidence"]),
            "table_count": len({(item[0], item[1]) for item in local_cells}),
            "root_op": str(record["ast"]["op"]),
            "notes": str(record["annotator_notes"]),
        })
    answer_summary = {}
    for output_type, values in sorted(answers.items()):
        answer_summary[output_type] = {
            "count": len(values), "min": min(values), "max": max(values),
            "mean": mean(values), "median": median(values),
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
        }
    repeated = [
        {"ids": ids, "expression": expression}
        for expression, ids in expression_groups.items() if len(ids) > 1
    ]
    return {
        "schema_version": "p24_tune_gold_audit_v1",
        "status": "descriptive_after_strict_validation",
        "count": len(gold),
        "gold_sha256": canonical_sha256(gold),
        "questions_sha256": canonical_sha256(questions),
        "specs_sha256": canonical_sha256(specs),
        "locked_opened": False,
        "output_type_counts": dict(sorted(output_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "operation_counts": dict(op_counts.most_common()),
        "answer_summary": answer_summary,
        "provenance": {
            "evidence_total": sum(evidence_counts),
            "evidence_unique_exact_cells": len(exact_cells),
            "unique_tables": len(exact_tables), "unique_reports": len(reports),
            "evidence_per_question_min": min(evidence_counts),
            "evidence_per_question_median": median(evidence_counts),
            "evidence_per_question_max": max(evidence_counts),
        },
        "review_flags": sorted(flags, key=lambda item: item["id"]),
        "repeated_expressions": repeated,
        "records": sorted(rows, key=lambda item: item["id"]),
    }
