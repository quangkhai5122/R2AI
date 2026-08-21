"""Merge high-confidence canonical-v2 answers into a frozen checkpoint.

Only checkpoint ``source=none`` rows are eligible. The gate is intentionally
strict and batch-wide; no question-id allowlist is used.
"""
import argparse
import ast
import json
import re
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.units import check_answer_unit
from vifinqa.finance.metrics_v2 import best_row_profile, qualifier_flags
from vifinqa.utils.viet_text import norm
from vifinqa.utils.io import read_jsonl, write_json, write_jsonl


_SCENARIO_RE = re.compile(r"\b(gia su|kich ban|neu .* thi)\b")
_DERIVED_RE = re.compile(
    r"\b(ty le|ty trong|chiem|he so|vong quay|hao mon|bien loi nhuan|"
    r"gap bao nhieu|tren tong|chia cho)\b"
)


def _query_labels(query):
    labels = []
    try:
        tree = ast.parse(str(query), mode="eval")
    except SyntaxError:
        return labels
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "contains" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            labels.append(node.args[0].value)
    return labels


def _eligible(base, candidate, route, min_confidence):
    reasons = []
    if base.get("source") != "none":
        reasons.append("base_not_none")
    if candidate.get("source") == "none" or candidate.get("status") != "ok":
        reasons.append("candidate_not_ok")
    profiles = route.get("metric_profile_keys") or []
    if not profiles:
        reasons.append("no_v2_profile")
    confidence = float(candidate.get("detail_conf", 0.0) or 0.0)
    if confidence < min_confidence:
        reasons.append("low_confidence")
    detail = str(candidate.get("detail", ""))
    if "UNIT-WARN" in detail or "AMBIGUOUS" in detail:
        reasons.append("warning")
    answer = candidate.get("answer")
    if not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
        reasons.append("nonfinite")
    elif check_answer_unit(float(answer), route.get("output_type", "number")):
        reasons.append("unit_sanity")
    query = str(candidate.get("pandas_query", ""))
    try:
        compile(query, "<query>", "eval")
    except SyntaxError:
        reasons.append("not_expression")

    question_norm = norm(route.get("question", ""))
    plan_op = (route.get("plan") or {}).get("op", "lookup")
    if _SCENARIO_RE.search(question_norm):
        reasons.append("scenario_requires_formula")
    if "gap bao nhieu lan" in question_norm and plan_op == "growth_pct":
        reasons.append("times_ratio_misread_as_growth")
    formula_source = candidate.get("source") == "rule_formula"
    derived_ok = formula_source or plan_op in {"growth_pct", "ratio", "margin", "cagr"}
    if _DERIVED_RE.search(question_norm) and not derived_ok:
        reasons.append("derived_question_without_formula")

    labels = _query_labels(query)
    if "quy khen thuong" in question_norm and not any(
            "quy khen thuong" in norm(label) for label in labels):
        reasons.append("target_metric_missing_from_query")
    if not labels:
        reasons.append("no_grounded_row_labels")
    route_profiles = route.get("metric_profile_keys") or []
    label_text = " ".join(norm(label) for label in labels)
    if "kha nang thanh toan lai vay" in question_norm:
        if "loi nhuan truoc" not in label_text or "chi phi lai vay" not in label_text:
            reasons.append("interest_coverage_operands_missing")
    if "long_term_borrowing_related" in route_profiles:
        if "vay" not in label_text or "lai vay" in label_text:
            reasons.append("borrowing_target_uses_interest_row")
    if "investment_financial_long_term" in route_profiles:
        if "dau tu" not in label_text or "du phong" in label_text:
            reasons.append("investment_target_uses_provision_row")
    if ("borrowing_interest_expense" in route_profiles
            and "kha nang thanh toan lai vay" in question_norm):
        reasons.append("interest_expense_is_selector_not_target")
    for label in labels:
        route_profiles = route.get("metric_profile_keys") or []
        column_profile = any(key == "bank_customer_loans_financial_asset_receivables"
                             for key in route_profiles)
        profile, _bonus, _reason = best_row_profile(route, label, "")
        if profile is None and not column_profile:
            reasons.append("row_profile_mismatch")
            break
    asked_flags = qualifier_flags(question_norm)
    if asked_flags.get("stock_flow") == "flow" and labels:
        row_flags = [qualifier_flags(label) for label in labels]
        if not any(flags.get("stock_flow") == "flow" for flags in row_flags):
            reasons.append("flow_question_uses_stock_row")
    return not reasons, list(dict.fromkeys(reasons))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--min-confidence", type=float, default=90.0)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base))
    candidates = {int(row["id"]): row for row in read_jsonl(Path(args.candidate))}
    retrieval = {int(row["id"]): row for row in read_jsonl(Path(args.retrieval))}
    if len(base_rows) != len(candidates) or len(base_rows) != len(retrieval):
        raise SystemExit("base/candidate/retrieval row counts differ")

    merged, decisions, accepted = [], [], []
    reason_counts = Counter()
    for base in base_rows:
        qid = int(base["id"])
        candidate = candidates[qid]
        route = retrieval[qid]["route"]
        if candidate.get("question") != base.get("question"):
            raise SystemExit(f"question mismatch id={qid}")
        ok, reasons = _eligible(base, candidate, route, args.min_confidence)
        if ok:
            row = dict(candidate)
            row["source"] = "metric_v2_fill"
            row["detail"] = (str(row.get("detail", ""))
                             + " | canonical_metric_v2 batch fill")
            merged.append(row)
            accepted.append(qid)
        else:
            merged.append(base)
            reason_counts.update(reasons)
        decisions.append({
            "id": qid, "accepted": ok, "reasons": reasons,
            "base_source": base.get("source"),
            "candidate_source": candidate.get("source"),
            "candidate_confidence": candidate.get("detail_conf", 0.0),
            "profiles": route.get("metric_profile_keys") or [],
        })

    write_jsonl(Path(args.out), merged)
    write_json(Path(args.audit), {
        "policy": "canonical_metric_v2_fill_only_v1",
        "min_confidence": args.min_confidence,
        "rows": len(merged), "accepted": len(accepted),
        "accepted_ids": accepted,
        "reason_counts": dict(reason_counts),
        "decisions": decisions,
    })
    print(f"merged {len(accepted)}/{len(merged)} rows -> {args.out}")
    print("accepted ids:", accepted)
    print("rejections:", dict(reason_counts))


if __name__ == "__main__":
    main()
