"""Merge high-confidence exact-cell verifier corrections into a checkpoint.

Unlike fill-only merging, this verifier may replace an existing answer, but
only for a direct canonical lookup whose row x column x context resolver has
high confidence. No question-ID allowlist is used.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.units import check_answer_unit
from vifinqa.utils.io import read_jsonl, write_json, write_jsonl


def eligible(base, candidate, route, min_confidence=95.0):
    reasons = []
    if candidate.get("source") != "rule_exact_cell" or candidate.get("status") != "ok":
        reasons.append("not_exact_cell")
    if (route.get("plan") or {}).get("op", "lookup") != "lookup":
        reasons.append("not_direct_lookup")
    if len(route.get("tickers") or []) != 1 or len(route.get("years") or []) != 1:
        reasons.append("not_single_dimension")
    if not route.get("metric_profile_keys"):
        reasons.append("no_canonical_profile")
    confidence = float(candidate.get("detail_conf", 0.0) or 0.0)
    if confidence < min_confidence:
        reasons.append("low_confidence")
    detail = str(candidate.get("detail", ""))
    if not detail.startswith("exact_cell "):
        reasons.append("missing_exact_cell_audit")
    answer = candidate.get("answer")
    if not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
        reasons.append("nonfinite")
    elif check_answer_unit(float(answer), route.get("output_type", "number")):
        reasons.append("unit_sanity")
    query = str(candidate.get("pandas_query", ""))
    try:
        compile(query, "<exact_cell_query>", "eval")
    except SyntaxError:
        reasons.append("not_expression")
    semantic = candidate.get("semantic") or {}
    if not semantic.get("ok") or not semantic.get("dataframe_refs"):
        reasons.append("semantic_grounding")
    base_answer = base.get("answer")
    if (isinstance(base_answer, (int, float)) and isinstance(answer, (int, float))
            and abs(float(base_answer) - float(answer)) <= 0.011):
        reasons.append("same_answer")
    return not reasons, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--min-confidence", type=float, default=95.0)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base))
    candidates = {int(row["id"]): row for row in read_jsonl(Path(args.candidate))}
    retrieval = {int(row["id"]): row for row in read_jsonl(Path(args.retrieval))}
    if len(base_rows) != len(candidates) or len(base_rows) != len(retrieval):
        raise SystemExit("base/candidate/retrieval row counts differ")

    merged, accepted, decisions = [], [], []
    reason_counts = Counter()
    for base in base_rows:
        qid = int(base["id"])
        candidate = candidates[qid]
        route = retrieval[qid]["route"]
        if candidate.get("question") != base.get("question"):
            raise SystemExit(f"question mismatch id={qid}")
        ok, reasons = eligible(base, candidate, route, args.min_confidence)
        if ok:
            row = dict(candidate)
            row["source"] = "exact_cell_verify"
            row["detail"] = str(row.get("detail", "")) + " | exact-cell verifier override"
            merged.append(row)
            accepted.append(qid)
        else:
            merged.append(base)
            reason_counts.update(reasons)
        decisions.append({
            "id": qid, "accepted": ok, "reasons": reasons,
            "base_source": base.get("source"),
            "candidate_source": candidate.get("source"),
            "base_answer": base.get("answer"),
            "candidate_answer": candidate.get("answer"),
            "candidate_confidence": candidate.get("detail_conf", 0.0),
            "profiles": route.get("metric_profile_keys") or [],
        })

    write_jsonl(Path(args.out), merged)
    write_json(Path(args.audit), {
        "policy": "exact_cell_verifier_v1",
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
