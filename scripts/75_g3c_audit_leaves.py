"""Audit label-blind G3C leaf formation on question-only inputs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.extraction.build_store import Store
from vifinqa.g3c.common import read_jsonl, write_json
from vifinqa.g3c.leaves import decompose_atomic_leaves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    retrieval = read_jsonl(args.retrieval)
    if len(questions) != len(retrieval):
        raise SystemExit("question/retrieval count mismatch")
    store = Store(Path(args.store_dir))
    route_by_id = {str(row["id"]): row for row in retrieval}
    operation_counts: Counter[str] = Counter()
    leaf_counts: Counter[int] = Counter()
    route_fact_counts: Counter[int] = Counter()
    missing_exact = []
    invariant_errors = []
    records = []
    for question in questions:
        qid = str(question["id"])
        baseline = route_by_id.get(qid)
        if baseline is None or baseline["question"] != question["question"]:
            raise SystemExit(f"missing/mismatched baseline route for {qid}")
        leaves = decompose_atomic_leaves(
            question["question"], baseline["route"], store
        )
        operation_counts.update(leaf.operation for leaf in leaves)
        leaf_counts[len(leaves)] += 1
        route_facts = len(
            (baseline["route"].get("plan") or {}).get("facts", [])
        )
        route_fact_counts[route_facts] += 1
        for leaf in leaves:
            if not leaf.report_ids:
                missing_exact.append({
                    "id": question["id"], "leaf": leaf.to_dict()
                })
        operations = {leaf.operation for leaf in leaves}
        if "cagr" in operations:
            periods = {leaf.period_year for leaf in leaves}
            years = sorted({
                int(value) for value in __import__("re").findall(
                    r"(?<!\d)(20\d{2})(?!\d)", question["question"]
                )
            })
            if len(years) >= 2 and periods != {years[0], years[-1]}:
                invariant_errors.append({
                    "id": question["id"],
                    "reason": "CAGR_not_endpoint_only",
                })
        if "scope_delta" in operations:
            scopes = {leaf.doc_type for leaf in leaves}
            if scopes != {"consolidated", "separate"}:
                invariant_errors.append({
                    "id": question["id"],
                    "reason": "scope_delta_missing_scope",
                })
        records.append({
            "id": question["id"],
            "route_fact_count": route_facts,
            "leaf_count": len(leaves),
            "operations": sorted(operations),
            "leaf_ids": [leaf.leaf_id for leaf in leaves],
        })
    report = {
        "schema_version": "g3c_leaf_audit_v1",
        "question_count": len(questions),
        "input_fields_used": ["id", "question", "route", "metric_registry", "store_metadata"],
        "gold_fields_used": [],
        "operation_counts": dict(sorted(operation_counts.items())),
        "leaf_count_distribution": {
            str(key): value for key, value in sorted(leaf_counts.items())
        },
        "route_fact_count_distribution": {
            str(key): value for key, value in sorted(route_fact_counts.items())
        },
        "missing_exact_report_leaves": missing_exact,
        "invariant_errors": invariant_errors,
        "passed": not missing_exact and not invariant_errors,
        "records": records,
    }
    write_json(args.out, report)
    print(json.dumps({
        "passed": report["passed"],
        "question_count": report["question_count"],
        "leaf_count_distribution": report["leaf_count_distribution"],
        "missing_exact_report_leaves": len(missing_exact),
        "invariant_errors": len(invariant_errors),
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
