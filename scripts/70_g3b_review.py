"""Independent source-cell and formula recomputation for required G3B gold."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from vifinqa.extraction.build_store import Store
from vifinqa.g3b.builder import REVIEW_QUEUE_NAME
from vifinqa.g3b.common import read_jsonl, write_jsonl
from vifinqa.utils.io import setup_stdout


def _recompute(family: str, evidence: list[dict]) -> float:
    values = [float(row["base_value"]) for row in evidence]
    if family == "ranking_argmax":
        return float(max(evidence, key=lambda row: row["base_value"])["period_year"])
    if family == "ranking_argmin":
        return float(min(evidence, key=lambda row: row["base_value"])["period_year"])
    if family == "count_positive":
        return float(sum(value > 0 for value in values))
    if family == "cagr":
        periods = int(evidence[0]["period_year"]) - int(evidence[1]["period_year"])
        return ((values[0] / values[1]) ** (1 / periods) - 1) * 100
    if family == "percentage_point_change":
        return ((values[0] / values[1]) - (values[2] / values[3])) * 100
    if family == "nested_margin_average":
        return ((values[0] / values[1]) + (values[2] / values[3])) / 2 * 100
    if family == "simple_average":
        return sum(values) / len(values) / 1e6
    if family in {"note_lookup", "prior_period_lookup"}:
        return values[0] / 1e6
    if family == "scope_delta":
        return (values[0] - values[1]) / 1e6
    if family == "debt_assets_ratio":
        return values[0] / values[1]
    raise ValueError(f"unsupported review family: {family}")


def _verify_source(store: Store, evidence: dict) -> None:
    cells = store.cells_of(
        evidence["ticker"], [evidence["report_id"]]
    )
    hit = cells[
        (cells.table_pos == int(evidence["table_pos"]))
        & (cells.row == int(evidence["row"]))
        & (cells.col == int(evidence["col"]))
    ]
    if len(hit) != 1:
        raise ValueError(
            f"source cell cardinality != 1: {evidence['fact_id']}"
        )
    row = hit.iloc[0]
    if not math.isclose(
        float(row.value), float(evidence["value"]),
        rel_tol=0.0, abs_tol=0.0,
    ):
        raise ValueError(f"source value drift: {evidence['fact_id']}")
    if not math.isclose(
        float(row.unit_scale), float(evidence["unit_scale"]),
        rel_tol=0.0, abs_tol=0.0,
    ):
        raise ValueError(f"source unit drift: {evidence['fact_id']}")
    line = store.line_no_of(
        evidence["report_id"], int(evidence["table_pos"])
    )
    if line != int(evidence["table_line"]):
        raise ValueError(f"source table-line drift: {evidence['fact_id']}")


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--corpus-dir", default="data/g3b_v1")
    parser.add_argument(
        "--out", default="data/g3b_v1/g3b_reviews.jsonl"
    )
    args = parser.parse_args()
    queue = read_jsonl(Path(args.corpus_dir) / REVIEW_QUEUE_NAME)
    if not queue:
        raise SystemExit("G3B review queue is empty")
    store = Store(Path(args.store_dir), cache_size=4)
    reviews = []
    for row in tqdm(
        queue, desc="G3B independent review", unit="record"
    ):
        for evidence in row["evidence"]:
            _verify_source(store, evidence)
        answer = _recompute(row["family"], row["evidence"])
        if not math.isclose(
            round(float(answer), 2),
            float(row["answer"]),
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise ValueError(
                f"independent answer mismatch id={row['id']}: "
                f"{answer} != {row['answer']}"
            )
        reviews.append({
            "id": row["id"],
            "subject_sha256": row["subject_sha256"],
            "status": "approved",
            "reviewer": "Codex agent independent evidence audit",
            "method": "source_cell_recheck_plus_family_recompute_v1",
            "source_cells_verified": len(row["evidence"]),
            "independent_answer": round(float(answer), 8),
            "human_domain_review": False,
        })
    write_jsonl(args.out, reviews)
    print(json.dumps({
        "reviewed": len(reviews),
        "approved": len(reviews),
        "human_domain_review": False,
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
