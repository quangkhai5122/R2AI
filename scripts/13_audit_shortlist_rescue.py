"""Audit opt-in no-candidate shortlist rescue without calling an LLM.

By default the script targets only records whose saved Selection trace says
``no_candidates``.  It writes a bounded JSON audit; retrieval, store and codegen
inputs are read-only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.generate import QuestionBundle
from vifinqa.extraction.build_store import Store
from vifinqa.utils.io import read_jsonl, setup_stdout, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(rows: list[dict], label: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        qid = int(row["id"])
        if qid in out:
            raise ValueError(f"duplicate {label} id: {qid}")
        out[qid] = row
    return out


def audit_rescue(retrieval_path: Path, codegen_path: Path, store_dir: Path,
                 out_path: Path, *, k: int = 0, table_k: int = 20,
                 min_score: float = 28.0, top_n: int = 12) -> dict:
    retrieval = _unique(read_jsonl(retrieval_path), "retrieval")
    codegen = _unique(read_jsonl(codegen_path), "codegen")
    if set(retrieval) != set(codegen):
        raise ValueError("retrieval/codegen id universes differ")
    target_ids = sorted(
        qid for qid, row in codegen.items()
        if (row.get("selection_trace") or {}).get("outcome") == "no_candidates"
    )
    store = Store(store_dir, cache_size=6)
    rows = []
    recovered = 0
    recovered_modes: dict[str, int] = {}
    for qid in target_ids:
        rec = retrieval[qid]
        rescue_bundle = QuestionBundle(
            rec, store, k=k, rescue_no_candidates=True,
            rescue_table_k=table_k, rescue_min_score=min_score,
        )
        candidates = rescue_bundle.shortlist(None, top_n=top_n)
        if not rescue_bundle.shortlist_trace.get("rescue_applied"):
            raise RuntimeError(
                f"saved no-candidate id {qid} now has a strict shortlist; "
                "scorer/retrieval drift detected"
            )
        recovered += bool(candidates)
        rescue_mode = str(rescue_bundle.shortlist_trace.get("rescue_mode", "none"))
        recovered_modes[rescue_mode] = recovered_modes.get(rescue_mode, 0) + 1
        route = rec.get("route") or {}
        rows.append({
            "id": qid,
            "question": rec.get("question", ""),
            "tickers": route.get("tickers") or [],
            "years": route.get("years") or [],
            "operation": (route.get("plan") or {}).get("op", "lookup"),
            "output_type": route.get("output_type", "number"),
            "strict_tables": int(
                rescue_bundle.shortlist_trace.get("strict_table_count", 0)
            ),
            "rescue_tables": len(rescue_bundle.tables),
            "rescue_candidates": len(candidates),
            "rescue_mode": rescue_mode,
            "candidates": [
                {
                    "report_id": c.report_id,
                    "table_pos": c.table_pos,
                    "row": c.row,
                    "col": c.col,
                    "label": c.label,
                    "col_name": c.col_name,
                    "score": c.score,
                    "fact_slot": c.fact_slot,
                }
                for c in candidates[:top_n]
            ],
        })
    audit = {
        "schema_version": 1,
        "policy": "strict_empty_then_2d_schema_rescue_v1",
        "parameters": {
            "k": k, "table_k": table_k, "min_score": min_score,
            "top_n": top_n,
        },
        "inputs": {
            "retrieval": {"path": str(retrieval_path),
                          "sha256": _sha256(retrieval_path)},
            "codegen": {"path": str(codegen_path),
                        "sha256": _sha256(codegen_path)},
        },
        "counts": {
            "target_no_candidates": len(target_ids),
            "recovered_nonempty": recovered,
            "still_empty": len(target_ids) - recovered,
            "by_mode": dict(sorted(recovered_modes.items())),
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, audit)
    return audit


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", default="artifacts/retrieval.jsonl")
    parser.add_argument(
        "--codegen",
        default=("artifacts/submission_sel14b_factaware/"
                 "codegen_sel14b_factaware.jsonl"),
    )
    parser.add_argument("--store", default="artifacts/store")
    parser.add_argument("--out", default="artifacts/shortlist_rescue_audit.json")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--table-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=28.0)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    if args.k < 0 or args.table_k < 1 or args.top_n < 1:
        parser.error("--k must be >=0; --table-k/--top-n must be >=1")
    if not 0.0 <= args.min_score <= 100.0:
        parser.error("--min-score must be between 0 and 100")
    audit = audit_rescue(
        Path(args.retrieval), Path(args.codegen), Path(args.store),
        Path(args.out), k=args.k, table_k=args.table_k,
        min_score=args.min_score, top_n=args.top_n,
    )
    print(json.dumps(audit["counts"], ensure_ascii=False))
    print(f"audit -> {args.out}")


if __name__ == "__main__":
    main()
