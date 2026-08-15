"""CPU-only audit of P2.2 atomic shortlist coverage and prompt size."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.generate import QuestionBundle
from vifinqa.extraction.build_store import Store
from vifinqa.utils.io import read_jsonl, setup_stdout


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(retrieval_path: Path, store_dir: Path, mask_path: Path,
          rescue: bool, table_k: int, min_score: float,
          limit: int = 0) -> dict:
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    ids = [int(x) for x in mask.get("ids") or []]
    if not ids or len(ids) != len(set(ids)):
        raise SystemExit("mask must contain unique non-empty ids")
    if limit:
        ids = ids[:limit]
    wanted = set(ids)
    retrieval = {int(row["id"]): row for row in read_jsonl(retrieval_path)}
    missing = wanted - set(retrieval)
    if missing:
        raise SystemExit(f"mask ids absent from retrieval: {sorted(missing)}")
    store = Store(store_dir, cache_size=4)
    rows, modes, families = [], Counter(), Counter()
    started = time.time()
    for position, qid in enumerate(ids, 1):
        t0 = time.time()
        bundle = QuestionBundle(
            retrieval[qid], store, k=0,
            rescue_no_candidates=rescue,
            rescue_table_k=table_k,
            rescue_min_score=min_score,
        )
        candidates = bundle.shortlist_v2(top_n=24)
        messages = bundle.select_v2_messages()
        trace = dict(bundle.shortlist_trace)
        atomic = dict(trace.get("atomic_slots") or {})
        modes[str(trace.get("rescue_mode") or "none")] += 1
        families.update(atomic.get("families") or ["routed_fact"])
        rows.append({
            "id": qid,
            "candidate_count": len(candidates),
            "raw_candidate_count": int(trace.get("raw_candidate_count") or 0),
            "atomic_slots_required": int(trace.get("atomic_fact_slots_required") or 0),
            "atomic_slots_present": int(trace.get("atomic_fact_slots_present") or 0),
            "atomic_fact_complete": bool(trace.get("atomic_fact_complete")),
            "semantic_slots_present": int(trace.get("semantic_fact_slots_present") or 0),
            "semantic_route_grounded": bool(trace.get("semantic_route_grounded")),
            "semantic_fact_complete": bool(trace.get("semantic_fact_complete")),
            "metric_grounding_rejections": trace.get("metric_grounding_rejections") or {},
            "atomic_families": atomic.get("families") or [],
            "atomic_truncated": bool(atomic.get("truncated")),
            "rescue_mode": trace.get("rescue_mode"),
            "prompt_chars": sum(len(str(message.get("content", ""))) for message in messages),
            "seconds": round(time.time() - t0, 3),
        })
        if position % 10 == 0 or position == len(ids):
            print(f"audit {position}/{len(ids)} | elapsed={(time.time()-started)/60:.1f}m",
                  flush=True)
    prompt_chars = [row["prompt_chars"] for row in rows]
    seconds = [row["seconds"] for row in rows]
    return {
        "schema_version": "p22_shortlist_audit_v2_semantic",
        "policy": {
            "mode": "select_v2", "k": 0, "top_n": 24,
            "rescue": rescue, "rescue_table_k": table_k,
            "rescue_min_score": min_score,
        },
        "inputs": {
            "retrieval": {"path": str(retrieval_path), "sha256": _sha256(retrieval_path)},
            "mask": {"path": str(mask_path), "sha256": _sha256(mask_path),
                     "name": mask.get("name")},
        },
        "summary": {
            "rows": len(rows),
            "nonempty": sum(row["candidate_count"] > 0 for row in rows),
            "empty": sum(row["candidate_count"] == 0 for row in rows),
            "atomic_fact_complete": sum(row["atomic_fact_complete"] for row in rows),
            "semantic_route_grounded": sum(row["semantic_route_grounded"] for row in rows),
            "semantic_fact_complete": sum(row["semantic_fact_complete"] for row in rows),
            "semantic_slots_present": sum(row["semantic_slots_present"] for row in rows),
            "atomic_truncated": sum(row["atomic_truncated"] for row in rows),
            "slots_required": sum(row["atomic_slots_required"] for row in rows),
            "slots_present": sum(row["atomic_slots_present"] for row in rows),
            "rescue_modes": dict(sorted(modes.items())),
            "families": dict(sorted(families.items())),
            "prompt_chars": {
                "min": min(prompt_chars),
                "median": statistics.median(prompt_chars),
                "max": max(prompt_chars),
            },
            "seconds": {"total": round(sum(seconds), 3),
                        "median": statistics.median(seconds),
                        "max": max(seconds)},
        },
        "rows": rows,
    }


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="artifacts/retrieval.jsonl")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rescue", action="store_true")
    parser.add_argument("--rescue-table-k", type=int, default=20)
    parser.add_argument("--rescue-min-score", type=float, default=28.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    report = audit(
        Path(args.retrieval), Path(args.store_dir), Path(args.mask),
        args.rescue, args.rescue_table_k, args.rescue_min_score, args.limit,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()

