"""Freeze causal target masks for Structured Selection v2 experiments.

B  = #19 structural-none, v1 rejected, non-year, non-empty shortlist.
C  = #19 no-candidate rows whose frozen rescue shortlist covers every routed F-slot.
BC = their disjoint union.

The script never reads P2.4 locked labels and never changes codegen artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.utils.io import read_jsonl, setup_stdout


SCHEMA_VERSION = "p22_target_mask_v1"
FROZEN_RETRIEVAL_SHA256 = "96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique(rows: list[dict], label: str) -> dict[int, dict]:
    out = {}
    for row in rows:
        try:
            qid = int(row["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{label}: invalid id in row") from exc
        if qid in out:
            raise SystemExit(f"{label}: duplicate id {qid}")
        out[qid] = row
    return out


def _is_structural_none(row: dict) -> bool:
    try:
        answer = float(row.get("answer"))
    except (TypeError, ValueError):
        return False
    return (
        row.get("source") == "none"
        and row.get("status") == "failed"
        and math.isfinite(answer) and answer == 0.0
        and str(row.get("pandas_query", "")).strip() == "0.0"
    )


def _mask(name: str, policy: str, ids: list[int], retrieval: dict[int, dict],
          inputs: dict) -> dict:
    ids = sorted(ids)
    by_output = Counter(
        str(retrieval[qid].get("route", {}).get("output_type", "number"))
        for qid in ids
    )
    by_operation = Counter(
        str((retrieval[qid].get("route", {}).get("plan") or {}).get("op", "lookup"))
        for qid in ids
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "policy": policy,
        "count": len(ids),
        "ids": ids,
        "strata": {
            "output_type": dict(sorted(by_output.items())),
            "operation": dict(sorted(by_operation.items())),
        },
        "inputs": inputs,
    }


def _canonical_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_idempotent(path: Path, obj: dict) -> None:
    payload = _canonical_bytes(obj)
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(
                f"refusing to overwrite a different frozen mask: {path}; "
                "use a new output directory after auditing the changed inputs"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_masks(codegen_path: Path, retrieval_path: Path, rescue_path: Path,
                expect_b: int = 55, expect_c: int = 48) -> dict[str, dict]:
    retrieval_sha = _sha256(retrieval_path)
    if retrieval_sha != FROZEN_RETRIEVAL_SHA256:
        raise SystemExit(
            "retrieval control drift: expected " + FROZEN_RETRIEVAL_SHA256
            + f", got {retrieval_sha}"
        )
    codegen = _unique(read_jsonl(codegen_path), "codegen")
    retrieval = _unique(read_jsonl(retrieval_path), "retrieval")
    if set(codegen) != set(retrieval):
        raise SystemExit("codegen/retrieval id universes differ")
    for qid in codegen:
        if str(codegen[qid].get("question", "")) != str(retrieval[qid].get("question", "")):
            raise SystemExit(f"question mismatch for id {qid}")

    rescue = json.loads(rescue_path.read_text(encoding="utf-8"))
    if rescue.get("counts", {}).get("target_no_candidates") != 142:
        raise SystemExit("rescue audit is not the frozen 142-row no-candidate audit")
    rescue_input_sha = str(
        rescue.get("inputs", {}).get("retrieval", {}).get("sha256", "")
    )
    if rescue_input_sha != retrieval_sha:
        raise SystemExit("rescue audit retrieval hash differs from frozen retrieval")
    rescue_rows = _unique(list(rescue.get("rows") or []), "rescue audit")

    no_candidate_ids = set()
    b_ids = []
    for qid, row in codegen.items():
        if not _is_structural_none(row):
            continue
        trace = row.get("selection_trace") or {}
        outcome = str(trace.get("outcome", ""))
        if outcome == "no_candidates":
            no_candidate_ids.add(qid)
        route = retrieval[qid].get("route") or {}
        if (outcome == "rejected"
                and int(trace.get("candidate_count") or 0) > 0
                and route.get("output_type") != "year"):
            b_ids.append(qid)

    c_ids = []
    partial_ids = []
    still_empty_ids = []
    for qid in sorted(no_candidate_ids):
        row = rescue_rows.get(qid)
        if row is None:
            raise SystemExit(f"rescue audit misses no-candidate id {qid}")
        candidates = list(row.get("candidates") or [])
        if not candidates:
            still_empty_ids.append(qid)
            continue
        facts = ((retrieval[qid].get("route") or {}).get("plan") or {}).get("facts") or []
        required = {f"F{i}" for i in range(1, len(facts) + 1)}
        present = {str(c.get("fact_slot") or "") for c in candidates}
        if required and required <= present:
            c_ids.append(qid)
        else:
            partial_ids.append(qid)

    if len(b_ids) != expect_b or len(c_ids) != expect_c:
        raise SystemExit(
            f"target-count drift: B={len(b_ids)} (expected {expect_b}), "
            f"C={len(c_ids)} (expected {expect_c})"
        )
    overlap = set(b_ids) & set(c_ids)
    if overlap:
        raise SystemExit(f"B/C masks unexpectedly overlap: {sorted(overlap)}")

    inputs = {
        "retrieval": {"path": str(retrieval_path), "sha256": retrieval_sha},
        "codegen_control": {"path": str(codegen_path), "sha256": _sha256(codegen_path)},
        "rescue_audit": {"path": str(rescue_path), "sha256": _sha256(rescue_path)},
    }
    masks = {
        "p22b_rejected_non_year.json": _mask(
            "P2.2-B",
            "structural_none AND selection_v1.outcome=rejected AND candidate_count>0 "
            "AND output_type!=year",
            b_ids, retrieval, inputs,
        ),
        "p22c_rescue_fact_complete.json": _mask(
            "P2.2-C",
            "selection_v1.outcome=no_candidates AND rescue_candidates>0 AND all routed F-slots present",
            c_ids, retrieval, inputs,
        ),
        "p22bc_combined.json": _mask(
            "P2.2-BC", "disjoint union of P2.2-B and P2.2-C",
            sorted(set(b_ids) | set(c_ids)), retrieval, inputs,
        ),
    }
    masks["p22_targets.audit.json"] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": inputs,
        "counts": {
            "B": len(b_ids), "C": len(c_ids), "BC": len(set(b_ids) | set(c_ids)),
            "no_candidates": len(no_candidate_ids),
            "rescue_partial_slots_excluded": len(partial_ids),
            "rescue_still_empty_excluded": len(still_empty_ids),
        },
        "excluded_ids": {
            "rescue_partial_slots": partial_ids,
            "rescue_still_empty": still_empty_ids,
        },
    }
    return masks


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--codegen", default="artifacts/codegen_p21r_all_v3.jsonl")
    parser.add_argument("--retrieval", default="artifacts/retrieval.jsonl")
    parser.add_argument("--rescue-audit", default="artifacts/shortlist_rescue_audit.json")
    parser.add_argument("--out-dir", default="artifacts/p22_targets")
    parser.add_argument("--expect-b", type=int, default=55)
    parser.add_argument("--expect-c", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    masks = build_masks(
        Path(args.codegen), Path(args.retrieval), Path(args.rescue_audit),
        args.expect_b, args.expect_c,
    )
    if not args.dry_run:
        out_dir = Path(args.out_dir)
        for name, obj in masks.items():
            _write_idempotent(out_dir / name, obj)
    counts = masks["p22_targets.audit.json"]["counts"]
    print(f"P2.2 target masks verified: B={counts['B']} C={counts['C']} BC={counts['BC']}")
    if args.dry_run:
        print("dry-run: no files written")
    else:
        print(f"OK -> {args.out_dir}")


if __name__ == "__main__":
    main()

