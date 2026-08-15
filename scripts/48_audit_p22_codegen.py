"""Fail-closed audit for a masked Structured Selection v2 Kaggle output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.semantic import all_dataframe_refs
from vifinqa.utils.io import read_jsonl, setup_stdout, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question(value) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def _unique(rows: list[dict], label: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        if "id" not in row:
            raise ValueError(f"{label} row has no id")
        qid = int(row["id"])
        if qid in out:
            raise ValueError(f"{label} has duplicate id={qid}")
        out[qid] = row
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def _mask(path: Path) -> set[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("ids") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("mask must be a non-empty JSON list or an object with ids")
    ids = [int(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("mask contains duplicate IDs")
    if isinstance(raw, dict) and raw.get("count") is not None \
            and int(raw["count"]) != len(ids):
        raise ValueError("mask count does not match ids")
    return set(ids)


def audit(codegen_path: Path, retrieval_path: Path, mask_path: Path,
          *, allow_incomplete: bool = False,
          allowed_attempt_mask_paths: tuple[Path, ...] = ()) -> dict:
    codegen = _unique(read_jsonl(codegen_path), "codegen")
    retrieval = _unique(read_jsonl(retrieval_path), "retrieval")
    target = _mask(mask_path)
    allowed_paths = tuple(Path(path) for path in allowed_attempt_mask_paths)
    allowed_masks = [(path, _mask(path)) for path in allowed_paths]
    allowed_attempted: set[int] = set().union(
        *(ids for _path, ids in allowed_masks),
    ) if allowed_masks else set()
    overlap = target & allowed_attempted
    if overlap:
        raise ValueError(
            f"target/upstream masks overlap: {sorted(overlap)[:10]}")
    if set(codegen) != set(retrieval):
        raise ValueError("codegen and retrieval ID universes differ")
    if not target <= set(retrieval):
        raise ValueError(f"mask has unknown IDs: {sorted(target - set(retrieval))[:10]}")

    if not allowed_attempted <= set(retrieval):
        unknown = allowed_attempted - set(retrieval)
        raise ValueError(f"upstream mask has unknown IDs: {sorted(unknown)[:10]}")
    signatures = {str(row.get("run_signature") or "") for row in codegen.values()}
    if "" in signatures or len(signatures) != 1:
        raise ValueError(f"expected exactly one non-empty run_signature, got {signatures}")

    outcomes, rejection_counts, shortlist_modes = Counter(), Counter(), Counter()
    attempted: set[int] = set()
    accepted: set[int] = set()
    for qid, row in codegen.items():
        if _question(row.get("question")) != _question(retrieval[qid].get("question")):
            raise ValueError(f"question mismatch at id={qid}")
        try:
            answer = float(row.get("answer"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid answer at id={qid}") from exc
        if not math.isfinite(answer):
            raise ValueError(f"non-finite answer at id={qid}")

        trace = row.get("selection_trace")
        marker = row.get("llm_attempt_status") == "completed"
        is_v2 = isinstance(trace, dict) and trace.get("mode") == "select_v2" \
            and int(trace.get("schema_version", 0)) == 2
        if marker or is_v2:
            if not (marker and is_v2):
                raise ValueError(f"incomplete v2 trace/completion contract at id={qid}")
            attempted.add(qid)
            outcome = str(trace.get("outcome") or "missing")
            if qid in target:
                outcomes[outcome] += 1
                rejection_counts.update(trace.get("rejection_counts") or {})
                shortlist_modes[str((trace.get("shortlist") or {}).get("rescue_mode")
                                     or "none")] += 1
            if outcome == "accepted":
                accepted.add(qid)
                if row.get("source") != "llm_select_v2" or row.get("status") != "ok":
                    raise ValueError(f"accepted v2 was not selected at id={qid}")
                query = str(row.get("pandas_query") or "")
                refs = all_dataframe_refs(query)
                evidence = {str(item.get("var") or "") for item in row.get("used_vars") or []
                            if isinstance(item, dict)}
                if not refs or not refs <= evidence:
                    raise ValueError(f"accepted v2 has incomplete evidence at id={qid}")

    unexpected = attempted - target - allowed_attempted
    if unexpected:
        raise ValueError(f"LLM attempted IDs outside mask: {sorted(unexpected)[:10]}")
    target_attempted = attempted & target
    target_accepted = accepted & target
    inherited_attempted = attempted & allowed_attempted
    pending = target - target_attempted
    if pending and not allow_incomplete:
        raise ValueError(
            f"masked run is incomplete: {len(pending)} pending IDs; "
            "resume with the exact same command/signature or pass --allow-incomplete for diagnostics"
        )

    return {
        "schema_version": "p22_codegen_audit_v2_upstream_mask",
        "inputs": {
            "codegen": {"path": str(codegen_path), "sha256": _sha256(codegen_path)},
            "retrieval": {"path": str(retrieval_path), "sha256": _sha256(retrieval_path)},
            "mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
            "allowed_attempt_masks": [
                {"path": str(path), "sha256": _sha256(path)}
                for path, _ids in allowed_masks
            ],
        },
        "run_signature": next(iter(signatures)),
        "counts": {
            "rows": len(codegen), "target": len(target),
            "attempted": len(target_attempted), "pending": len(pending),
            "accepted": len(target_accepted),
            "rejected": len(target_attempted - target_accepted),
            "inherited_attempted": len(inherited_attempted),
        },
        "outcomes": dict(sorted(outcomes.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "shortlist_rescue_modes": dict(sorted(shortlist_modes.items())),
        "inherited_attempted_ids": sorted(inherited_attempted),
        "accepted_ids": sorted(target_accepted),
        "pending_ids": sorted(pending),
    }


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codegen", required=True)
    parser.add_argument("--retrieval", default="artifacts/retrieval.jsonl")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--allow-attempted-from-mask", action="append", default=[],
        help="repeatable upstream mask whose inherited completed v2 traces are allowed",
    )
    args = parser.parse_args()
    report = audit(Path(args.codegen), Path(args.retrieval), Path(args.mask),
                   allow_incomplete=args.allow_incomplete,
                   allowed_attempt_mask_paths=tuple(
                       Path(path) for path in args.allow_attempted_from_mask))
    if args.out:
        write_json(Path(args.out), report)
    print(json.dumps({"counts": report["counts"], "outcomes": report["outcomes"],
                      "rejections": report["rejection_counts"]},
                     ensure_ascii=False, indent=2))
    if args.out:
        print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
