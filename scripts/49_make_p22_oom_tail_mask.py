"""Derive a frozen P2.2 tail mask from a flushed Kaggle checkpoint.

This is a recovery tool, not a resume shortcut.  It verifies the complete
1,012-row checkpoint, identifies target IDs carrying a completed Selection-v2
attempt, and writes ``parent_mask - completed`` as a new fingerprintable mask.
The recovered tail must be run into a new output file and merged afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.utils.io import read_jsonl, setup_stdout


SCHEMA_VERSION = "p22_oom_tail_mask_v1"


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


def _read_mask(path: Path) -> tuple[dict, set[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("ids") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("parent mask must be a non-empty JSON list/object.ids")
    ids = [int(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("parent mask contains duplicate IDs")
    if isinstance(raw, dict) and raw.get("count") is not None \
            and int(raw["count"]) != len(ids):
        raise ValueError("parent mask count does not match ids")
    return raw if isinstance(raw, dict) else {}, set(ids)


def _canonical_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def _write_idempotent(path: Path, obj: dict) -> None:
    payload = _canonical_bytes(obj)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                f"refusing to overwrite a different frozen tail mask: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_tail_mask(checkpoint_path: Path, parent_mask_path: Path,
                    retrieval_path: Path, *, expect_pending: int = -1) -> dict:
    checkpoint_path = Path(checkpoint_path)
    parent_mask_path = Path(parent_mask_path)
    retrieval_path = Path(retrieval_path)
    checkpoint = _unique(read_jsonl(checkpoint_path), "checkpoint")
    retrieval = _unique(read_jsonl(retrieval_path), "retrieval")
    parent_raw, parent_ids = _read_mask(parent_mask_path)

    if set(checkpoint) != set(retrieval):
        raise ValueError("checkpoint and retrieval ID universes differ")
    if not parent_ids <= set(retrieval):
        unknown = sorted(parent_ids - set(retrieval))[:10]
        raise ValueError(f"parent mask has unknown IDs: {unknown}")
    for qid in checkpoint:
        if _question(checkpoint[qid].get("question")) != _question(
                retrieval[qid].get("question")):
            raise ValueError(f"question mismatch at id={qid}")

    signatures = {
        str(row.get("run_signature") or "").strip()
        for row in checkpoint.values()
    }
    if "" in signatures or len(signatures) != 1:
        raise ValueError(
            "checkpoint must contain exactly one non-empty run_signature"
        )
    run_signature = next(iter(signatures))

    retrieval_sha = _sha256(retrieval_path)
    parent_retrieval_sha = str(
        ((parent_raw.get("inputs") or {}).get("retrieval") or {}).get(
            "sha256", ""
        )
    )
    if parent_retrieval_sha and parent_retrieval_sha != retrieval_sha:
        raise ValueError("parent mask retrieval hash differs from retrieval")

    completed: set[int] = set()
    outcomes = Counter()
    for qid, row in checkpoint.items():
        trace = row.get("selection_trace")
        marker = row.get("llm_attempt_status") == "completed"
        is_v2 = (
            isinstance(trace, dict)
            and trace.get("mode") == "select_v2"
            and int(trace.get("schema_version", 0)) == 2
        )
        if marker or is_v2:
            if not (marker and is_v2):
                raise ValueError(
                    f"incomplete Selection-v2 completion contract at id={qid}"
                )
            completed.add(qid)
            outcomes[str(trace.get("outcome") or "missing")] += 1

    unexpected = completed - parent_ids
    if unexpected:
        raise ValueError(
            f"completed Selection-v2 IDs outside parent mask: "
            f"{sorted(unexpected)[:10]}"
        )
    pending = sorted(parent_ids - completed)
    if not pending:
        raise ValueError("parent mask is already complete; no recovery tail exists")
    if expect_pending >= 0 and len(pending) != expect_pending:
        raise ValueError(
            f"pending-count mismatch: got {len(pending)}, "
            f"expected {expect_pending}"
        )

    by_output = Counter(
        str((retrieval[qid].get("route") or {}).get("output_type") or "number")
        for qid in pending
    )
    by_operation = Counter(
        str((((retrieval[qid].get("route") or {}).get("plan") or {}).get("op"))
            or "lookup")
        for qid in pending
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "P2.2-B-tail-OOM",
        "policy": (
            "parent target mask minus completed select_v2 attempts in the "
            "flushed checkpoint"
        ),
        "count": len(pending),
        "ids": pending,
        "strata": {
            "output_type": dict(sorted(by_output.items())),
            "operation": dict(sorted(by_operation.items())),
        },
        "checkpoint": {
            "completed_count": len(completed),
            "completed_ids": sorted(completed),
            "outcomes": dict(sorted(outcomes.items())),
            "run_signature": run_signature,
        },
        "parent": {
            "name": str(parent_raw.get("name") or parent_mask_path.stem),
            "count": len(parent_ids),
        },
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
            "parent_mask": {
                "path": str(parent_mask_path),
                "sha256": _sha256(parent_mask_path),
            },
            "retrieval": {
                "path": str(retrieval_path),
                "sha256": retrieval_sha,
            },
        },
        "recovery_contract": {
            "new_output_required": True,
            "merge_after_tail": True,
            "resume_original_checkpoint": False,
        },
    }


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--parent-mask",
        default="artifacts/p22_targets/p22b_rejected_non_year.json",
    )
    parser.add_argument("--retrieval", default="artifacts/retrieval.jsonl")
    parser.add_argument(
        "--out", default="artifacts/p22_targets/p22b_oom_tail.json",
    )
    parser.add_argument("--expect-pending", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mask = build_tail_mask(
        Path(args.checkpoint), Path(args.parent_mask), Path(args.retrieval),
        expect_pending=args.expect_pending,
    )
    if not args.dry_run:
        _write_idempotent(Path(args.out), mask)
    print(json.dumps({
        "parent": mask["parent"]["count"],
        "completed": mask["checkpoint"]["completed_count"],
        "pending": mask["count"],
        "pending_ids": mask["ids"],
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("dry-run: no mask written")
    else:
        print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
