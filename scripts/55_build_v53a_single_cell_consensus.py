"""Preflight or build the CPU-only v5.3a single-cell repair overlay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.single_cell_consensus import (  # noqa: E402
    build_single_cell_consensus_overlay,
    discover_single_cell_consensus,
)
from vifinqa.utils.io import read_jsonl, setup_stdout  # noqa: E402


def _ids(text: str) -> set[int]:
    return {int(value.strip()) for value in text.split(",") if value.strip()}


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--audit")
    parser.add_argument("--expect-selected-ids", default="")
    parser.add_argument("--expect-target-ids", default="")
    parser.add_argument("--expect-primary-signature", default="")
    parser.add_argument("--expect-primary-sha256", default="")
    parser.add_argument("--expect-retrieval-sha256", default="")
    args = parser.parse_args()
    if args.preflight:
        proposals, audit = discover_single_cell_consensus(
            read_jsonl(args.primary), read_jsonl(args.retrieval),
            Path(args.store_dir), mode="repair",
        )
        print(json.dumps({
            "selected_ids": [proposal.qid for proposal in proposals],
            "proposals": [{"id": p.qid, "answer": p.answer,
                           "provenance": p.provenance} for p in proposals],
            "discovery": audit,
        }, ensure_ascii=False, indent=2))
        return
    if not args.out:
        parser.error("--out is required unless --preflight is used")
    result = build_single_cell_consensus_overlay(
        args.primary, args.retrieval, args.store_dir, args.out,
        mode="repair", audit_path=args.audit or None,
        expected_selected_ids=_ids(args.expect_selected_ids),
        expected_target_ids=(_ids(args.expect_target_ids)
                             if args.expect_target_ids else None),
        expected_primary_signature=args.expect_primary_signature,
        expected_primary_sha256=args.expect_primary_sha256,
        expected_retrieval_sha256=args.expect_retrieval_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
