"""Preflight or build the CPU-only v5.2b multi-operand repair overlay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.semantic_repair_v52b import (
    build_multi_operand_repair_overlay,
    discover_multi_operand_repairs,
)
from vifinqa.utils.io import read_jsonl, setup_stdout


def _ids(value: str) -> set[int]:
    try:
        ids = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("IDs must be comma-separated integers") from exc
    if not ids:
        raise argparse.ArgumentTypeError("at least one expected selected ID is required")
    return ids


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True,
                        help="complete frozen v5.2a codegen JSONL")
    parser.add_argument("--retrieval", required=True,
                        help="frozen retrieval JSONL")
    parser.add_argument("--store-dir", required=True, help="frozen store directory")
    parser.add_argument("--preflight", action="store_true",
                        help="discover proposals without writing files")
    parser.add_argument("--out", default="", help="new complete overlay JSONL")
    parser.add_argument("--audit", default="",
                        help="audit JSON (default: OUT with .audit.json suffix)")
    parser.add_argument("--expect-selected-ids", type=_ids,
                        help="exact comma-separated allowlist from preflight")
    parser.add_argument("--expect-primary-signature", default="",
                        help="exact frozen primary run signature")
    parser.add_argument("--expect-primary-sha256", default="",
                        help="exact frozen primary file SHA-256")
    parser.add_argument("--expect-retrieval-sha256", default="",
                        help="exact frozen retrieval SHA-256")
    args = parser.parse_args()

    primary = Path(args.primary)
    retrieval = Path(args.retrieval)
    store_dir = Path(args.store_dir)
    if args.preflight:
        proposals, discovery = discover_multi_operand_repairs(
            read_jsonl(primary), read_jsonl(retrieval), store_dir,
        )
        print(json.dumps({
            "selected_ids": [item.qid for item in proposals],
            "selected": [
                {
                    "id": item.qid,
                    "operation": item.provenance["operation"],
                    "answer": item.answer,
                    "trigger_reasons": item.provenance["trigger_reasons"],
                    "operand_support_counts": [
                        operand["silver_support_count"]
                        for operand in item.provenance["operands"]
                    ],
                }
                for item in proposals
            ],
            "discovery": discovery,
        }, ensure_ascii=False, indent=2))
        return

    required = {
        "--out": args.out,
        "--expect-selected-ids": args.expect_selected_ids,
        "--expect-primary-signature": args.expect_primary_signature,
        "--expect-primary-sha256": args.expect_primary_sha256,
        "--expect-retrieval-sha256": args.expect_retrieval_sha256,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("build mode requires " + ", ".join(missing))

    out = Path(args.out)
    audit_path = Path(args.audit) if args.audit else out.with_suffix(".audit.json")
    audit = build_multi_operand_repair_overlay(
        primary,
        retrieval,
        store_dir,
        out,
        audit_path,
        expected_selected_ids=args.expect_selected_ids,
        expected_primary_signature=args.expect_primary_signature,
        expected_primary_sha256=args.expect_primary_sha256,
        expected_retrieval_sha256=args.expect_retrieval_sha256,
    )
    print(f"policy            : {audit['policy']}")
    print(f"selected          : {audit['counts']['selected']}")
    print(f"selected ids      : {audit['selected_ids']}")
    print(f"run signature     : {audit['run_signature']}")
    print(f"codegen SHA-256   : {audit['output']['sha256']}")
    print(f"codegen           : {audit['output']['path']}")
    print(f"audit             : {audit_path}")


if __name__ == "__main__":
    main()
