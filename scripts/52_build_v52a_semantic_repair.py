"""Build the CPU-only v5.2a column/period/unit repair overlay."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.semantic_repair import build_semantic_repair_overlay
from vifinqa.utils.io import setup_stdout


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
                        help="complete frozen primary codegen JSONL")
    parser.add_argument("--retrieval", required=True,
                        help="retrieval JSONL with routes/questions")
    parser.add_argument("--store-dir", required=True, help="frozen store directory")
    parser.add_argument("--out", required=True, help="new complete overlay JSONL")
    parser.add_argument("--audit", default="",
                        help="audit JSON (default: OUT with .audit.json suffix)")
    parser.add_argument("--expect-selected-ids", required=True, type=_ids,
                        help="exact comma-separated allowlist discovered in preflight")
    parser.add_argument("--expect-primary-signature", required=True,
                        help="exact frozen primary run signature")
    parser.add_argument("--expect-primary-sha256", required=True,
                        help="exact frozen primary file SHA-256")
    parser.add_argument("--expect-retrieval-sha256", required=True,
                        help="exact frozen retrieval SHA-256")
    args = parser.parse_args()

    out = Path(args.out)
    audit_path = Path(args.audit) if args.audit else out.with_suffix(".audit.json")
    audit = build_semantic_repair_overlay(
        Path(args.primary),
        Path(args.retrieval),
        Path(args.store_dir),
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
