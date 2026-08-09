"""Merge Selection 14B (primary) with Selection 7B (safe fallback).

Only a structural placeholder in the primary run can be replaced.  Successful
14B answers and all rule answers remain untouched.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.hybrid import merge_codegen_hybrid
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True,
                        help="complete primary codegen JSONL (normally 14B)")
    parser.add_argument("--fallback", required=True,
                        help="complete fallback codegen JSONL (normally 7B)")
    parser.add_argument("--out", required=True, help="hybrid codegen JSONL")
    parser.add_argument("--audit", default="",
                        help="audit JSON (default: OUT with .audit.json suffix)")
    parser.add_argument("--expect-primary-signature", default="",
                        help="optional exact primary run_signature guard")
    parser.add_argument("--expect-fallback-signature", default="",
                        help="optional exact fallback run_signature guard")
    args = parser.parse_args()

    audit = merge_codegen_hybrid(
        Path(args.primary),
        Path(args.fallback),
        Path(args.out),
        Path(args.audit) if args.audit else None,
        expected_primary_signature=args.expect_primary_signature,
        expected_fallback_signature=args.expect_fallback_signature,
    )
    counts = audit["counts"]
    print(f"hybrid policy      : {audit['policy']}")
    print(f"kept primary      : {counts['kept_primary']}")
    print(f"used fallback     : {counts['used_fallback']}")
    print(f"unresolved        : {counts['unresolved']}")
    print(f"hybrid signature  : {audit['hybrid_run_signature']}")
    print(f"codegen            : {audit['output']['path']}")
    audit_path = Path(args.audit) if args.audit else Path(args.out).with_suffix(
        ".audit.json"
    )
    print(f"audit              : {audit_path}")


if __name__ == "__main__":
    main()
