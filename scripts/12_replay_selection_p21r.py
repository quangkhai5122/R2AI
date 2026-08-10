"""Replay saved P2.1 Selection attempts with deterministic P2.1r synthesis.

The input codegen JSONL is read-only.  By default, only structural ``none``
records may be replaced; successful LLM and rule answers are preserved.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.selection_replay import replay_selection_artifact
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True,
                        help="retrieval JSONL used by the original Selection run")
    parser.add_argument("--codegen", required=True,
                        help="complete P2.1 codegen JSONL containing selection_trace")
    parser.add_argument("--store", default="artifacts/store",
                        help="normalized report store (default: artifacts/store)")
    parser.add_argument("--out", required=True,
                        help="new replayed codegen JSONL; must differ from --codegen")
    parser.add_argument("--audit", default="",
                        help="audit JSON (default: OUT with .audit.json suffix)")
    parser.add_argument("--k", type=int, default=0,
                        help="original codegen table cap; 0 = route dynamic budget")
    parser.add_argument("--top-n", type=int, default=12,
                        help="original Selection shortlist size (default: 12)")
    parser.add_argument(
        "--replace-policy", choices=("none_only", "trace_failures"),
        default="none_only",
        help=("none_only preserves every successful final answer (default); "
              "trace_failures is an explicit overwrite ablation"),
    )
    parser.add_argument(
        "--output-types", default="all",
        help=("comma-separated route output types to replay; default 'all'. "
              "Example: --output-types year"),
    )
    args = parser.parse_args()

    out = Path(args.out)
    audit_path = Path(args.audit) if args.audit else out.with_suffix(".audit.json")
    audit = replay_selection_artifact(
        Path(args.retrieval), Path(args.codegen), Path(args.store), out,
        audit_path,
        k=args.k, top_n=args.top_n, replace_policy=args.replace_policy,
        output_types=args.output_types,
    )
    counts = audit["counts"]
    print(f"policy              : {audit['policy']}")
    print(f"replace policy      : {audit['replace_policy']}")
    print(f"kept                : {counts['kept_non_eligible']}")
    print(f"skipped output type : {counts['skipped_by_output_type']}")
    print(f"replayed            : {counts['replayed']}")
    print(f"unresolved          : {counts['unresolved']}")
    print(f"replay signature    : {audit['output']['run_signature']}")
    print(f"codegen             : {out}")
    print(f"audit               : {audit_path}")


if __name__ == "__main__":
    main()
