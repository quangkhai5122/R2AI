"""Validate phased Kaggle outputs and freeze the merged official R4 artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c_official.execution import (
    validate_embedding_result,
    validate_rerank_pair_result,
)
from vifinqa.g3c_official.finalize import finalize_official_result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    embedding = sub.add_parser("validate-embedding")
    embedding.add_argument("--payload", required=True)
    embedding.add_argument("--results", required=True)
    pair = sub.add_parser("validate-pair")
    pair.add_argument("--payload", required=True)
    pair.add_argument("--embedding-results", required=True)
    pair.add_argument("--results", required=True)
    pair.add_argument("--shards", nargs=2, type=int, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--payload", required=True)
    final.add_argument("--embedding-results", required=True)
    final.add_argument("--pair-a", required=True)
    final.add_argument("--pair-b", required=True)
    final.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "validate-embedding":
        result = validate_embedding_result(
            payload_dir=args.payload,
            result_dir=args.results,
            require_qwen=True,
        )
    elif args.command == "validate-pair":
        result = validate_rerank_pair_result(
            payload_dir=args.payload,
            embedding_result_dir=args.embedding_results,
            result_dir=args.results,
            expected_shards=tuple(args.shards),
            require_qwen=True,
        )
    else:
        result, audit, freeze = finalize_official_result(
            payload_dir=args.payload,
            embedding_result_dir=args.embedding_results,
            rerank_pair_dirs=[args.pair_a, args.pair_b],
            output_dir=args.out_dir,
        )
        result = {
            "status": result["status"],
            "result_fingerprint": result["result_fingerprint"],
            "audit_fingerprint": audit["audit_fingerprint"],
            "artifact_fingerprint": freeze["artifact_fingerprint"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
