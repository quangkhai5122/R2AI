"""Kaggle entrypoint for exact two-T4 official G3C execution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vifinqa.g3c_official.execution import (
    run_embedding_orchestrator,
    run_embedding_worker,
    run_rerank_pair_orchestrator,
    run_rerank_worker,
    validate_embedding_result,
    validate_rerank_pair_result,
)
from vifinqa.g3c_official.payload import validate_official_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate_payload = sub.add_parser("validate-payload")
    validate_payload.add_argument("--payload", required=True)

    embedding = sub.add_parser("embedding")
    embedding.add_argument("--payload", required=True)
    embedding.add_argument("--out", required=True)
    embedding.add_argument("--backend", choices=("qwen", "fake"), default="qwen")

    worker_embed = sub.add_parser("worker-embed")
    worker_embed.add_argument("--payload", required=True)
    worker_embed.add_argument("--out", required=True)
    worker_embed.add_argument("--worker-index", type=int, required=True)
    worker_embed.add_argument("--backend", choices=("qwen", "fake"), default="qwen")

    rerank = sub.add_parser("rerank-pair")
    rerank.add_argument("--payload", required=True)
    rerank.add_argument("--embedding-results", required=True)
    rerank.add_argument("--out", required=True)
    rerank.add_argument("--shards", nargs=2, type=int, required=True)
    rerank.add_argument("--backend", choices=("qwen", "fake"), default="qwen")

    worker_rerank = sub.add_parser("worker-rerank")
    worker_rerank.add_argument("--payload", required=True)
    worker_rerank.add_argument("--embedding-results", required=True)
    worker_rerank.add_argument("--out", required=True)
    worker_rerank.add_argument("--shard-index", type=int, required=True)
    worker_rerank.add_argument("--backend", choices=("qwen", "fake"), default="qwen")

    validate_embedding = sub.add_parser("validate-embedding")
    validate_embedding.add_argument("--payload", required=True)
    validate_embedding.add_argument("--results", required=True)

    validate_pair = sub.add_parser("validate-rerank-pair")
    validate_pair.add_argument("--payload", required=True)
    validate_pair.add_argument("--embedding-results", required=True)
    validate_pair.add_argument("--results", required=True)
    validate_pair.add_argument("--shards", nargs=2, type=int, required=True)
    args = parser.parse_args()

    if args.command == "validate-payload":
        result = validate_official_payload(args.payload)
    elif args.command == "embedding":
        result = run_embedding_orchestrator(
            payload_dir=args.payload,
            output_dir=args.out,
            runner_path=Path(__file__).resolve(),
            backend=args.backend,
        )
    elif args.command == "worker-embed":
        result = run_embedding_worker(
            payload_dir=args.payload,
            output_dir=args.out,
            worker_index=args.worker_index,
            backend=args.backend,
        )
    elif args.command == "rerank-pair":
        result = run_rerank_pair_orchestrator(
            payload_dir=args.payload,
            embedding_result_dir=args.embedding_results,
            output_dir=args.out,
            runner_path=Path(__file__).resolve(),
            shard_indices=tuple(args.shards),
            backend=args.backend,
        )
    elif args.command == "worker-rerank":
        result = run_rerank_worker(
            payload_dir=args.payload,
            embedding_result_dir=args.embedding_results,
            output_dir=args.out,
            shard_index=args.shard_index,
            backend=args.backend,
        )
    elif args.command == "validate-embedding":
        result = validate_embedding_result(
            payload_dir=args.payload,
            result_dir=args.results,
            require_qwen=True,
        )
    else:
        result = validate_rerank_pair_result(
            payload_dir=args.payload,
            embedding_result_dir=args.embedding_results,
            result_dir=args.results,
            expected_shards=tuple(args.shards),
            require_qwen=True,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
