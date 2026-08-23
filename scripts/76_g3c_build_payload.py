"""Build or validate a label-free G3C Kaggle GPU payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c.payload import build_gpu_payload, validate_gpu_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--mode", choices=("dev", "promotion"), required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument(
        "--config", default="configs/g3c_qwen_retrieval_v1.json"
    )
    build.add_argument("--questions")
    build.add_argument("--baseline-retrieval")
    build.add_argument("--store-dir", default="artifacts/store")
    build.add_argument("--candidate-freeze")
    build.add_argument("--kaggle-dataset-id")
    build.add_argument(
        "--protocol-freeze",
        default=(
            "experiments/g3c_qwen_retrieval_v1/"
            "dev_protocol_freeze_v2.json"
        ),
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument("--payload", required=True)
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_gpu_payload(args.payload)
    else:
        questions = args.questions or (
            "data/g3b_v1/g3b_dev_questions.jsonl"
            if args.mode == "dev"
            else "data/g3b_v1/g3b_promotion_questions.jsonl"
        )
        baseline = args.baseline_retrieval or (
            "artifacts/g3b_v1/b0_dev_retrieval.jsonl"
            if args.mode == "dev"
            else "artifacts/g3b_v1/b0_promotion_retrieval.jsonl"
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        kaggle_dataset_id = args.kaggle_dataset_id or (
            "lequangkhai5122005/vifinqa-g3c-qwen-retrieval-dev-v1"
            if args.mode == "dev"
            else (
                "lequangkhai5122005/"
                "vifinqa-g3c-qwen-retrieval-promotion-v1"
            )
        )
        result = build_gpu_payload(
            repo_root=ROOT,
            output_dir=args.out_dir,
            mode=args.mode,
            config_path=args.config,
            questions_path=questions,
            baseline_retrieval_path=baseline,
            store_dir=args.store_dir,
            source_git_head=head,
            source_git_dirty=bool(status.strip()),
            source_git_status_sha256=hashlib.sha256(status.encode("utf-8")).hexdigest(),
            kaggle_dataset_id=kaggle_dataset_id,
            protocol_freeze_path=args.protocol_freeze,
            candidate_freeze_path=args.candidate_freeze,
        )
    print(json.dumps({
        "mode": result["mode"],
        "question_count": result["question_count"],
        "selected_stage": result.get("selected_stage"),
        "payload_fingerprint": result["payload_fingerprint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
