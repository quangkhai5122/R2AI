"""Build or validate the frozen official 1,012-question G3C payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c_official.payload import (
    build_official_payload,
    validate_official_payload,
)

DEFAULT_OUT = ROOT / "artifacts/g3c_v1/official_payload"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--payload", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--kaggle-dataset-id",
        default="lequangkhai5122005/vifinqa-g3c-r4-official-1012-v1",
    )
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_official_payload(args.payload)
    else:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        result = build_official_payload(
            repo_root=ROOT,
            output_dir=args.payload,
            questions_path=ROOT / "data/ViFinQA/questions/questions.jsonl",
            baseline_retrieval_path=ROOT / "artifacts/clean_v1/retrieval.jsonl",
            store_dir=ROOT / "artifacts/store",
            config_path=ROOT / "configs/g3c_qwen_retrieval_v1.json",
            execution_config_path=ROOT / "configs/g3c_official_1012_v1.json",
            source_protocol_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/"
                "dev_protocol_freeze_v2.json"
            ),
            official_protocol_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/"
                "official_protocol_freeze.json"
            ),
            candidate_path=(
                ROOT / "artifacts/g3c_v1/dev_local_eval/"
                "g3c_candidate_freeze.json"
            ),
            closeout_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/"
                "pb_r4_closeout.json"
            ),
            canary_manifest_path=(
                ROOT / "artifacts/g3c_v1/official_preflight/"
                "exact_numeric_canary.json"
            ),
            canary_vectors_path=(
                ROOT / "artifacts/g3c_v1/official_preflight/"
                "exact_numeric_canary_vectors.npz"
            ),
            promotion_result_dir=(
                ROOT / "artifacts/g3c_v1/promotion_qwen_results"
            ),
            kaggle_dataset_id=args.kaggle_dataset_id,
            source_git_head=head,
            source_git_dirty=bool(status.strip()),
            source_git_status_sha256=hashlib.sha256(
                status.encode("utf-8")
            ).hexdigest(),
        )
    print(json.dumps({
        "mode": result["mode"],
        "question_count": result["question_count"],
        "selected_stage": result["selected_stage"],
        "payload_fingerprint": result["payload_fingerprint"],
        "workload_fingerprint": result["workload_fingerprint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
