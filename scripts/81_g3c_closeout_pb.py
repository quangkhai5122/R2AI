"""Build or validate the immutable G3C P-B/R4 closeout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c_official.closeout import (
    build_pb_closeout,
    validate_pb_closeout,
)

CLOSEOUT = ROOT / "experiments/g3c_qwen_retrieval_v1/pb_r4_closeout.json"
REGISTRY = ROOT / "experiments/g3c_qwen_retrieval_v1/registry.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    if args.command == "build":
        closeout, registry = build_pb_closeout(
            repo_root=ROOT,
            output_path=CLOSEOUT,
            registry_path=REGISTRY,
            config_path=ROOT / "configs/g3c_qwen_retrieval_v1.json",
            protocol_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/"
                "dev_protocol_freeze_v2.json"
            ),
            candidate_path=(
                ROOT / "artifacts/g3c_v1/dev_local_eval/"
                "g3c_candidate_freeze.json"
            ),
            promotion_payload_dir=ROOT / "artifacts/g3c_v1/promotion_payload",
            promotion_result_dir=(
                ROOT / "artifacts/g3c_v1/promotion_qwen_results"
            ),
            promotion_import_path=(
                ROOT / "artifacts/g3c_v1/promotion_qwen_import_validation.json"
            ),
            promotion_marker_path=(
                ROOT / "artifacts/g3c_v1/promotion_local_eval/"
                "PROMOTION_EVALUATION_OPENED.json"
            ),
            baseline_evaluation_path=(
                ROOT / "artifacts/g3b_v1/b0_promotion_evaluation.json"
            ),
            candidate_evaluation_path=(
                ROOT / "artifacts/g3c_v1/promotion_local_eval/r4/"
                "g3b_evaluation.json"
            ),
            paired_path=(
                ROOT / "artifacts/g3c_v1/promotion_local_eval/r4/"
                "paired_vs_r0.json"
            ),
        )
    else:
        closeout, registry = validate_pb_closeout(
            repo_root=ROOT,
            closeout_path=CLOSEOUT,
            registry_path=REGISTRY,
        )
    print(json.dumps({
        "status": closeout["status"],
        "candidate_fingerprint": closeout["candidate_fingerprint"],
        "closeout_fingerprint": closeout["closeout_fingerprint"],
        "registry_fingerprint": registry["registry_fingerprint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
