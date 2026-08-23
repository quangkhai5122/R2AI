"""Apply the pre-registered dev gate and freeze one G3C candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3c.common import read_json
from vifinqa.g3c.freeze import (
    build_dev_selection,
    freeze_selected_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/g3c_qwen_retrieval_v1.json"
    )
    parser.add_argument("--gpu-result-manifest", required=True)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--selection-out", required=True)
    parser.add_argument("--freeze-out", required=True)
    args = parser.parse_args()
    index = read_json(args.evaluation_index)
    if index.get("mode") != "dev":
        raise SystemExit("selection index is not dev policy")
    evaluations = {
        stage: value["evaluation"]
        for stage, value in index["stages"].items()
    }
    selection = build_dev_selection(
        config_path=args.config,
        gpu_result_manifest_path=args.gpu_result_manifest,
        evaluations=evaluations,
        output_path=args.selection_out,
    )
    summary = {
        "gate_passed": selection["gate_passed"],
        "selected_stage": selection["selected_stage"],
        "selection_out": args.selection_out,
    }
    if selection["gate_passed"]:
        freeze = freeze_selected_candidate(
            config_path=args.config,
            gpu_result_manifest_path=args.gpu_result_manifest,
            selection_path=args.selection_out,
            output_path=args.freeze_out,
        )
        summary["freeze_out"] = args.freeze_out
        summary["candidate_fingerprint"] = (
            freeze["candidate_fingerprint"]
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not selection["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
