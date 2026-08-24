"""Build or validate the exact-execution official G3C protocol freeze."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c_official.protocol import (
    build_official_protocol_freeze,
    validate_official_protocol,
)

FREEZE = ROOT / "experiments/g3c_qwen_retrieval_v1/official_protocol_freeze.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    if args.command == "build":
        if FREEZE.exists():
            raise SystemExit("refusing to overwrite official protocol freeze")
        result = build_official_protocol_freeze(
            repo_root=ROOT,
            output_path=FREEZE,
            source_config_path=ROOT / "configs/g3c_qwen_retrieval_v1.json",
            execution_config_path=ROOT / "configs/g3c_official_1012_v1.json",
            source_protocol_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/"
                "dev_protocol_freeze_v2.json"
            ),
            candidate_path=(
                ROOT / "artifacts/g3c_v1/dev_local_eval/"
                "g3c_candidate_freeze.json"
            ),
            closeout_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/"
                "pb_r4_closeout.json"
            ),
            registry_path=(
                ROOT / "experiments/g3c_qwen_retrieval_v1/registry.json"
            ),
            canary_manifest_path=(
                ROOT / "artifacts/g3c_v1/official_preflight/"
                "exact_numeric_canary.json"
            ),
            canary_vectors_path=(
                ROOT / "artifacts/g3c_v1/official_preflight/"
                "exact_numeric_canary_vectors.npz"
            ),
        )
    else:
        result = validate_official_protocol(
            repo_root=ROOT,
            freeze_path=FREEZE,
            verify_worktree=True,
        )
    print(json.dumps({
        "status": result["status"],
        "official_protocol_fingerprint": result[
            "official_protocol_fingerprint"
        ],
        "behavior_tree_sha256": result["behavior_tree_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
