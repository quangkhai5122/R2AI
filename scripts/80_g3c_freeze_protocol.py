"""Build or validate the pre-Qwen G3C dev protocol freeze."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c.protocol import (
    build_protocol_freeze,
    validate_protocol_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument(
        "--config", default="configs/g3c_qwen_retrieval_v1.json"
    )
    parser.add_argument(
        "--freeze",
        default=(
            "experiments/g3c_qwen_retrieval_v1/"
            "dev_protocol_freeze_v2.json"
        ),
    )
    args = parser.parse_args()
    if args.command == "build":
        result = build_protocol_freeze(
            repo_root=ROOT,
            config_path=args.config,
            output_path=args.freeze,
        )
    else:
        result = validate_protocol_freeze(
            repo_root=ROOT,
            config_path=args.config,
            freeze_path=args.freeze,
            verify_worktree=True,
        )
    print(json.dumps({
        "status": result["status"],
        "config_sha256": result["config_sha256"],
        "behavior_tree_sha256": result["behavior_tree_sha256"],
        "protocol_fingerprint": result["protocol_fingerprint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
