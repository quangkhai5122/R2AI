"""Build the label-free exact-numeric canary from frozen Promotion evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c_official.canary import (
    build_numeric_canary,
    load_numeric_canary,
)

DEFAULT_DIR = ROOT / "artifacts/g3c_v1/official_preflight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--out-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()
    output = Path(args.out_dir)
    manifest = output / "exact_numeric_canary.json"
    vectors = output / "exact_numeric_canary_vectors.npz"
    if args.command == "build":
        if manifest.exists() or vectors.exists():
            raise SystemExit(
                "refusing to overwrite an existing exact numeric canary"
            )
        output.mkdir(parents=True, exist_ok=True)
        result = build_numeric_canary(
            promotion_payload_dir=ROOT / "artifacts/g3c_v1/promotion_payload",
            promotion_result_dir=(
                ROOT / "artifacts/g3c_v1/promotion_qwen_results"
            ),
            manifest_output_path=manifest,
            vectors_output_path=vectors,
        )
    else:
        result = load_numeric_canary(manifest, vectors)
    print(json.dumps({
        "canary_fingerprint": result["canary_fingerprint"],
        "embedding_cases": len(result["embedding_cases"]),
        "reranker_cases": len(result["reranker_cases"]),
        "exact_cached_replay_passed": result["exact_cached_replay_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
