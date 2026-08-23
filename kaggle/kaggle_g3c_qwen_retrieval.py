#!/usr/bin/env python3
"""Kaggle entrypoint for a manifest-validated G3C GPU payload."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=("qwen", "fake"), default="qwen")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    payload = Path(args.payload).resolve()
    code_root = payload / "code"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    from vifinqa.g3c.pipeline import run_gpu_pipeline

    result = run_gpu_pipeline(
        payload, Path(args.out_dir),
        backend=args.backend, limit=args.limit,
    )
    print(json.dumps({
        "run_signature": result["run_signature"],
        "mode": result["mode"],
        "backend": result["backend"],
        "stages_written": result["stages_written"],
        "question_count": result["question_count"],
        "scientific_evidence_valid": result["scientific_evidence_valid"],
        "runtime": result["runtime"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
