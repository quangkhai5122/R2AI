"""Build an explicitly non-uploadable G3B offline submission."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.submission.build import build_submission
from vifinqa.utils.io import setup_stdout


def _assert_clean(path: Path) -> None:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("route") or {}).get("clean_profile") != "clean":
                raise SystemExit(
                    f"retrieval line {line_number} is not clean-profile"
                )
            count += 1
    if not count:
        raise SystemExit("retrieval is empty")


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--codegen", required=True)
    parser.add_argument(
        "--questions", default="data/g3b_v1/g3b_questions.jsonl"
    )
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sub-k", type=int, default=5)
    args = parser.parse_args()
    retrieval = Path(args.retrieval)
    _assert_clean(retrieval)
    path = build_submission(
        retrieval,
        Path(args.codegen),
        Path(args.store_dir),
        Path(args.out_dir),
        sub_k=args.sub_k,
        pos_mode="line",
        questions_path=Path(args.questions),
        expand_docs=False,
        offline_eval=True,
    )
    print(f"G3B offline submission -> {path}")
    print("DO NOT UPLOAD: synthetic question ids are outside the competition set")


if __name__ == "__main__":
    main()
