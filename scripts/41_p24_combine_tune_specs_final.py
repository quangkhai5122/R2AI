"""Combine the three reviewed parts with corrected complex P2.4 specs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24 import canonical_sha256  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl  # noqa: E402


PARTS = (
    "01_simple_lookup.jsonl", "03_verified_multifact.jsonl",
    "04_verified_direct.jsonl", "06_complex_corrected.jsonl",
)


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", default="artifacts/devset_p24/authoring_parts")
    parser.add_argument("--questions", default="artifacts/devset_p24/p24_tune_questions.jsonl")
    parser.add_argument("--output", default="artifacts/devset_p24/p24_tune_authoring.corrected.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    paths = [Path(args.parts_dir) / name for name in PARTS]
    if not all(path.is_file() for path in paths):
        raise SystemExit(f"missing required part: {[str(path) for path in paths if not path.is_file()]}")
    records = [record for path in paths for record in read_jsonl(path)]
    ids = [int(record["id"]) for record in records]
    wanted = {int(record["id"]) for record in read_jsonl(args.questions)}
    if len(ids) != len(set(ids)) or set(ids) != wanted:
        raise SystemExit("corrected authoring parts do not exactly cover tune IDs")
    records.sort(key=lambda record: int(record["id"]))
    write_jsonl(output, records)
    print(json.dumps({"count": len(records), "parts": [str(path) for path in paths],
        "output": str(output), "records_sha256": canonical_sha256(records),
        "locked_opened": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
