"""Retain provenance references removed by P2.4 constant folding."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24 import canonical_sha256  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl  # noqa: E402


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/devset_p24/p24_tune_authoring.folded.jsonl")
    parser.add_argument("--output", default="artifacts/devset_p24/p24_tune_authoring.final.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    records, audit = read_jsonl(args.input), []
    for record in records:
        tree = ast.parse(record["expression"], mode="eval")
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                and re.fullmatch(r"E[1-9][0-9]*", node.id)}
        expected = {f"E{index}" for index in range(1, len(record["cells"]) + 1)}
        missing = sorted(expected - used, key=lambda item: int(item[1:]))
        if missing:
            record["expression"] = (
                f"({record['expression']}) + 0 * sum({', '.join(missing)})"
            )
            record["notes"] = (str(record.get("notes", "")) +
                " Zero-weight provenance terms retain exact inputs consumed by folded medians.")
            audit.append({"id": int(record["id"]), "retained": missing})
    write_jsonl(output, records)
    print(json.dumps({"count": len(records), "output": str(output),
        "retained": audit, "records_sha256": canonical_sha256(records),
        "locked_opened": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
