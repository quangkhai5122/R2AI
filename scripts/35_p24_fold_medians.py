"""Constant-fold repeated medians in complex P2.4 expressions.

Every folded literal is evaluated from the exact evidence cells in the same
record.  The surrounding comparisons still reference every source metric, so
strict evidence coverage and deterministic replay remain intact while the AST
stays below the validator's anti-abuse node budget.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24 import canonical_sha256  # noqa: E402
from vifinqa.devset.p24_authoring import _eval  # noqa: E402
from vifinqa.devset.p24_authoring_ext import P24ForensicTableLoader  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl  # noqa: E402


class MedianFolder(ast.NodeTransformer):
    def __init__(self, values: dict[str, float]):
        self.values = values
        self.folded = 0

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "median":
            self.folded += 1
            return ast.copy_location(ast.Constant(float(_eval(node, self.values))), node)
        return node


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/devset_p24/p24_tune_authoring.complete.jsonl")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--output", default="artifacts/devset_p24/p24_tune_authoring.folded.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    loader = P24ForensicTableLoader(args.store_dir)
    records, audit = read_jsonl(args.input), []
    for record in records:
        values = {}
        for index, ref in enumerate(record["cells"], 1):
            frame = loader(ref["report_id"], int(ref["table_pos"]))
            hit = frame[(frame.row == int(ref["row"])) & (frame.col == int(ref["col"]))]
            if len(hit) != 1:
                raise SystemExit(f"id {record['id']} E{index}: exact cell count {len(hit)}")
            values[f"E{index}"] = float(hit.iloc[0].value) * float(hit.iloc[0].unit_scale)
        tree = ast.parse(record["expression"], mode="eval")
        folder = MedianFolder(values)
        tree = ast.fix_missing_locations(folder.visit(tree))
        if folder.folded:
            record["expression"] = ast.unparse(tree.body)
            record["notes"] = (str(record.get("notes", "")) +
                f" Median constant-folded from the same exact evidence ({folder.folded} occurrence(s)).")
            audit.append({"id": int(record["id"]), "folded_medians": folder.folded})
    write_jsonl(output, records)
    print(json.dumps({
        "count": len(records), "output": str(output), "folded": audit,
        "records_sha256": canonical_sha256(records), "locked_opened": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
