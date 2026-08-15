"""P2.4 tune authoring extensions kept isolated from the submission pipeline.

The production tidy serializer intentionally remains frozen.  The forensic
loader below augments its DataFrame with numeric raw-grid cells that were
mistaken for note codes by OCR (for example ``307.293`` in the first value
column).  Exact authoring specs still have to name a row and column, so these
extra cells cannot be selected accidentally by the gold builder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..extraction.build_store import Store
from ..retrieval.serialize import TIDY_COLS, df_roundtrip, tidy_csv_text
from ..utils.io import read_jsonl, write_jsonl
from ..utils.viet_num import parse_vn_number
from . import p24 as core
from .p24_authoring import (
    P24AuthoringError,
    build_tune_gold_records,
)


# Gold-only typed operations.  Updating these module-level contracts makes the
# existing strict validator enforce the extended AST without changing runtime
# codegen or retrieval behavior.
core.ALLOWED_AST_OPS.update({
    "power", "min", "max", "median", "count_true",
    "gt", "ge", "lt", "le", "eq", "ne", "and", "or", "if_else",
    "argmax_project", "argmin_project",
})
core.EXACT_ARITY.update({
    "power": 2, "gt": 2, "ge": 2, "lt": 2, "le": 2,
    "eq": 2, "ne": 2, "if_else": 3,
})
core.MIN_ARITY.update({
    "min": 1, "max": 1, "median": 1, "count_true": 1,
    "and": 2, "or": 2, "argmax_project": 4, "argmin_project": 4,
})


class P24ForensicTableLoader:
    """Load exact evidence while retaining raw numeric OCR cells.

    This class is used only by P2.4 authoring and audit.  It never writes to the
    store and does not alter the DataFrames used to build Kaggle submissions.
    """

    def __init__(self, store_dir: Path | str):
        self.store = Store(Path(store_dir))
        self.cache: dict[tuple[str, int], pd.DataFrame] = {}

    def __call__(self, report_id: str, table_pos: int) -> pd.DataFrame:
        key = (str(report_id), int(table_pos))
        if key not in self.cache:
            ticker = key[0].split("_")[0]
            tables = self.store.tables_of(ticker, [key[0]])
            hit = tables[
                (tables.report_id == key[0]) & (tables.table_pos == key[1])
            ]
            if len(hit) != 1:
                raise P24AuthoringError(
                    f"store evidence table {key[0]}|{key[1]}: expected one row, "
                    f"found {len(hit)}"
                )
            meta = hit.iloc[0].to_dict()
            tidy = df_roundtrip(tidy_csv_text(meta))
            grid = json.loads(meta["grid_json"])
            header = grid[0] if grid else []
            unit_scale = meta.get("unit_scale")
            scale = 1.0 if unit_scale is None or pd.isna(unit_scale) or not unit_scale else float(unit_scale)
            occupied = {(int(row.row), int(row.col)) for row in tidy.itertuples()}
            extras: list[dict[str, Any]] = []
            for row_no, raw_row in enumerate(grid[:400]):
                label_parts = [
                    str(cell).strip() for cell in raw_row[:6]
                    if str(cell).strip() and parse_vn_number(cell) is None
                ]
                label = " ".join(label_parts)[:160]
                for col_no, cell in enumerate(raw_row):
                    value = parse_vn_number(cell)
                    if value is None or (row_no, col_no) in occupied:
                        continue
                    extras.append({
                        "row": row_no,
                        "label": label,
                        "code": "",
                        "col": col_no,
                        "col_name": (
                            str(header[col_no]) if row_no > 0 and col_no < len(header) else ""
                        )[:80],
                        "value": float(value),
                        "unit_scale": scale,
                    })
            if extras:
                tidy = pd.concat(
                    [tidy, pd.DataFrame(extras, columns=TIDY_COLS)],
                    ignore_index=True,
                ).sort_values(["row", "col"], kind="stable").reset_index(drop=True)
            self.cache[key] = tidy
        return self.cache[key].copy()


def validate_extended_ast_shape(node: dict[str, Any]) -> None:
    """Enforce pair arity not expressible through the legacy min-arity map."""
    if not isinstance(node, dict):
        return
    if node.get("kind") == "op":
        op, args = node.get("op"), node.get("args", [])
        if op in {"argmax_project", "argmin_project"} and len(args) % 2:
            raise P24AuthoringError(f"{op} requires score/project pairs")
        for arg in args:
            validate_extended_ast_shape(arg)


def build_extended_tune_gold_file(
    specs_path: Path | str,
    bundle_dir: Path | str,
    output_path: Path | str,
    store_dir: Path | str,
) -> dict[str, Any]:
    bundle, output = Path(bundle_dir), Path(output_path)
    if output.exists():
        raise P24AuthoringError(f"refusing to overwrite {output}")
    loader = P24ForensicTableLoader(store_dir)
    records = build_tune_gold_records(
        read_jsonl(specs_path),
        read_jsonl(bundle / "p24_tune_questions.jsonl"),
        read_jsonl(bundle / "p24_tune_gold.template.jsonl"),
        table_loader=loader,
    )
    for record in records:
        validate_extended_ast_shape(record["ast"])
    write_jsonl(output, records)
    return {
        "count": len(records),
        "output": str(output),
        "records_sha256": core.canonical_sha256(records),
        "bundle_fingerprint_sha256": core.read_json(bundle / core.MANIFEST_NAME)[
            "bundle_fingerprint_sha256"
        ],
        "locked_opened": False,
    }
