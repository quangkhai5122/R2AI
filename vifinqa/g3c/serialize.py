"""Value-free passage serialization for guarded G3C retrieval."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable

from ..retrieval.serialize import grid_of
from ..utils.viet_num import parse_vn_number
from ..utils.viet_text import norm
from .common import canonical_json_sha256

_NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:\s*%)?(?![0-9A-Za-z])"
)
_YEAR_TOKEN_RE = re.compile(r"^(?:19|20)\d{2}$")


def table_key(value: dict) -> tuple[str, int]:
    return (str(value["report_id"]), int(value["table_pos"]))


def passage_id(kind: str, value: dict) -> str:
    body = {
        "kind": kind,
        "report_id": str(value["report_id"]),
        "table_pos": int(value["table_pos"]),
    }
    if kind == "row":
        body.update({
            "row": int(value.get("row", -1)),
            "label": norm(str(value.get("label", ""))),
            "row_code": str(value.get("row_code", "")),
            "col_name": norm(str(value.get("col_name", ""))),
        })
    return f"{kind}-{canonical_json_sha256(body)[:20]}"


def table_passage(meta: dict, max_labels: int = 80) -> dict:
    """Serialize metadata and row labels while excluding numeric cell values."""
    grid = grid_of(meta)
    header = [
        sanitize_numeric_text(str(cell), preserve_years=True)
        for cell in (grid[0][:16] if grid else [])
        if str(cell).strip()
    ]
    labels = []
    for row in grid[1:]:
        label = _first_label(row)
        if label:
            labels.append(sanitize_numeric_text(label, preserve_years=True))
        if len(labels) >= max_labels:
            break
    context = sanitize_numeric_text(str(meta.get("context", ""))[-320:])
    unit = unit_label(meta.get("unit_scale"), meta.get("unit_source"))
    text = (
        f"ticker={meta['ticker']}; report_id={meta['report_id']}; "
        f"report_year={int(meta['year'])}; scope={meta['doc_type']}; "
        f"table_position={int(meta['table_pos'])}; page={int(meta['page'])}; "
        f"unit={unit}\n"
        f"header: {' | '.join(header)}\n"
        f"row labels: {' | '.join(labels)}\n"
        f"context without values: {context}"
    )
    value = {
        "passage_id": passage_id("table", meta),
        "kind": "table",
        "report_id": str(meta["report_id"]),
        "ticker": str(meta["ticker"]),
        "report_year": int(meta["year"]),
        "doc_type": str(meta["doc_type"]),
        "table_pos": int(meta["table_pos"]),
        "page": int(meta["page"]),
        "content": text,
    }
    value["content_sha256"] = canonical_json_sha256(text)
    return value


def row_passages(
    cells: Iterable[dict], meta: dict, limit: int = 400
) -> list[dict]:
    """Group cell metadata into value-free row passages."""
    grouped: dict[tuple, dict] = {}
    for cell in cells:
        if (
            str(cell.get("report_id")) != str(meta["report_id"])
            or int(cell.get("table_pos", -1)) != int(meta["table_pos"])
        ):
            continue
        key = (
            int(cell.get("row", -1)),
            norm(str(cell.get("label", ""))),
            str(cell.get("row_code", "") or ""),
        )
        row = grouped.setdefault(key, {
            "row": key[0],
            "label": key[1],
            "row_code": key[2],
            "columns": set(),
        })
        column = sanitize_numeric_text(
            str(cell.get("col_name", "")), preserve_years=True
        )
        if column:
            row["columns"].add(column)
    output = []
    for row in sorted(grouped.values(), key=lambda item: item["row"])[:limit]:
        value = {
            "report_id": str(meta["report_id"]),
            "ticker": str(meta["ticker"]),
            "report_year": int(meta["year"]),
            "doc_type": str(meta["doc_type"]),
            "table_pos": int(meta["table_pos"]),
            "page": int(meta["page"]),
            "row": int(row["row"]),
            "label": row["label"],
            "row_code": row["row_code"],
            "col_name": " | ".join(sorted(row["columns"])),
        }
        value["passage_id"] = passage_id("row", value)
        value["kind"] = "row"
        value["content"] = (
            f"ticker={value['ticker']}; report_id={value['report_id']}; "
            f"report_year={value['report_year']}; scope={value['doc_type']}; "
            f"table_position={value['table_pos']}; page={value['page']}; "
            f"unit={unit_label(meta.get('unit_scale'), meta.get('unit_source'))}; "
            f"row={value['row']}; row_code={value['row_code']}; "
            f"row_label={value['label']}; periods={value['col_name']}"
        )
        value["content_sha256"] = canonical_json_sha256(value["content"])
        output.append(value)
    return output


def candidate_from_meta(meta: dict) -> dict:
    scale = meta.get("unit_scale")
    finite_scale = (
        float(scale)
        if scale is not None and not (
            isinstance(scale, float) and math.isnan(scale)
        )
        else None
    )
    return {
        "report_id": str(meta["report_id"]),
        "ticker": str(meta["ticker"]),
        "report_year": int(meta["year"]),
        "doc_type": str(meta["doc_type"]),
        "table_pos": int(meta["table_pos"]),
        "page": int(meta["page"]),
        "unit_scale": finite_scale,
        "unit_source": str(meta.get("unit_source", "")),
        "n_rows": int(meta.get("n_rows", 0)),
    }


def sanitize_numeric_text(text: str, preserve_years: bool = True) -> str:
    def replacement(match: re.Match[str]) -> str:
        token = match.group(0).strip()
        unsigned = token.lstrip("+-")
        if preserve_years and _YEAR_TOKEN_RE.fullmatch(unsigned):
            return token
        return "<num>"
    return " ".join(_NUMBER_RE.sub(replacement, text).split())


def unit_label(scale: object, source: object = "") -> str:
    try:
        number = float(scale)
    except (TypeError, ValueError):
        number = float("nan")
    if math.isfinite(number):
        if number == 1_000_000_000:
            magnitude = "billions"
        elif number == 1_000_000:
            magnitude = "millions"
        elif number == 1_000:
            magnitude = "thousands"
        elif number == 1:
            magnitude = "base-units"
        else:
            magnitude = "known-other-scale"
    else:
        magnitude = "unknown-scale"
    return f"{magnitude}; source={source or 'unknown'}"


def _first_label(row: list[str]) -> str:
    parts = []
    for cell in row[:4]:
        text = str(cell or "").strip()
        if not text:
            continue
        if parse_vn_number(text) is None:
            parts.append(text)
    return " ".join(parts)[:240]
