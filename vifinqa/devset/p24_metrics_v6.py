"""Final exact-label resolver with strict value-column classification."""
from __future__ import annotations

import re
from typing import Any

from .p24_metrics import _norm
from .p24_metrics_v5 import StandardMetricResolverV5


_METADATA_HEADERS = {
    "chitieu", "item", "items", "stt", "no", "number",
    "ms", "maso", "code", "tm", "thuyetminh", "notes", "note",
}


def _header_key(value: object) -> str:
    return re.sub(r"[^0-9a-z]+", "", _norm(value))


class StandardMetricResolverV6(StandardMetricResolverV5):
    """Reselect every resolved row from columns that can contain amounts."""

    def candidates(
        self, ticker: str, year: int, doc_type: str, metric: str,
    ) -> list[dict[str, Any]]:
        hits = super().candidates(ticker, year, doc_type, metric)
        report_id = f"{ticker.upper()}_financial_statements_{int(year)}_{doc_type}"
        frame = self._cells(ticker)
        frame = frame[frame.report_id == report_id]
        cleaned = []
        for hit in hits:
            row_cells = frame[
                (frame.table_pos.astype(int) == int(hit["table_pos"]))
                & (frame.row.astype(int) == int(hit["row"]))
            ]
            choices = []
            for cell in row_cells.itertuples():
                header_norm = _norm(cell.col_name)
                header = _header_key(cell.col_name)
                if not header or header in _METADATA_HEADERS:
                    continue
                score = -0.01 * int(cell.col)
                if re.search(rf"\b{int(year)}\b", header_norm):
                    score += 40.0
                if any(term in header for term in ("namnay", "currentyear", "cuoinam", "ending")):
                    score += 30.0
                if any(term in header for term in ("namtruoc", "previousyear", "sodaunam", "daunam", "beginning")):
                    score -= 30.0
                choices.append((score, cell))
            if not choices:
                continue
            _, cell = max(choices, key=lambda item: item[0])
            cleaned.append({
                **hit,
                "col": int(cell.col), "col_name": str(cell.col_name),
                "value": float(cell.value), "unit_scale": float(cell.unit_scale),
                "resolution": str(hit.get("resolution", "exact")) + "+strict_value_column",
            })
        return cleaned
