"""P2.4 resolver with a narrow VAS-code fallback for damaged total labels."""
from __future__ import annotations

import re
from typing import Any

from .p24_metrics import METRICS, _canon_code, _norm
from .p24_metrics_v4 import StandardMetricResolverV4


class StandardMetricResolverV5(StandardMetricResolverV4):
    """Use exact labels first; permit code 100 when its label is OCR-damaged."""

    def candidates(
        self, ticker: str, year: int, doc_type: str, metric: str,
    ) -> list[dict[str, Any]]:
        hits = super().candidates(ticker, year, doc_type, metric)
        if hits or metric != "current_assets":
            return hits
        definition = METRICS[metric]
        wanted = {_canon_code(code) for code in definition.codes}
        report_id = f"{ticker.upper()}_financial_statements_{int(year)}_{doc_type}"
        frame = self._cells(ticker)
        frame = frame[frame.report_id == report_id]
        frame = frame[frame.row_code.map(_canon_code).isin(wanted)]
        results: list[dict[str, Any]] = []
        for (table_pos, row_no), row_cells in frame.groupby(["table_pos", "row"]):
            choices = []
            for cell in row_cells.itertuples():
                header_norm = _norm(cell.col_name)
                header = header_norm.replace(" ", "")
                if "maso" in header or "thuyetminh" in header:
                    continue
                score = -0.01 * int(cell.col)
                if re.search(rf"\b{int(year)}\b", header_norm):
                    score += 40.0
                if any(term in header for term in ("namnay", "cuoinam")):
                    score += 30.0
                if any(term in header for term in ("namtruoc", "sodaunam", "daunam")):
                    score -= 30.0
                choices.append((score, cell))
            if not choices:
                continue
            header_score, cell = max(choices, key=lambda item: item[0])
            results.append({
                "metric": metric, "report_id": report_id,
                "table_pos": int(table_pos), "row": int(row_no), "col": int(cell.col),
                "label": str(cell.label), "code": str(cell.row_code),
                "col_name": str(cell.col_name), "value": float(cell.value),
                "unit_scale": float(cell.unit_scale),
                "score": round(3000.0 - int(table_pos) + header_score, 4),
                "code_match": True, "resolution": "exact_vas_code_100",
            })
        return sorted(results, key=lambda item: item["score"], reverse=True)
