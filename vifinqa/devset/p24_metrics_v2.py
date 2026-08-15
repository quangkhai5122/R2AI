"""Stricter P2.4 statement resolver layered over :mod:`p24_metrics`."""
from __future__ import annotations

import re
from typing import Any

from .p24_metrics import METRICS, StandardMetricResolver, _canon_code, _norm


class StandardMetricResolverV2(StandardMetricResolver):
    """Require VAS-code agreement when available and reject compact headers."""

    def candidates(self, ticker: str, year: int, doc_type: str, metric: str) -> list[dict[str, Any]]:
        hits = super().candidates(ticker, year, doc_type, metric)
        definition = METRICS[metric]
        wanted_codes = {_canon_code(x) for x in definition.codes}
        report_id = f"{ticker.upper()}_financial_statements_{int(year)}_{doc_type}"
        frame = self._cells(ticker)
        frame = frame[frame.report_id == report_id]
        rescored = []
        for hit in hits:
            row_cells = frame[
                (frame.table_pos.astype(int) == int(hit["table_pos"]))
                & (frame.row.astype(int) == int(hit["row"]))
            ]
            choices = []
            for cell in row_cells.itertuples():
                header = _norm(cell.col_name).replace(" ", "")
                if "maso" in header or "thuyetminh" in header:
                    continue
                header_score = -0.01 * int(cell.col)
                if re.search(rf"\b{int(year)}\b", _norm(cell.col_name)):
                    header_score += 40.0
                if any(x in header for x in ("namnay", "socuoiyear", "socuo năm", "socuo inam", "socuo inam")):
                    header_score += 30.0
                if any(x in header for x in ("sonamdau", "sodaunam", "namtruoc", "daunam")):
                    header_score -= 30.0
                choices.append((header_score, cell))
            if choices:
                _, cell = max(choices, key=lambda item: item[0])
                hit = {**hit, "col": int(cell.col), "col_name": str(cell.col_name),
                       "value": float(cell.value), "unit_scale": float(cell.unit_scale)}
            code_match = _canon_code(hit["code"]) in wanted_codes
            hit["score"] = round(float(hit["score"]) + (70.0 if code_match else -70.0), 4)
            hit["code_match"] = code_match
            rescored.append(hit)
        if any(item["code_match"] for item in rescored):
            rescored = [item for item in rescored if item["code_match"]]
        return sorted(rescored, key=lambda item: item["score"], reverse=True)
