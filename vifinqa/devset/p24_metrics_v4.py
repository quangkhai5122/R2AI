"""Whitespace-tolerant exact accounting-label resolver for P2.4."""
from __future__ import annotations

import re
from typing import Any

from .p24_metrics import METRICS, _canon_code, _norm
from .p24_metrics_v2 import StandardMetricResolverV2


def _compact(value: object) -> str:
    return _norm(value).replace(" ", "")


class StandardMetricResolverV4(StandardMetricResolverV2):
    """Trust only full accounting phrases, ignoring OCR whitespace loss."""

    def candidates(
        self, ticker: str, year: int, doc_type: str, metric: str,
    ) -> list[dict[str, Any]]:
        definition = METRICS[metric]
        report_id = f"{ticker.upper()}_financial_statements_{int(year)}_{doc_type}"
        frame = self._cells(ticker)
        frame = frame[frame.report_id == report_id]
        labels = tuple(_compact(label) for label in definition.labels)
        rejects = tuple(_compact(label) for label in definition.reject)
        codes = {_canon_code(code) for code in definition.codes}
        grouped: dict[tuple[int, int], Any] = {}
        for row in frame.itertuples():
            label = _compact(row.label)
            if any(term and term in label for term in rejects):
                continue
            phrase = max(
                (2 if label == target else 1 if target in label else 0)
                for target in labels
            )
            if phrase == 0:
                continue
            code_match = _canon_code(row.row_code) in codes
            key = (int(row.table_pos), int(row.row))
            score = 1000.0 * phrase + 200.0 * code_match - int(row.table_pos)
            if key not in grouped or score > grouped[key][0]:
                grouped[key] = (score, row, phrase, code_match)

        results: list[dict[str, Any]] = []
        for (table_pos, row_no), (score, exemplar, phrase, code_match) in grouped.items():
            row_cells = frame[
                (frame.table_pos.astype(int) == table_pos)
                & (frame.row.astype(int) == row_no)
            ]
            choices = []
            for cell in row_cells.itertuples():
                header_norm = _norm(cell.col_name)
                header = header_norm.replace(" ", "")
                if "maso" in header or "thuyetminh" in header:
                    continue
                header_score = -0.01 * int(cell.col)
                if re.search(rf"\b{int(year)}\b", header_norm):
                    header_score += 40.0
                if any(term in header for term in ("namnay", "cuoinam", "socuo inam", "socuo nam")):
                    header_score += 30.0
                if any(term in header for term in ("namtruoc", "sodaunam", "daunam")):
                    header_score -= 30.0
                choices.append((header_score, cell))
            if not choices:
                continue
            header_score, cell = max(choices, key=lambda item: item[0])
            results.append({
                "metric": metric, "report_id": report_id,
                "table_pos": table_pos, "row": row_no, "col": int(cell.col),
                "label": str(exemplar.label), "code": str(exemplar.row_code),
                "col_name": str(cell.col_name), "value": float(cell.value),
                "unit_scale": float(cell.unit_scale),
                "score": round(score + header_score, 4),
                "code_match": True,
                "resolution": "code_and_compact_exact_label" if code_match else "compact_exact_label",
            })
        return sorted(results, key=lambda item: item["score"], reverse=True)
