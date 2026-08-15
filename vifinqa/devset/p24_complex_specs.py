"""Generate the remaining complex P2.4 tune authoring specifications.

This module is deliberately isolated from retrieval and submission code.  It
resolves standard VAS statement metrics to exact cells, then expresses every
filter/rank/project calculation in the typed P2.4 expression language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .p24_authoring import AUTHORING_SCHEMA, P24AuthoringError
from .p24_metrics_v2 import StandardMetricResolverV2


def _ref(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": str(hit["report_id"]),
        "table_pos": int(hit["table_pos"]),
        "row": int(hit["row"]),
        "col": int(hit["col"]),
    }


@dataclass
class CellBook:
    resolver: StandardMetricResolverV2
    cells: list[dict[str, Any]] = field(default_factory=list)
    _ids: dict[tuple[str, int, int, int], str] = field(default_factory=dict)

    def exact(self, ref: dict[str, Any]) -> str:
        clean = _ref(ref)
        key = (
            clean["report_id"], clean["table_pos"], clean["row"], clean["col"],
        )
        if key not in self._ids:
            self.cells.append(clean)
            self._ids[key] = f"E{len(self.cells)}"
        return self._ids[key]

    def metric(
        self,
        ticker: str,
        year: int,
        metric: str,
        doc_type: str = "consolidated",
    ) -> str:
        hit = self.resolver.resolve(ticker, year, doc_type, metric)
        if not hit.get("code_match", False):
            raise P24AuthoringError(
                f"standard metric lacks code agreement: {ticker} {year} "
                f"{doc_type} {metric}: {hit}"
            )
        return self.exact(hit)


def _record(qid: int, book: CellBook, expression: str, notes: str) -> dict[str, Any]:
    return {
        "schema_version": AUTHORING_SCHEMA,
        "id": int(qid),
        "cells": book.cells,
        "expression": expression,
        "notes": notes,
    }


def _m(book: CellBook, ticker: str, year: int, metric: str) -> str:
    return book.metric(ticker, year, metric)


def _de(book: CellBook, ticker: str, year: int) -> str:
    return f"({_m(book, ticker, year, 'total_liabilities')} / {_m(book, ticker, year, 'equity')})"


def _margin(book: CellBook, ticker: str, year: int, numerator: str) -> str:
    return f"({_m(book, ticker, year, numerator)} / {_m(book, ticker, year, 'net_revenue')})"


def _and(parts: list[str]) -> str:
    return "(" + " and ".join(parts) + ")"


def _if(condition: str, yes: str, no: str = "0") -> str:
    return f"({yes} if {condition} else {no})"


def _average_selected(values: list[str], conditions: list[str]) -> str:
    numerator = "sum(" + ", ".join(
        _if(condition, value) for value, condition in zip(values, conditions)
    ) + ")"
    denominator = "count_true(" + ", ".join(conditions) + ")"
    return f"({numerator} / {denominator})"


def _spec_372(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b = CellBook(resolver)
    pairs = []
    for year in range(2021, 2025):
        quick = (
            f"(({_m(b, 'VRE', year, 'current_assets')} - "
            f"{_m(b, 'VRE', year, 'inventory')}) / "
            f"{_m(b, 'VRE', year, 'current_liabilities')})"
        )
        following = (
            f"({_m(b, 'VRE', year + 1, 'cfo')} / "
            f"{_m(b, 'VRE', year + 1, 'current_liabilities')})"
        )
        pairs.extend([quick, following])
    return _record(372, b, f"argmin_project({', '.join(pairs)})",
        "Rank VRE quick ratios in 2021-2024 and project the following year's CFO/current-liabilities ratio.")


def _spec_375(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, tickers = CellBook(resolver), ["DCM", "DPM", "GVR", "PRT"]
    des, cover = [], []
    for ticker in tickers:
        des.append(_de(b, ticker, 2021))
        interest = _m(b, ticker, 2021, "interest_expense")
        cover.append(f"(({_m(b, ticker, 2021, 'pbt')} + {interest}) / {interest})")
    med = f"median({', '.join(des)})"
    high = [f"({value} > {med})" for value in des]
    low = [f"({value} <= {med})" for value in des]
    expr = f"({_average_selected(cover, high)} - {_average_selected(cover, low)})"
    return _record(375, b, expr,
        "Difference between mean interest coverage above the four-company median D/E and the remaining group.")


def _spec_383(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b = CellBook(resolver)
    gross = {year: _margin(b, "MWG", year, "gross_profit") for year in range(2021, 2025)}
    cfo = {year: _margin(b, "MWG", year, "cfo") for year in range(2021, 2025)}
    med = f"median({', '.join(cfo.values())})"
    conditions = [
        _and([f"({gross[year]} > {gross[year - 1]})", f"({cfo[year]} > {med})"])
        for year in range(2022, 2025)
    ]
    return _record(383, b, f"count_true({', '.join(conditions)})",
        "Count 2022-2024 years with improving gross margin and CFO margin above the 2021-2024 median.")


def _spec_397(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b = CellBook(resolver)
    pairs = []
    for ticker in ["DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"]:
        ca = _m(b, ticker, 2024, "current_assets")
        inv = _m(b, ticker, 2024, "inventory")
        cl = _m(b, ticker, 2024, "current_liabilities")
        current, quick = f"({ca} / {cl})", f"(({ca} - {inv}) / {cl})"
        pairs.extend([_if(f"({current} > 1.5)", quick, "1e99"), f"({inv} / 1e12)"])
    return _record(397, b, f"argmin_project({', '.join(pairs)})",
        "Filter 2024 current ratio above 1.5, rank by quick ratio, and project inventory in trillion VND.")


def _spec_417(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, pairs = CellBook(resolver), []
    for ticker in ["MSN", "DBC", "ASM", "MPC", "OGC"]:
        revenue = _m(b, ticker, 2024, "net_revenue")
        score = (
            f"(({_m(b, ticker, 2024, 'cfo')} / {revenue}) - "
            f"({_m(b, ticker, 2024, 'pat')} / {revenue}))"
        )
        pairs.extend([score, _de(b, ticker, 2024)])
    return _record(417, b, f"argmax_project({', '.join(pairs)})",
        "Rank the five food companies by CFO margin minus net margin and project 2024 D/E.")


def _spec_425(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, pairs = CellBook(resolver), []
    for year in range(2021, 2025):
        avg_equity = f"average({_m(b, 'FPT', year - 1, 'equity')}, {_m(b, 'FPT', year, 'equity')})"
        roae = f"({_m(b, 'FPT', year, 'pat')} / {avg_equity})"
        diluted_eps_thousand = f"({_m(b, 'FPT', year, 'basic_eps')} / 1.1 / 1000)"
        pairs.extend([roae, diluted_eps_thousand])
    return _record(425, b, f"argmax_project({', '.join(pairs)})",
        "Select the highest ROAE year using beginning/end equity; 10% more shares with unchanged profit makes scenario EPS = reported basic EPS/1.1, expressed in thousand VND/share.")


def _spec_446(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, tickers = CellBook(resolver), ["DBC", "MCH", "MSN", "OGC", "QNS", "VNM"]
    des = [_de(b, ticker, 2024) for ticker in tickers]
    pats = [_m(b, ticker, 2024, "pat") for ticker in tickers]
    med = f"median({', '.join(des)})"
    selected = "sum(" + ", ".join(
        _if(f"({de} < {med})", pat) for de, pat in zip(des, pats)
    ) + ")"
    return _record(446, b, f"({selected} / sum({', '.join(pats)}) * 100)",
        "Share of aggregate positive PAT contributed by companies below the six-company median D/E.")


def _spec_447(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, tickers = CellBook(resolver), ["ASM", "DBC", "MCH", "MSN", "OGC", "VNM"]
    growth, pbt, interest, gross_margin = [], [], [], []
    for ticker in tickers:
        r24, r25 = _m(b, ticker, 2024, "net_revenue"), _m(b, ticker, 2025, "net_revenue")
        growth.append(f"(({r25} - {r24}) / {r24})")
        pbt.append(_m(b, ticker, 2025, "pbt"))
        interest.append(_m(b, ticker, 2025, "interest_expense"))
        gross_margin.append(_margin(b, ticker, 2025, "gross_profit"))
    med = f"median({', '.join(growth)})"
    selected = [f"({item} > {med})" for item in growth]
    numerator = "sum(" + ", ".join(
        _if(cond, f"({profit} + {cost})")
        for cond, profit, cost in zip(selected, pbt, interest)
    ) + ")"
    denominator_pairs = []
    for cond, margin, cost in zip(selected, gross_margin, interest):
        denominator_pairs.extend([_if(cond, margin, "-1e99"), cost])
    denominator = f"argmax_project({', '.join(denominator_pairs)})"
    return _record(447, b, f"({numerator} / {denominator})",
        "Filter above-median 2024-2025 revenue growth; divide selected aggregate PBT plus interest by interest of the selected highest-gross-margin company.")


def _spec_468(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, values, conditions = CellBook(resolver), [], []
    for ticker in ["DLG", "HHV", "VSC"]:
        pat = _m(b, ticker, 2020, "pat")
        avg_assets = f"average({_m(b, ticker, 2019, 'total_assets')}, {_m(b, ticker, 2020, 'total_assets')})"
        values.append(f"(({pat} - {_m(b, ticker, 2020, 'cfo')}) / {avg_assets} * 100)")
        conditions.append(f"({pat} > 0)")
    return _record(468, b, _average_selected(values, conditions),
        "Average accrual ratio for the 2020 positive-PAT subset, using average 2019/2020 total assets.")


def _steel_revenue_spec(resolver: StandardMetricResolverV2, qid: int, years: list[int], output_year: int) -> dict[str, Any]:
    b, selected = CellBook(resolver), []
    for ticker in ["HPG", "HSG", "MSR", "NKG"]:
        ratios = [
            f"({_m(b, ticker, year, 'pat')} / {_m(b, ticker, year, 'net_revenue')})"
            for year in years
        ]
        selected.append(_if(_and([f"({ratio} > 0)" for ratio in ratios]),
                            _m(b, ticker, output_year, "net_revenue")))
    return _record(qid, b, f"(sum({', '.join(selected)}) / 1e12)",
        f"Sum {output_year} revenue for steel companies with positive PAT/revenue in every year {years}; VND to trillion VND.")


def _spec_481(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, candidates = CellBook(resolver), []
    for year in range(2022, 2025):
        revenue, pat = _m(b, "CEO", year, "net_revenue"), _m(b, "CEO", year, "pat")
        candidates.append(_if(f"(({pat} / {revenue}) > 0.1)", f"({revenue} / 1e9)", "1e99"))
    return _record(481, b, f"min({', '.join(candidates)})",
        "Minimum CEO revenue among 2022-2024 years with net margin above 10%, in billion VND.")


def _spec_508(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b = CellBook(resolver)
    balances = [
        b.exact({"report_id": "OCB_financial_statements_2021_separate", "table_pos": 55, "row": 1, "col": 1}),
        b.exact({"report_id": "ACB_financial_statements_2021_separate", "table_pos": 47, "row": 1, "col": 1}),
        b.exact({"report_id": "STB_financial_statements_2021_separate", "table_pos": 63, "row": 5, "col": 1}),
    ]
    outputs = ["0", "0", f"({b.exact({'report_id': 'STB_financial_statements_2021_separate', 'table_pos': 7, 'row': 11, 'col': 2})} / 1e6)"]
    return _record(508, b, f"argmax_project({', '.join(x for pair in zip(balances, outputs) for x in pair)})",
        "Exact parent-bank prepaid-allocation balances identify STB; project STB net other operating income in million VND. Non-winning projections are inert placeholders.")


def _spec_512(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, pairs = CellBook(resolver), []
    for year in [2015, 2018, 2019, 2021, 2022, 2023]:
        pairs.extend([
            _m(b, "HSG", year, "equity"),
            f"({_m(b, 'HSG', year, 'long_term_borrowings')} / 1e9)",
        ])
    return _record(512, b, f"argmax_project({', '.join(pairs)})",
        "Rank HSG fiscal year-end equity across the six stated years and project long-term borrowings in billion VND.")


def _spec_516(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b = CellBook(resolver)
    funds = [
        b.exact({"report_id": "ACB_financial_statements_2015_separate", "table_pos": 82, "row": 7, "col": 1}),
        b.exact({"report_id": "ACB_financial_statements_2019_separate", "table_pos": 58, "row": 9, "col": 1}),
        b.exact({"report_id": "ACB_financial_statements_2022_separate", "table_pos": 65, "row": 8, "col": 1}),
    ]
    current = b.exact({"report_id": "ACB_financial_statements_2022_separate", "table_pos": 107, "row": 7, "col": 9})
    previous = b.exact({"report_id": "ACB_financial_statements_2022_separate", "table_pos": 108, "row": 7, "col": 9})
    change = f"(({current} - {previous}) / abs({previous}) * 100)"
    projections = ["0", "0", change]
    return _record(516, b, f"argmax_project({', '.join(x for pair in zip(funds, projections) for x in pair)})",
        "Exact parent-ACB welfare-fund balances select 2022; compare total recorded derivative/other-financial-asset value at 2022 versus 2021 year-end.")


def _spec_539(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, pairs = CellBook(resolver), []
    for ticker in ["BSR", "PLX", "PVT"]:
        interest = _m(b, ticker, 2019, "interest_expense")
        coverage = f"(({_m(b, ticker, 2019, 'pbt')} + {interest}) / {interest})"
        pairs.extend([_de(b, ticker, 2019), coverage])
    return _record(539, b, f"argmax_project({', '.join(pairs)})",
        "Rank BSR/PLX/PVT by 2019 D/E and project interest coverage for the maximum.")


def _spec_551(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, pairs = CellBook(resolver), []
    for ticker in ["GEE", "GEX", "SAM"]:
        positive = _and([f"({_m(b, ticker, year, 'cfo')} > 0)" for year in [2022, 2023, 2024]])
        margin = _margin(b, ticker, 2024, "pat")
        pairs.extend([_if(positive, margin, "-1e99"), f"({margin} * 100)"])
    return _record(551, b, f"argmax_project({', '.join(pairs)})",
        "Filter companies with positive CFO in all three years, rank 2024 net margin, and report percent.")


def _fertilizer_spec(resolver: StandardMetricResolverV2, qid: int) -> dict[str, Any]:
    b, values, conditions = CellBook(resolver), [], []
    for ticker in ["DCM", "DPM", "PRT"]:
        r19, r20 = _m(b, ticker, 2019, "net_revenue"), _m(b, ticker, 2020, "net_revenue")
        g19, g20 = _m(b, ticker, 2019, "gross_profit"), _m(b, ticker, 2020, "gross_profit")
        conditions.append(f"({r20} > {r19})")
        values.append(f"((({g20} / {r20}) - ({g19} / {r19})) * 100)")
    return _record(qid, b, _average_selected(values, conditions),
        "Mean gross-margin percentage-point change for DCM/DPM/PRT companies with positive 2019-2020 revenue growth.")


def _spec_570(resolver: StandardMetricResolverV2) -> dict[str, Any]:
    b, pairs = CellBook(resolver), []
    for ticker in ["HPG", "HSG", "MSR", "NKG"]:
        inv20, inv21, inv22 = (
            _m(b, ticker, year, "inventory") for year in [2020, 2021, 2022]
        )
        days21 = f"(365 * average({inv20}, {inv21}) / abs({_m(b, ticker, 2021, 'cogs')}))"
        days22 = f"(365 * average({inv21}, {inv22}) / abs({_m(b, ticker, 2022, 'cogs')}))"
        change_days = f"({days22} - {days21})"
        margin_change = (
            f"((({_m(b, ticker, 2022, 'gross_profit')} / {_m(b, ticker, 2022, 'net_revenue')}) - "
            f"({_m(b, ticker, 2021, 'gross_profit')} / {_m(b, ticker, 2021, 'net_revenue')})) * 100)"
        )
        pairs.extend([change_days, margin_change])
    return _record(570, b, f"argmax_project({', '.join(pairs)})",
        "Rank increase in 365*average inventory/COGS from 2021 to 2022 and project gross-margin change in percentage points.")


def build_complex_tune_specs(store_dir: Path | str) -> list[dict[str, Any]]:
    """Return the 21 audited complex tune specs without reading locked files."""
    resolver = StandardMetricResolverV2(store_dir)
    records = [
        _spec_372(resolver), _spec_375(resolver), _spec_383(resolver),
        _spec_397(resolver), _spec_417(resolver), _spec_425(resolver),
        _spec_446(resolver), _spec_447(resolver), _spec_468(resolver),
        _steel_revenue_spec(resolver, 473, [2020, 2021, 2022], 2022),
        _spec_481(resolver),
        _steel_revenue_spec(resolver, 493, [2021, 2022, 2023], 2023),
        _spec_508(resolver), _spec_512(resolver), _spec_516(resolver),
        _spec_539(resolver), _spec_551(resolver),
        _steel_revenue_spec(resolver, 552, [2021, 2022, 2023], 2023),
        _fertilizer_spec(resolver, 554), _spec_570(resolver),
        _fertilizer_spec(resolver, 576),
    ]
    expected = {372, 375, 383, 397, 417, 425, 446, 447, 468, 473, 481,
                493, 508, 512, 516, 539, 551, 552, 554, 570, 576}
    actual = {int(record["id"]) for record in records}
    if len(records) != len(actual) or actual != expected:
        raise P24AuthoringError(f"complex spec id mismatch: {sorted(actual ^ expected)}")
    return sorted(records, key=lambda record: int(record["id"]))
