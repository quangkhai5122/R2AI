"""Complete branch-by-branch semantic audit for all 21 complex tune cases."""
from __future__ import annotations

from statistics import mean, median
from typing import Any

from .p24_semantic_audit import _entry, _values, build_complex_semantic_audit


def build_complete_complex_semantic_audit(gold: list[dict[str, Any]]) -> dict[str, Any]:
    audit = build_complex_semantic_audit(gold)
    rows = {int(record["id"]): record for record in gold}
    checks = []

    r, v = rows[372], _values(rows[372])
    years = [2021, 2022, 2023, 2024]
    quick = [(v[0] - v[1]) / v[2], (v[5] - v[6]) / v[4],
             (v[9] - v[10]) / v[8], (v[13] - v[14]) / v[12]]
    projected = [v[3] / v[4], v[7] / v[8], v[11] / v[12], v[15] / v[16]]
    selected = min(range(4), key=lambda i: quick[i])
    checks.append(_entry(r, {"kind": "argmin_following_year_project",
        "selected_base_year": years[selected], "projected_year": years[selected] + 1,
        "quick_ratio": dict(zip(years, quick)),
        "following_cfo_current_liabilities": dict(zip(years, projected))}, projected[selected]))

    r, v = rows[383], _values(rows[383])
    years = [2021, 2022, 2023, 2024]
    gross = [v[0] / v[1], v[2] / v[3], v[4] / v[5], v[6] / v[7]]
    cfo = [v[8] / v[1], v[9] / v[3], v[10] / v[5], v[11] / v[7]]
    threshold = median(cfo)
    selected_years = [years[i] for i in range(1, 4)
                      if gross[i] > gross[i - 1] and cfo[i] > threshold]
    checks.append(_entry(r, {"kind": "count_after_first_with_dual_condition",
        "median_cfo_margin": threshold, "selected_years": selected_years,
        "gross_margin": dict(zip(years, gross)), "cfo_margin": dict(zip(years, cfo))},
        float(len(selected_years))))

    r, v = rows[468], _values(rows[468])
    tickers = ["DLG", "HHV", "VSC"]
    ratios = [(v[4 * i] - v[4 * i + 3]) / mean([v[4 * i + 1], v[4 * i + 2]]) * 100
              for i in range(3)]
    selected = [i for i in range(3) if v[4 * i] > 0]
    checks.append(_entry(r, {"kind": "positive_pat_conditional_average",
        "selected": [tickers[i] for i in selected],
        "accrual_ratio_percent": dict(zip(tickers, ratios))},
        mean(ratios[i] for i in selected)))

    def steel_check(qid: int, years_arg: list[int], output_year: int) -> dict[str, Any]:
        record, values = rows[qid], _values(rows[qid])
        tickers_arg = ["HPG", "HSG", "MSR", "NKG"]
        eligible, revenue = [], []
        for i, ticker in enumerate(tickers_arg):
            base = 6 * i
            margins = [values[base + 2 * j] / values[base + 2 * j + 1] for j in range(3)]
            if all(value > 0 for value in margins):
                eligible.append(ticker)
                revenue.append(values[base + 5])
        return _entry(record, {"kind": "all_years_positive_margin_conditional_sum",
            "years": years_arg, "output_year": output_year, "selected": eligible}, sum(revenue) / 1e12)

    checks.append(steel_check(473, [2020, 2021, 2022], 2022))

    r, v = rows[481], _values(rows[481])
    years = [2022, 2023, 2024]
    eligible = [i for i in range(3) if v[2 * i + 1] / v[2 * i] > 0.1]
    selected = min(eligible, key=lambda i: v[2 * i])
    checks.append(_entry(r, {"kind": "filter_argmin", "eligible_years": [years[i] for i in eligible],
        "selected_year": years[selected]}, v[2 * selected] / 1e9))

    checks.append(steel_check(493, [2021, 2022, 2023], 2023))

    r, v = rows[539], _values(rows[539])
    tickers = ["BSR", "PLX", "PVT"]
    de = [v[4 * i + 2] / v[4 * i + 3] for i in range(3)]
    coverage = [(v[4 * i + 1] + v[4 * i]) / v[4 * i] for i in range(3)]
    selected = max(range(3), key=lambda i: de[i])
    checks.append(_entry(r, {"kind": "argmax_project", "selected": tickers[selected],
        "de": dict(zip(tickers, de)), "interest_coverage": dict(zip(tickers, coverage))},
        coverage[selected]))

    r, v = rows[551], _values(rows[551])
    tickers = ["GEE", "GEX", "SAM"]
    eligible = [i for i in range(3) if all(v[5 * i + j] > 0 for j in range(3))]
    margins = [v[5 * i + 3] / v[5 * i + 4] for i in range(3)]
    selected = max(eligible, key=lambda i: margins[i])
    checks.append(_entry(r, {"kind": "all_years_positive_cfo_argmax",
        "eligible": [tickers[i] for i in eligible], "selected": tickers[selected],
        "net_margin": dict(zip(tickers, margins))}, margins[selected] * 100))

    checks.append(steel_check(552, [2021, 2022, 2023], 2023))

    def fertilizer_check(qid: int) -> dict[str, Any]:
        record, values = rows[qid], _values(rows[qid])
        tickers_arg = ["DCM", "DPM", "PRT"]
        changes = [(values[4 * i + 3] / values[4 * i + 1]
                    - values[4 * i + 2] / values[4 * i]) * 100 for i in range(3)]
        eligible_arg = [i for i in range(3) if values[4 * i + 1] > values[4 * i]]
        return _entry(record, {"kind": "positive_revenue_growth_conditional_average",
            "selected": [tickers_arg[i] for i in eligible_arg],
            "gross_margin_change_pp": dict(zip(tickers_arg, changes))},
            mean(changes[i] for i in eligible_arg))

    checks.append(fertilizer_check(554))
    checks.append(fertilizer_check(576))

    existing = {int(item["id"]) for item in audit["checks"]}
    added = {int(item["id"]) for item in checks}
    if existing & added or len(existing | added) != 21:
        raise ValueError(f"semantic check universe mismatch: existing={existing}, added={added}")
    audit["checks"] = sorted(audit["checks"] + checks, key=lambda item: int(item["id"]))
    audit["detailed_check_count"] = len(audit["checks"])
    audit["schema_version"] = "p24_complex_semantic_audit_v2"
    return audit
