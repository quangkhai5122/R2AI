"""Human-readable branch audits for the complex P2.4 tune questions."""
from __future__ import annotations

import math
import re
from statistics import mean, median
from typing import Any

from .p24_metrics import _norm


COMPLEX_IDS = {372, 375, 383, 397, 417, 425, 446, 447, 468, 473, 481,
               493, 508, 512, 516, 539, 551, 552, 554, 570, 576}


def _values(record: dict[str, Any]) -> list[float]:
    return [float(item["value"]) * float(item["unit_scale"])
            for item in record["evidence"]]


def _close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=0.011)


def _entry(record: dict[str, Any], detail: dict[str, Any], recomputed: float) -> dict[str, Any]:
    answer = float(record["output"]["value"])
    if not _close(answer, round(float(recomputed), int(record["output"]["round_decimals"]))):
        raise ValueError(f"id {record['id']}: semantic branch replay mismatch")
    return {"id": int(record["id"]), "answer": answer, "recomputed": recomputed, **detail}


def build_complex_semantic_audit(gold: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {int(record["id"]): record for record in gold}
    if not COMPLEX_IDS <= set(rows):
        raise ValueError(f"missing complex ids: {sorted(COMPLEX_IDS-set(rows))}")
    checks = []

    r, v = rows[375], _values(rows[375])
    tickers = ["DCM", "DPM", "GVR", "PRT"]
    de = [v[i] / v[i + 1] for i in range(0, 16, 4)]
    cover = [(v[i + 3] + v[i + 2]) / v[i + 2] for i in range(0, 16, 4)]
    threshold = median(de)
    high = [i for i, value in enumerate(de) if value > threshold]
    low = [i for i, value in enumerate(de) if value <= threshold]
    recomputed = mean(cover[i] for i in high) - mean(cover[i] for i in low)
    checks.append(_entry(r, {"kind": "median_partition", "median_de": threshold,
        "above": [tickers[i] for i in high], "remaining": [tickers[i] for i in low],
        "interest_coverage": dict(zip(tickers, cover))}, recomputed))

    r, v = rows[397], _values(rows[397])
    tickers = ["DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"]
    current = [v[i] / v[i + 2] for i in range(0, 24, 3)]
    quick = [(v[i] - v[i + 1]) / v[i + 2] for i in range(0, 24, 3)]
    eligible = [i for i, value in enumerate(current) if value > 1.5]
    selected = min(eligible, key=lambda i: quick[i])
    checks.append(_entry(r, {"kind": "filter_argmin_project",
        "eligible": [tickers[i] for i in eligible], "selected": tickers[selected],
        "current_ratio": dict(zip(tickers, current)), "quick_ratio": dict(zip(tickers, quick))},
        v[3 * selected + 1] / 1e12))

    r, v = rows[417], _values(rows[417])
    tickers = ["MSN", "DBC", "ASM", "MPC", "OGC"]
    scores = [(v[i + 1] - v[i + 2]) / v[i] for i in range(0, 25, 5)]
    selected = max(range(5), key=lambda i: scores[i])
    checks.append(_entry(r, {"kind": "argmax_project", "selected": tickers[selected],
        "cfo_minus_net_margin": dict(zip(tickers, scores))},
        v[5 * selected + 3] / v[5 * selected + 4]))

    r, v = rows[425], _values(rows[425])
    years, roae, eps = [2021, 2022, 2023, 2024], [], []
    # E1=2020 equity; each year adds end equity, PAT, basic EPS.
    for index in range(4):
        start = 0 if index == 0 else 1 + 3 * (index - 1)
        end, pat, basic_eps = 1 + 3 * index, 2 + 3 * index, 3 + 3 * index
        roae.append(v[pat] / mean([v[start], v[end]]))
        eps.append(v[basic_eps] / 1.1 / 1000)
    selected = max(range(4), key=lambda i: roae[i])
    checks.append(_entry(r, {"kind": "scenario_argmax_project", "selected_year": years[selected],
        "roae": dict(zip(years, roae)), "diluted_eps_thousand": dict(zip(years, eps)),
        "assumption": "10 percent more shares from year start; unchanged profit; EPS divided by 1.1"},
        eps[selected]))

    r, v = rows[446], _values(rows[446])
    tickers = ["DBC", "MCH", "MSN", "OGC", "QNS", "VNM"]
    de = [v[2 * i] / v[2 * i + 1] for i in range(6)]
    pats, threshold = v[12:18], median(de)
    selected = [i for i, value in enumerate(de) if value < threshold]
    recomputed = sum(pats[i] for i in selected) / sum(pats) * 100
    checks.append(_entry(r, {"kind": "below_median_share", "median_de": threshold,
        "selected": [tickers[i] for i in selected], "de": dict(zip(tickers, de))}, recomputed))

    r, v = rows[447], _values(rows[447])
    tickers = ["ASM", "DBC", "MCH", "MSN", "OGC", "VNM"]
    growth = [(v[5 * i + 1] - v[5 * i]) / v[5 * i] for i in range(6)]
    threshold = median(growth)
    selected = [i for i, value in enumerate(growth) if value > threshold]
    gross_margin = [v[5 * i + 4] / v[5 * i + 1] for i in range(6)]
    denominator_firm = max(selected, key=lambda i: gross_margin[i])
    numerator = sum(v[5 * i + 2] + v[5 * i + 3] for i in selected)
    recomputed = numerator / v[5 * denominator_firm + 3]
    checks.append(_entry(r, {"kind": "filter_aggregate_argmax_denominator",
        "median_growth": threshold, "selected": [tickers[i] for i in selected],
        "denominator_firm": tickers[denominator_firm],
        "growth": dict(zip(tickers, growth)),
        "gross_margin": dict(zip(tickers, gross_margin)),
        "numerator": numerator, "denominator_interest": v[5 * denominator_firm + 3]}, recomputed))

    r, v = rows[508], _values(rows[508])
    banks = ["OCB", "ACB", "STB"]
    selected = max(range(3), key=lambda i: v[i])
    checks.append(_entry(r, {"kind": "exact_note_argmax_project", "selected": banks[selected],
        "prepaid_allocation_balances": dict(zip(banks, v[:3]))}, v[3] / 1e6))

    r, v = rows[512], _values(rows[512])
    years = [2015, 2018, 2019, 2021, 2022, 2023]
    selected = max(range(6), key=lambda i: v[2 * i])
    checks.append(_entry(r, {"kind": "argmax_project", "selected_year": years[selected],
        "equity": {year: v[2 * i] for i, year in enumerate(years)}},
        v[2 * selected + 1] / 1e9))

    r, v = rows[516], _values(rows[516])
    years = [2015, 2019, 2022]
    selected = max(range(3), key=lambda i: v[i])
    change = (v[3] - v[4]) / abs(v[4]) * 100
    checks.append(_entry(r, {"kind": "exact_note_argmax_project",
        "selected_year": years[selected], "welfare_fund": dict(zip(years, v[:3])),
        "derivative_current": v[3], "derivative_previous": v[4],
        "interpretation": "total recorded derivative and other financial assets row"}, change))

    r, v = rows[570], _values(rows[570])
    tickers = ["HPG", "HSG", "MSR", "NKG"]
    increase, margin_change = [], []
    for i in range(4):
        x = 9 * i
        days21 = 365 * mean([v[x], v[x + 1]]) / abs(v[x + 3])
        days22 = 365 * mean([v[x + 1], v[x + 2]]) / abs(v[x + 4])
        increase.append(days22 - days21)
        margin_change.append((v[x + 5] / v[x + 6] - v[x + 7] / v[x + 8]) * 100)
    selected = max(range(4), key=lambda i: increase[i])
    checks.append(_entry(r, {"kind": "inventory_days_argmax_project", "selected": tickers[selected],
        "inventory_days_increase": dict(zip(tickers, increase)),
        "gross_margin_change_pp": dict(zip(tickers, margin_change))}, margin_change[selected]))

    metadata = {"", "chitieu", "item", "items", "stt", "no", "number",
                "ms", "maso", "code", "tm", "thuyetminh", "notes", "note"}
    bad_columns = []
    for qid in sorted(COMPLEX_IDS):
        for evidence in rows[qid]["evidence"]:
            header = re.sub(r"[^0-9a-z]+", "", _norm(evidence["col_name"]))
            if header in metadata:
                bad_columns.append({"id": qid, "evidence_id": evidence["evidence_id"],
                                    "col_name": evidence["col_name"]})
    duplicate_invariants = {
        "493_equals_552": rows[493]["output"]["value"] == rows[552]["output"]["value"],
        "554_equals_576": rows[554]["output"]["value"] == rows[576]["output"]["value"],
    }
    if bad_columns or not all(duplicate_invariants.values()):
        raise ValueError(f"semantic invariants failed: metadata={bad_columns}, duplicates={duplicate_invariants}")
    return {"schema_version": "p24_complex_semantic_audit_v1", "count": len(COMPLEX_IDS),
            "locked_opened": False, "metadata_value_columns": bad_columns,
            "duplicate_invariants": duplicate_invariants, "checks": checks}
