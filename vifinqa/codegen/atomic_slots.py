"""Conservative runtime planner for role-aware atomic metric slots.

The frozen retrieval route often describes a nested financial formula as one
long metric with role ``value``.  That starves the shortlist of numerator,
denominator, filter and projection cells.  This planner expands only a small
registry of well-defined financial formula families.  It does not calculate an
answer and does not mutate retrieval artifacts; Selection v1 remains unchanged.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..router.decompose import split_ratio_metric
from ..utils.viet_text import norm


MAX_ATOMIC_SLOTS = 24

_COUNT_COMPARATOR = re.compile(
    r"(?:lon hon|nho hon|cao hon|thap hon|nhieu hon|it hon|"
    r"khong nho hon|khong lon hon|bang|tu)\s+[-+]?\d"
)
_COUNT_METRIC = re.compile(
    r"(?=(?:\bghi nhan\b|\bco\b)\s+(?P<metric>.+?)\s+"
    r"(?=(?:lon hon|nho hon|cao hon|thap hon|nhieu hon|it hon|"
    r"khong nho hon|khong lon hon|bang|tu)\s+[-+]?\d))"
)
_PAREN_TICKER = re.compile(r"\(([A-Z][A-Z0-9]{1,4})\)")
_SUBJECT = re.compile(r"\b(?:cong ty me|ngan hang me|khoi ngan hang me)\b")
_NAMED_TARGET = re.compile(
    r"\btai\s+(?P<target>.+?)\s+cua\s+"
    r"(?:ctcp|cong ty|tong cong ty|tap doan|ngan hang)\b"
)
_TARGET_STOP = {
    "cong", "ty", "ctcp", "co", "phan", "tnhh", "tap", "doan",
    "tong", "ngan", "hang", "thuong", "mai",
}
_BRAND_TICKERS = {"mbbank": "MBB", "mb bank": "MBB",
                  "eximbank": "EIB", "exim bank": "EIB",
                  "sabeco": "SAB", "masan": "MSN",
                  "dai duong": "OGC", "vinamilk": "VNM",
                  "binh son": "BSR", "pvtrans": "PVT"}

# Mentions that require an atomic leaf in a nested question.  This registry is
# deliberately conservative: it is a fail-closed *coverage* check, not another
# parser. A question is not sent to the LLM if the deterministic planner sees
# a required financial quantity but did not materialise a corresponding leaf.
_METRIC_REQUIREMENTS: tuple[
        tuple[str, re.Pattern[str], tuple[tuple[str, ...], ...]], ...] = (
    ("operating_cash_flow",
     re.compile(r"(?:luu chuyen|dong) tien thuan tu hoat dong kinh doanh"),
     (("luu chuyen tien thuan tu hoat dong kinh doanh",
       "dong tien thuan tu hoat dong kinh doanh"),)),
    ("net_profit", re.compile(r"loi nhuan sau thue"),
     (("loi nhuan sau thue",),)),
    ("revenue", re.compile(r"doanh thu thuan"), (("doanh thu thuan",),)),
    ("current_assets", re.compile(r"tai san ngan han"),
     (("tai san ngan han",),)),
    ("current_debt", re.compile(r"no ngan han"), (("no ngan han",),)),
    ("total_assets", re.compile(r"tong (?:cong )?tai san"),
     (("tong tai san", "tong cong tai san"),)),
    ("borrowings", re.compile(r"no vay"), (("no vay",),)),
    ("liabilities", re.compile(r"no phai tra"), (("no phai tra",),)),
    ("equity", re.compile(r"von chu so huu"), (("von chu so huu",),)),
    ("interest_expense", re.compile(r"chi phi lai vay"),
     (("chi phi lai vay",),)),
    ("pretax_profit", re.compile(r"loi nhuan (?:ke toan )?truoc thue"),
     (("loi nhuan truoc thue", "loi nhuan ke toan truoc thue"),)),
    ("credit_provision", re.compile(r"chi phi du phong rui ro tin dung"),
     (("chi phi du phong rui ro tin dung",),)),
    ("preprovision_profit", re.compile(r"loi nhuan truoc du phong"),
     (("loi nhuan truoc du phong",),)),
    ("sga", re.compile(r"\bs\s*&\s*g\s*&\s*a\b|\bsg\s*&?\s*a\b"),
     (("chi phi ban hang",), ("chi phi quan ly doanh nghiep",),
      ("doanh thu thuan",))),
    ("pretax_margin", re.compile(r"bien loi nhuan truoc thue"),
     (("loi nhuan truoc thue", "loi nhuan ke toan truoc thue"),
      ("doanh thu thuan",))),
)


@dataclass(frozen=True)
class Component:
    metric: str
    role: str
    period_role: str = "same_period"
    year_offset: int = 0


@dataclass(frozen=True)
class Family:
    name: str
    pattern: re.Pattern[str]
    components: tuple[Component, ...]


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


FAMILIES: tuple[Family, ...] = (
    Family(
        "quick_ratio", _rx(r"(?:he so|kha nang) thanh toan nhanh"),
        (
            Component("tai san ngan han", "rank"),
            Component("hang ton kho", "rank"),
            Component("no ngan han", "denominator"),
        ),
    ),
    Family(
        "current_ratio", _rx(r"(?:he so|kha nang) thanh toan (?:hien hanh|ngan han)"),
        (Component("tai san ngan han", "numerator"),
         Component("no ngan han", "denominator")),
    ),
    Family(
        "operating_cashflow_to_current_debt",
        _rx(r"(?:he so )?(?:dong tien|luu chuyen tien).*hoat dong.*(?:tren|so voi).*no ngan han"),
        (Component("luu chuyen tien thuan tu hoat dong kinh doanh", "project", "next_period", 1),
         Component("no ngan han", "denominator", "next_period", 1)),
    ),
    Family(
        "cfo_margin", _rx(r"cfo margin|bien (?:dong tien|luu chuyen tien).*hoat dong"),
        (Component("luu chuyen tien thuan tu hoat dong kinh doanh", "numerator"),
         Component("doanh thu thuan", "denominator")),
    ),
    Family(
        "earnings_per_share",
        _rx(r"(?:loi nhuan|lai)\s+(?:co ban\s+)?tren moi co phieu|eps\b"),
        (Component("loi nhuan sau thue phan bo cho co dong", "numerator"),
         Component("so luong co phieu dang luu hanh", "denominator")),
    ),
    Family(
        "debt_to_assets",
        _rx(r"no vay.*(?:tren|so voi).*tong tai san"),
        (Component("no vay", "numerator"),
         Component("tong tai san", "denominator")),
    ),
    Family(
        "gross_margin", _rx(r"bien loi nhuan gop"),
        (Component("loi nhuan gop", "numerator"),
         Component("doanh thu thuan", "denominator")),
    ),
    Family(
        "net_margin", _rx(r"bien loi nhuan (?:rong|sau thue)|\bros\b"),
        (Component("loi nhuan sau thue", "numerator"),
         Component("doanh thu thuan", "denominator")),
    ),
    Family(
        "debt_to_equity", _rx(r"(?:ty so |he so )?d\s*/?\s*e\b|no phai tra.*(?:tren|so voi).*von chu so huu"),
        (Component("no phai tra", "filter"),
         Component("von chu so huu", "denominator")),
    ),
    Family(
        "interest_coverage", _rx(r"(?:he so |kha nang )?thanh toan lai vay|interest coverage"),
        (Component("loi nhuan truoc thue", "numerator"),
         Component("chi phi lai vay", "denominator")),
    ),
    Family(
        "effective_tax_rate", _rx(r"thue suat thuc te"),
        (Component("chi phi thue thu nhap doanh nghiep", "numerator"),
         Component("loi nhuan truoc thue", "denominator")),
    ),
    Family(
        "roe", _rx(r"\broe\b|ty suat loi nhuan.*von chu so huu"),
        (Component("loi nhuan sau thue", "numerator"),
         Component("von chu so huu", "denominator")),
    ),
    Family(
        "roa", _rx(r"\broa\b|ty suat loi nhuan.*tong tai san"),
        (Component("loi nhuan sau thue", "numerator"),
         Component("tong tai san", "denominator")),
    ),
    Family(
        "depreciation_admin_share",
        _rx(r"ty trong chi phi khau hao.*(?:trong|tren).*chi phi quan ly"),
        (Component("chi phi khau hao tai san co dinh", "numerator"),
         Component("chi phi quan ly doanh nghiep", "denominator")),
    ),
    Family(
        "segment_asset_share",
        _rx(r"ty trong tai san bo phan .*?(?:tren|trong) tong tai san"),
        (Component("tai san bo phan", "numerator"),
         Component("tong tai san", "denominator")),
    ),
)


def _combos(route: dict) -> list[dict]:
    plan_facts = list((route.get("plan") or {}).get("facts") or [])
    combos, seen = [], set()
    for fact in plan_facts:
        key = (str(fact.get("ticker") or ""), _year(fact.get("year")),
               str(fact.get("doc_type") or route.get("doc_type") or "consolidated"))
        if key in seen:
            continue
        seen.add(key)
        combos.append({"ticker": key[0], "year": key[1], "doc_type": key[2]})
    # Do not let a stale/incomplete plan erase explicit route entities or
    # periods. The stable dedupe below keeps the common case unchanged.
    tickers = list(route.get("tickers") or ([""] if not combos else []))
    years = list(route.get("years") or [None])
    for ticker in tickers:
        for year in years:
            key = (str(ticker or ""), _year(year),
                   str(route.get("doc_type") or "consolidated"))
            if key in seen:
                continue
            seen.add(key)
            combos.append({"ticker": key[0], "year": key[1],
                           "doc_type": key[2]})
    return combos


def plan_atomic_slots(question: str, route: dict) -> tuple[list[dict], dict]:
    """Return atomic facts plus deterministic planner trace."""
    q = norm(question)
    entity_guard = _entity_guard(question, q, route)
    repaired_metric, metric_repair = _repair_metric(q, route)
    semantic_anchors = _semantic_anchors(q)

    matches = [family for family in FAMILIES if family.pattern.search(q)]
    components: list[Component] = []
    component_keys = set()
    for family in matches:
        for component in family.components:
            key = (component.metric, component.role,
                   component.period_role, component.year_offset)
            if key not in component_keys:
                components.append(component)
                component_keys.add(key)

    plan = route.get("plan") or {}
    base_facts = list(plan.get("facts") or [])
    # A clean explicit A-tren-B decomposition is also a safe atomic expansion.
    if not components:
        metric = re.sub(
            r"^(?:gia tri )?trung binh\s+(?:ty le\s+)?", "", repaired_metric,
        ).strip()
        num, den = split_ratio_metric(metric)
        if num and den:
            components = [Component(num, "numerator"),
                          Component(den, "denominator")]
    planner_guard = _planner_completeness(q, components, repaired_metric)
    grounding_reasons = [
        item["reason"] for item in (entity_guard, planner_guard)
        if not item["ok"] and item["reason"]
    ]
    route_grounded = bool(entity_guard["ok"] and planner_guard["ok"])
    route_grounding_reason = "; ".join(grounding_reasons)

    slots = []
    if components:
        next_period_phrase = bool(re.search(r"nam sau nam|nam tiep theo", q))
        for combo in _combos(route):
            for component in components:
                offset = component.year_offset if next_period_phrase else 0
                year = combo["year"] + offset if combo["year"] is not None else None
                slots.append({
                    **combo,
                    "year": year,
                    "metric": component.metric,
                    "role": component.role,
                    "period_role": (component.period_role if offset else "same_period"),
                    "family": next((f.name for f in matches
                                    if component in f.components), "explicit_ratio"),
                    "semantic_anchors": [],
                    "route_grounded": route_grounded,
                    "route_grounding_reason": route_grounding_reason,
                })
    else:
        op = str(plan.get("op") or "lookup")
        output_type = str(route.get("output_type") or "number")
        fallback_role = (
            "filter" if op == "count" or output_type == "count"
            else "rank" if op == "ranking" or output_type == "year"
            else "value"
        )
        for fact in base_facts:
            slots.append({
                "ticker": str(fact.get("ticker") or ""),
                "year": _year(fact.get("year")),
                "doc_type": str(fact.get("doc_type") or route.get("doc_type") or "consolidated"),
                "metric": repaired_metric or str(fact.get("metric") or ""),
                "role": (str(fact.get("role") or fallback_role)
                         if str(fact.get("role") or "value") != "value"
                         else fallback_role),
                "period_role": _period_role(q),
                "family": "routed_fact",
                "semantic_anchors": semantic_anchors,
                "route_grounded": route_grounded,
                "route_grounding_reason": route_grounding_reason,
            })
        if not base_facts:
            for combo in _combos(route):
                slots.append({
                    **combo, "metric": repaired_metric, "role": fallback_role,
                    "period_role": _period_role(q), "family": "routed_fact",
                    "semantic_anchors": semantic_anchors,
                    "route_grounded": route_grounded,
                    "route_grounding_reason": route_grounding_reason,
                })
    # Stable dedupe, then fail-soft truncation. The trace makes truncation visible.
    deduped, seen = [], set()
    for slot in slots:
        key = (slot["ticker"], slot["year"], slot["doc_type"],
               norm(slot["metric"]), slot["role"], slot["period_role"])
        if key in seen or not slot["metric"]:
            continue
        seen.add(key)
        deduped.append(slot)
    truncated = len(deduped) > MAX_ATOMIC_SLOTS
    deduped = deduped[:MAX_ATOMIC_SLOTS]
    trace = {
        "policy": "atomic_metric_slots_v2_semantic",
        "families": [family.name for family in matches],
        "slot_count": len(deduped),
        "truncated": truncated,
        "roles": dict(Counter(slot["role"] for slot in deduped)),
        "period_roles": dict(Counter(slot["period_role"] for slot in deduped)),
        "metric_repair": metric_repair,
        "semantic_anchors": semantic_anchors,
        "entity_guard": entity_guard,
        "planner_guard": planner_guard,
    }
    return deduped, trace


def _repair_metric(q: str, route: dict) -> tuple[str, dict]:
    original = norm(str(route.get("metric_norm") or ""))
    if str(route.get("output_type") or "") == "count" and _COUNT_COMPARATOR.search(q):
        matches = list(_COUNT_METRIC.finditer(q))
        if matches:
            metric = matches[-1].group("metric").strip()
            metric = re.sub(r"^(?:so tien|gia tri|muc)\s+", "", metric).strip()
            if len(metric.split()) >= 2:
                return metric, {"applied": metric != original,
                                "source": "count_comparator", "from": original,
                                "to": metric}
    return original, {"applied": False, "source": "route", "from": original,
                      "to": original}


def _semantic_anchors(q: str) -> list[str]:
    match = _NAMED_TARGET.search(q)
    if not match:
        return []
    tokens = [token for token in match.group("target").split()
              if token not in _TARGET_STOP]
    if len(tokens) < 2 or len(tokens) > 10:
        return []
    return [" ".join(tokens)]


def _entity_guard(question: str, q: str, route: dict) -> dict:
    routed = {str(value).strip().upper() for value in route.get("tickers") or []
              if str(value).strip()}
    mentioned = {match.group(1).upper() for match in _PAREN_TICKER.finditer(question)}
    for alias, ticker in _BRAND_TICKERS.items():
        if re.search(rf"(?<![0-9a-z]){re.escape(alias)}(?![0-9a-z])", q):
            mentioned.add(ticker)
    reasons = []
    missing = sorted(mentioned - routed)
    if missing:
        reasons.append(f"explicit entities absent from route: {missing}")
    if ("tru di" in q or len(_SUBJECT.findall(q)) >= 2) and len(routed) < 2:
        reasons.append("multi-entity comparison has fewer than two routed tickers")
    return {"ok": not reasons, "routed_tickers": sorted(routed),
            "explicit_mentions": sorted(mentioned),
            "reason": "; ".join(reasons)}

def _planner_completeness(q: str, components: list[Component],
                          fallback_metric: str) -> dict:
    available = [norm(component.metric) for component in components
                 if norm(component.metric)]
    if not available and norm(fallback_metric):
        available = [norm(fallback_metric)]
    mentioned, missing = [], []
    for name, pattern, requirement_groups in _METRIC_REQUIREMENTS:
        if not pattern.search(q):
            continue
        mentioned.append(name)
        for alternatives in requirement_groups:
            if not any(
                    _metric_equivalent(metric, alias)
                    for metric in available for alias in alternatives):
                missing.append(f"{name}:{alternatives[0]}")
    missing = list(dict.fromkeys(missing))
    return {
        "ok": not missing,
        "mentioned": mentioned,
        "available": available,
        "missing": missing,
        "reason": (
            f"missing atomic leaves: {missing}" if missing else ""
        ),
    }


def _metric_equivalent(left: str, right: str) -> bool:
    left, right = norm(left), norm(right)
    return bool(left and right and (left in right or right in left))


def _year(value) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 1900 <= value <= 2100 else None


def _period_role(q: str) -> str:
    if re.search(r"cuoi nam|cuoi ky|31\s*/\s*12|31\s+thang\s+12", q):
        return "ending"
    if re.search(
            r"dau nam|dau ky|(?<!\d)0?1\s*/\s*0?1(?!\d)|1\s+thang\s+1", q):
        return "beginning"
    if re.search(r"nam truoc|ky truoc", q):
        return "prior"
    if re.search(r"nam nay|ky nay", q):
        return "current"
    return "same_period"

