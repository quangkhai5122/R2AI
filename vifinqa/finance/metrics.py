"""Canonical Vietnamese financial metrics used across the pipeline.

This is deliberately a thin schema layer over the existing table store. It
normalizes common VAS line-item names, preserves safe aliases, and describes
derived metrics through their component line items. Retrieval, row linking and
formula execution can therefore share one definition without rewriting the
underlying corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from ..utils.viet_text import norm


@dataclass(frozen=True)
class CanonicalMetric:
    key: str
    label: str
    aliases: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    statement: str = "other"
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    qualifier_phrases: tuple[str, ...] = ()
    components: tuple[str, ...] = ()

    @property
    def variants(self) -> tuple[str, ...]:
        return _dedupe((self.label, *self.aliases))

    @property
    def is_derived(self) -> bool:
        return bool(self.components)


@dataclass(frozen=True)
class MetricMatch:
    metric: CanonicalMetric
    alias: str
    start: int
    end: int


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen, out = set(), []
    for value in values:
        value = norm(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _metric(key: str, label: str, aliases=(), codes=(), statement="other",
            required=(), forbidden=(), qualifiers=(),
            components=()) -> CanonicalMetric:
    return CanonicalMetric(
        key=key,
        label=norm(label),
        aliases=_dedupe(norm(a) for a in aliases),
        codes=tuple(str(c) for c in codes),
        statement=statement,
        required_phrases=_dedupe(norm(a) for a in required),
        forbidden_phrases=_dedupe(norm(a) for a in forbidden),
        qualifier_phrases=_dedupe(norm(a) for a in qualifiers),
        components=tuple(components),
    )


_LINE_ITEMS = [
    # B01-DN: balance sheet
    _metric("current_assets", "tai san ngan han",
            ("tong tai san ngan han", "tai san luu dong va dau tu ngan han"),
            ("100",), "balance_sheet", ("tai san ngan han",)),
    _metric("cash", "tien va cac khoan tuong duong tien",
            ("tien va tuong duong tien", "tien mat va cac khoan tuong duong tien"),
            ("110",), "balance_sheet", ("tien", "tuong duong tien")),
    _metric("short_term_investments", "dau tu tai chinh ngan han",
            ("cac khoan dau tu tai chinh ngan han", "dau tu nam giu den ngay dao han ngan han"),
            ("120", "123"), "balance_sheet", ("dau tu", "ngan han")),
    _metric("short_term_receivables", "cac khoan phai thu ngan han",
            ("phai thu ngan han", "tong cac khoan phai thu ngan han"),
            ("130",), "balance_sheet", ("phai thu", "ngan han")),
    _metric("trade_receivables_short_term", "phai thu ngan han cua khach hang",
            ("phai thu khach hang ngan han", "phai thu cua khach hang ngan han"),
            ("131",), "balance_sheet", ("phai thu", "khach hang", "ngan han")),
    _metric("supplier_prepayments_short_term", "tra truoc cho nguoi ban ngan han",
            ("tra truoc nguoi ban ngan han", "tien tra truoc cho nguoi ban ngan han"),
            ("132",), "balance_sheet", ("tra truoc", "nguoi ban", "ngan han")),
    _metric("inventory", "hang ton kho", ("hang ton kho rong", "ton kho"),
            ("140", "141"), "balance_sheet", ("hang ton kho", "ton kho")),
    _metric("long_term_assets", "tai san dai han", ("tong tai san dai han",),
            ("200",), "balance_sheet", ("tai san dai han",)),
    _metric("fixed_assets", "tai san co dinh", ("tong tai san co dinh",),
            ("220",), "balance_sheet", ("tai san co dinh",), ("khau hao",),
            ("khau hao",)),
    _metric("tangible_fixed_assets", "tai san co dinh huu hinh",
            ("tcdn huu hinh",), ("221",), "balance_sheet",
            ("tai san co dinh huu hinh",)),
    _metric("construction_in_progress", "chi phi xay dung co ban do dang",
            ("xay dung co ban do dang", "chi phi xay dung co ban dang do"),
            ("242",), "balance_sheet", ("xay dung", "do dang")),
    _metric("total_assets", "tong tai san", ("tong cong tai san",),
            ("270",), "balance_sheet", ("tong tai san", "tong cong tai san")),
    _metric("liabilities", "no phai tra", ("tong no phai tra", "tong cong no phai tra"),
            ("300",), "balance_sheet", ("no phai tra",),
            ("no ngan han", "no dai han")),
    _metric("current_liabilities", "no ngan han",
            ("tong no ngan han", "no phai tra ngan han"),
            ("310",), "balance_sheet", ("no ngan han", "no phai tra ngan han"),
            qualifiers=("ben lien quan",)),
    _metric("trade_payables_short_term", "phai tra nguoi ban ngan han",
            ("phai tra ngan han cho nguoi ban", "phai tra cho nguoi ban ngan han"),
            ("311",), "balance_sheet", ("phai tra", "nguoi ban", "ngan han")),
    _metric("short_term_borrowings", "vay va no thue tai chinh ngan han",
            ("vay ngan han", "vay va no ngan han"),
            ("320",), "balance_sheet", ("vay", "ngan han")),
    _metric("long_term_liabilities", "no dai han", ("tong no dai han",),
            ("330",), "balance_sheet", ("no dai han",)),
    _metric("equity", "von chu so huu",
            ("tong von chu so huu", "nguon von chu so huu"),
            ("400", "410"), "balance_sheet", ("von chu so huu",),
            ("von gop cua chu so huu",)),
    _metric("contributed_capital", "von gop cua chu so huu",
            ("von dau tu cua chu so huu", "von gop chu so huu"),
            ("411",), "balance_sheet", ("von", "chu so huu")),
    _metric("retained_earnings", "loi nhuan sau thue chua phan phoi",
            ("lnst chua phan phoi", "loi nhuan chua phan phoi"),
            ("421",), "balance_sheet", ("loi nhuan", "chua phan phoi")),

    # B02-DN: income statement
    _metric("sales_revenue", "doanh thu ban hang va cung cap dich vu",
            ("doanh thu ban hang",), ("01", "1"), "income_statement",
            ("doanh thu", "ban hang"), ("doanh thu thuan",)),
    _metric("revenue_deductions", "cac khoan giam tru doanh thu",
            ("giam tru doanh thu",), ("02", "2"), "income_statement",
            ("giam tru", "doanh thu")),
    _metric("net_revenue", "doanh thu thuan",
            ("doanh thu thuan ve ban hang va cung cap dich vu",),
            ("10",), "income_statement", ("doanh thu thuan",)),
    _metric("cost_of_goods_sold", "gia von hang ban", ("gia von",),
            ("11",), "income_statement", ("gia von",)),
    _metric("gross_profit", "loi nhuan gop",
            ("loi nhuan gop ve ban hang va cung cap dich vu",),
            ("20",), "income_statement", ("loi nhuan gop",)),
    _metric("financial_income", "doanh thu hoat dong tai chinh",
            ("doanh thu tai chinh",), ("21",), "income_statement",
            ("doanh thu", "tai chinh")),
    _metric("financial_expense", "chi phi tai chinh", ("chi phi hoat dong tai chinh",),
            ("22",), "income_statement", ("chi phi", "tai chinh")),
    _metric("interest_expense", "chi phi lai vay", ("lai tien vay",),
            ("23",), "income_statement", ("chi phi lai vay", "lai tien vay"),
            ("da tra", "thuc tra", "thanh toan")),
    _metric("selling_expense", "chi phi ban hang", (), ("25",),
            "income_statement", ("chi phi ban hang",)),
    _metric("administrative_expense", "chi phi quan ly doanh nghiep", (), ("26",),
            "income_statement", ("chi phi quan ly doanh nghiep",)),
    _metric("operating_profit", "loi nhuan thuan tu hoat dong kinh doanh",
            ("loi nhuan thuan hoat dong kinh doanh",), ("30",),
            "income_statement", ("loi nhuan thuan", "hoat dong kinh doanh")),
    _metric("other_income", "thu nhap khac", (), ("31",),
            "income_statement", ("thu nhap khac",)),
    _metric("other_expense", "chi phi khac", (), ("32",),
            "income_statement", ("chi phi khac",)),
    _metric("other_profit", "loi nhuan khac", (), ("40",),
            "income_statement", ("loi nhuan khac",)),
    _metric("pretax_profit", "loi nhuan truoc thue",
            ("tong loi nhuan ke toan truoc thue", "loi nhuan ke toan truoc thue", "lntt"),
            ("50",), "income_statement",
            ("loi nhuan truoc thue", "loi nhuan ke toan truoc thue")),
    _metric("current_income_tax", "chi phi thue thu nhap doanh nghiep hien hanh",
            ("chi phi thue tndn hien hanh", "thue thu nhap doanh nghiep hien hanh"),
            ("51",), "income_statement", ("thue", "thu nhap", "hien hanh"),
            ("ca nhan", "hoan lai")),
    _metric("deferred_income_tax", "chi phi thue thu nhap doanh nghiep hoan lai",
            ("chi phi thue tndn hoan lai", "thue thu nhap hoan lai"),
            ("52",), "income_statement", ("thue", "thu nhap", "hoan lai"),
            ("ca nhan", "hien hanh")),
    _metric("net_profit", "loi nhuan sau thue",
            ("loi nhuan sau thue thu nhap doanh nghiep", "loi nhuan thuan sau thue", "lnst"),
            ("60",), "income_statement",
            ("loi nhuan sau thue", "loi nhuan thuan sau thue"),
            ("chua phan phoi", "thuoc ve co dong", "cua co dong", "phan bo cho"),
            ("chua phan phoi", "thuoc ve co dong", "cua co dong", "phan bo cho")),
    _metric("basic_eps", "lai co ban tren co phieu",
            ("lai co ban moi co phieu", "eps co ban"),
            ("70",), "income_statement", ("lai co ban", "co phieu")),

    # B03-DN: cash-flow statement
    _metric("cfo", "luu chuyen tien thuan tu hoat dong kinh doanh",
            ("dong tien thuan tu hoat dong kinh doanh", "lctt tu hoat dong kinh doanh"),
            ("20",), "cash_flow",
            ("luu chuyen tien thuan tu hoat dong kinh doanh",
             "dong tien thuan tu hoat dong kinh doanh")),
    _metric("cfi", "luu chuyen tien thuan tu hoat dong dau tu",
            ("dong tien thuan tu hoat dong dau tu",), ("30",), "cash_flow",
            ("tien thuan", "hoat dong dau tu")),
    _metric("cff", "luu chuyen tien thuan tu hoat dong tai chinh",
            ("dong tien thuan tu hoat dong tai chinh",), ("40",), "cash_flow",
            ("tien thuan", "hoat dong tai chinh")),
    _metric("net_cash_change", "luu chuyen tien thuan trong ky",
            ("tien va tuong duong tien tang giam trong ky", "luu chuyen tien thuan trong nam"),
            ("50",), "cash_flow", ("tien", "trong ky")),
    _metric("opening_cash", "tien va tuong duong tien dau ky",
            ("tien va tuong duong tien dau nam",), ("60",), "cash_flow",
            ("tien", "tuong duong tien", "dau")),
    _metric("closing_cash", "tien va tuong duong tien cuoi ky",
            ("tien va tuong duong tien cuoi nam",), ("70",), "cash_flow",
            ("tien", "tuong duong tien", "cuoi")),
]


_DERIVED = [
    _metric("quick_ratio", "he so thanh toan nhanh",
            ("ty so thanh toan nhanh", "thanh toan nhanh"),
            components=("current_assets", "inventory", "current_liabilities")),
    _metric("current_ratio", "he so thanh toan hien hanh",
            ("ty so thanh toan hien hanh", "tai san ngan han gap bao nhieu lan no ngan han"),
            components=("current_assets", "current_liabilities")),
    _metric("debt_equity", "no phai tra tren von chu so huu",
            ("he so no phai tra tren von chu so huu", "ty le no phai tra tren von chu so huu",
             "no phai tra chia cho von chu so huu", "d/e"),
            components=("liabilities", "equity")),
    _metric("debt_assets", "no phai tra tren tong tai san",
            ("no phai tra chia cho tong tai san", "he so no tren tai san", "debt/assets"),
            components=("liabilities", "total_assets")),
    _metric("gross_margin", "bien loi nhuan gop",
            ("bien gop", "loi nhuan gop tren doanh thu thuan"),
            components=("gross_profit", "net_revenue")),
    _metric("net_margin", "bien loi nhuan rong",
            ("bien loi nhuan sau thue", "loi nhuan sau thue tren doanh thu thuan"),
            components=("net_profit", "net_revenue")),
    _metric("cfo_margin", "cfo margin",
            ("cfo tren doanh thu", "dong tien kinh doanh tren doanh thu",
             "dong tien kinh doanh tren doanh thu thuan",
             "luu chuyen tien thuan tu hoat dong kinh doanh tren doanh thu thuan"),
            components=("cfo", "net_revenue")),
    _metric("cfo_net_profit", "cfo tren lnst",
            ("cfo/lnst", "cfo tren loi nhuan sau thue", "he so chuyen doi loi nhuan",
             "dong tien kinh doanh tren loi nhuan sau thue",
             "luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan sau thue"),
            components=("cfo", "net_profit")),
    _metric("cfo_current_liabilities", "cfo tren no ngan han",
            ("cfo/no ngan han", "dong tien hoat dong tren no ngan han",
             "dong tien kinh doanh tren no ngan han"),
            components=("cfo", "current_liabilities")),
    _metric("roa", "roa", ("loi nhuan sau thue tren tong tai san",),
            components=("net_profit", "total_assets")),
    _metric("roe", "roe", ("loi nhuan sau thue tren von chu so huu",),
            components=("net_profit", "equity")),
    _metric("working_capital", "von luu dong rong", (),
            components=("current_assets", "current_liabilities")),
    _metric("inventory_assets", "ty trong hang ton kho",
            ("hang ton kho tren tong tai san", "hang ton kho chia cho tong tai san"),
            components=("inventory", "total_assets")),
    _metric("sga_expense", "tong chi phi ban hang va chi phi quan ly doanh nghiep",
            ("chi phi ban hang va chi phi quan ly doanh nghiep", "sg a", "sga"),
            components=("selling_expense", "administrative_expense")),
    _metric("sga_intensity", "ty trong chi phi ban hang va quan ly doanh nghiep",
            ("chi phi ban hang va quan ly doanh nghiep tren doanh thu thuan",),
            components=("selling_expense", "administrative_expense", "net_revenue")),
    _metric("interest_coverage", "he so kha nang thanh toan lai vay",
            ("kha nang thanh toan lai vay", "loi nhuan truoc lai vay va thue"),
            components=("pretax_profit", "interest_expense")),
    _metric("fixed_asset_turnover", "vong quay tai san co dinh", (),
            components=("net_revenue", "fixed_assets")),
    _metric("total_asset_turnover", "vong quay tong tai san", (),
            components=("net_revenue", "total_assets")),
]


METRICS: dict[str, CanonicalMetric] = {
    metric.key: metric for metric in (*_LINE_ITEMS, *_DERIVED)
}

_ALIAS_TO_METRICS: dict[str, tuple[CanonicalMetric, ...]] = {}
for _registered_metric in METRICS.values():
    for _registered_alias in _registered_metric.variants:
        _ALIAS_TO_METRICS[_registered_alias] = (
            *_ALIAS_TO_METRICS.get(_registered_alias, ()), _registered_metric)
_ALIAS_RE = re.compile(
    r"(?<![0-9a-z])(?:"
    + "|".join(re.escape(alias) for alias in
               sorted(_ALIAS_TO_METRICS, key=len, reverse=True))
    + r")(?![0-9a-z])"
)


def get_metric(key: str) -> CanonicalMetric:
    return METRICS[key]


def find_metrics(text: str, include_derived: bool = True) -> list[MetricMatch]:
    """Find non-overlapping canonical concepts, preferring longest aliases."""
    text_norm = norm(text)
    if not text_norm:
        return []
    return list(_find_metrics_norm(text_norm, include_derived))


@lru_cache(maxsize=65536)
def _find_metrics_norm(text_norm: str,
                       include_derived: bool) -> tuple[MetricMatch, ...]:
    matches = []
    for found in _ALIAS_RE.finditer(text_norm):
        alias = found.group(0)
        choices = [metric for metric in _ALIAS_TO_METRICS[alias]
                   if include_derived or not metric.is_derived]
        if not choices:
            continue
        metric = min(choices, key=lambda item: (item.is_derived, item.key))
        matches.append(MetricMatch(
            metric, alias, found.start(), found.end()))
    return tuple(matches)


def metric_keys(texts: Iterable[str], expand_derived: bool = True) -> list[str]:
    normalized = _dedupe(texts)
    return list(_metric_keys_cached(normalized, expand_derived))


@lru_cache(maxsize=32768)
def _metric_keys_cached(texts: tuple[str, ...],
                        expand_derived: bool) -> tuple[str, ...]:
    seen, out = set(), []
    for text in texts:
        for match in find_metrics(text):
            keys = (match.metric.components if match.metric.is_derived and expand_derived
                    else (match.metric.key,))
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return tuple(out)


def expand_metric_variants(phrases: Iterable[str], question: str = "",
                           aliases_per_metric: int = 3) -> list[str]:
    """Append safe canonical aliases and derived-metric component names."""
    originals = _dedupe(norm(p) for p in phrases)
    return list(_expand_metric_variants_cached(
        originals, norm(question), int(aliases_per_metric)))


@lru_cache(maxsize=32768)
def _expand_metric_variants_cached(originals: tuple[str, ...], question: str,
                                   aliases_per_metric: int) -> tuple[str, ...]:
    search = [*originals]
    if question:
        search.append(norm(question))
    keys = metric_keys(search, expand_derived=True)
    expanded = list(originals)
    asked_text = " ".join(search)
    for key in keys:
        metric = METRICS[key]
        # A qualified note metric must not collapse to its aggregate line item.
        # For example, "no ngan han voi ben lien quan" is not code 310.
        if any(phrase in asked_text for phrase in metric.qualifier_phrases):
            continue
        expanded.extend(metric.variants[:max(1, aliases_per_metric)])
    return _dedupe(expanded)


def code_expectation(phrases: Iterable[str], label: str = "") -> tuple[set[str], bool]:
    """Return expected VAS codes and whether the label is a known mismatch."""
    phrase_list = _dedupe(phrases)
    asked_text, asked_frozen = _asked_metric_context(phrase_list)
    asked = set(asked_frozen)
    if not asked:
        return set(), False
    if label:
        labelled = set(metric_keys([label], expand_derived=False))
        common = asked & labelled
        if common:
            label_norm = norm(label)
            forbidden_label = any(
                forbidden in label_norm and forbidden not in asked_text
                for key in common for forbidden in METRICS[key].forbidden_phrases
            )
            qualifier_mismatch = any(
                (qualifier in label_norm) != (qualifier in asked_text)
                for key in common for qualifier in METRICS[key].qualifier_phrases
            )
            if forbidden_label or qualifier_mismatch:
                return set(), True
            codes = {code for key in common for code in METRICS[key].codes}
            return codes, False
        if labelled:
            return set(), True
    codes = {code for key in asked for code in METRICS[key].codes}
    return codes, False


@lru_cache(maxsize=32768)
def _asked_metric_context(phrases: tuple[str, ...]) -> tuple[str, frozenset[str]]:
    return " ".join(phrases), frozenset(metric_keys(phrases, expand_derived=True))
