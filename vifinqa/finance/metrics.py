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
class MetricQualifiers:
    """Semantic dimensions that distinguish otherwise similar line items.

    Empty means unspecified.  Values intentionally use a tiny controlled
    vocabulary so router, retrieval and codegen can share the same contract.
    """

    stock_flow: str = ""       # stock | flow
    gross_net: str = ""        # gross | net
    maturity: str = ""         # short | medium | long
    period: str = ""           # opening | closing
    granularity: str = ""      # aggregate | detail
    sign: str = ""             # signed | absolute

    def to_dict(self) -> dict[str, str]:
        return {
            "stock_flow": self.stock_flow,
            "gross_net": self.gross_net,
            "maturity": self.maturity,
            "period": self.period,
            "granularity": self.granularity,
            "sign": self.sign,
        }


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
    component_year_offsets: tuple[int, ...] = ()
    row_aliases: tuple[str, ...] = ()
    context_phrases: tuple[str, ...] = ()
    qualifiers: MetricQualifiers = MetricQualifiers()

    @property
    def variants(self) -> tuple[str, ...]:
        return _dedupe((self.label, *self.aliases))

    @property
    def row_variants(self) -> tuple[str, ...]:
        """Canonical query phrases plus terse labels used inside note tables."""
        return _dedupe((*self.variants, *self.row_aliases))

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
            components=(), *, row_aliases=(), context=(),
            component_year_offsets=(),
            stock_flow="", gross_net="", maturity="",
            granularity="", sign="") -> CanonicalMetric:
    if not stock_flow:
        if statement == "balance_sheet":
            stock_flow = "stock"
        elif statement in ("income_statement", "cash_flow"):
            stock_flow = "flow"
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
        component_year_offsets=tuple(int(value) for value in component_year_offsets),
        row_aliases=_dedupe(norm(a) for a in row_aliases),
        context_phrases=_dedupe(norm(a) for a in context),
        qualifiers=MetricQualifiers(
            stock_flow=stock_flow,
            gross_net=gross_net,
            maturity=maturity,
            granularity=granularity,
            sign=sign,
        ),
    )


_LINE_ITEMS = [
    # B01-DN: balance sheet
    _metric("current_assets", "tai san ngan han",
            ("tong tai san ngan han", "tai san luu dong va dau tu ngan han"),
            ("100",), "balance_sheet", ("tai san ngan han",),
            maturity="short", granularity="aggregate"),
    _metric("cash", "tien va cac khoan tuong duong tien",
            ("tien va tuong duong tien", "tien mat va cac khoan tuong duong tien"),
            ("110",), "balance_sheet", ("tien", "tuong duong tien")),
    _metric("short_term_investments", "dau tu tai chinh ngan han",
            ("cac khoan dau tu tai chinh ngan han", "dau tu nam giu den ngay dao han ngan han"),
            ("120", "123"), "balance_sheet", ("dau tu", "ngan han"),
            maturity="short"),
    _metric("short_term_receivables", "cac khoan phai thu ngan han",
            ("phai thu ngan han", "tong cac khoan phai thu ngan han"),
            ("130",), "balance_sheet", ("phai thu", "ngan han"),
            maturity="short", granularity="aggregate"),
    _metric("trade_receivables_short_term", "phai thu ngan han cua khach hang",
            ("phai thu khach hang ngan han", "phai thu cua khach hang ngan han"),
            ("131",), "balance_sheet", ("phai thu", "khach hang", "ngan han"),
            maturity="short", granularity="detail"),
    _metric("supplier_prepayments_short_term", "tra truoc cho nguoi ban ngan han",
            ("tra truoc nguoi ban ngan han", "tien tra truoc cho nguoi ban ngan han"),
            ("132",), "balance_sheet", ("tra truoc", "nguoi ban", "ngan han"),
            context=("bang can doi ke toan",), maturity="short",
            granularity="detail"),
    _metric("inventory", "hang ton kho", ("hang ton kho rong", "ton kho"),
            ("140", "141"), "balance_sheet", ("hang ton kho", "ton kho")),
    _metric("long_term_assets", "tai san dai han", ("tong tai san dai han",),
            ("200",), "balance_sheet", ("tai san dai han",),
            maturity="long", granularity="aggregate"),
    _metric("fixed_assets", "tai san co dinh", ("tong tai san co dinh",),
            ("220",), "balance_sheet", ("tai san co dinh",),
            ("khau hao", "tai san co dinh huu hinh", "tai san co dinh vo hinh",
             "tscd huu hinh", "tscd vo hinh"),
            ("khau hao",)),
    _metric("tangible_fixed_assets", "tai san co dinh huu hinh",
            ("tcdn huu hinh",), ("221",), "balance_sheet",
            ("tai san co dinh huu hinh",)),
    _metric("intangible_fixed_assets", "tai san co dinh vo hinh",
            ("tscd vo hinh", "gia tri con lai cua tai san co dinh vo hinh",
             "gia tri con lai tai san co dinh vo hinh"),
            ("227",), "balance_sheet", ("tai san co dinh vo hinh",),
            gross_net="net"),
    _metric("construction_in_progress", "chi phi xay dung co ban do dang",
            ("xay dung co ban do dang", "chi phi xay dung co ban dang do"),
            ("242",), "balance_sheet", ("xay dung", "do dang"),
            context=("bang can doi ke toan",)),
    _metric("total_assets", "tong tai san", ("tong cong tai san",),
            ("270",), "balance_sheet", ("tong tai san", "tong cong tai san"),
            granularity="aggregate"),
    _metric("liabilities", "no phai tra", ("tong no phai tra", "tong cong no phai tra"),
            ("300",), "balance_sheet", ("no phai tra",),
            ("no ngan han", "no dai han"), granularity="aggregate"),
    _metric("current_liabilities", "no ngan han",
            ("tong no ngan han", "no phai tra ngan han"),
            ("310",), "balance_sheet", ("no ngan han", "no phai tra ngan han"),
            qualifiers=("ben lien quan",), maturity="short", granularity="aggregate"),
    _metric("trade_payables_short_term", "phai tra nguoi ban ngan han",
            ("phai tra ngan han cho nguoi ban", "phai tra cho nguoi ban ngan han"),
            ("311",), "balance_sheet", ("phai tra", "nguoi ban", "ngan han"),
            maturity="short", granularity="detail"),
    _metric("short_term_borrowings", "vay va no thue tai chinh ngan han",
            ("vay ngan han", "vay va no ngan han", "no vay ngan han"),
            ("320",), "balance_sheet", ("vay", "ngan han"),
            maturity="short", granularity="aggregate"),
    _metric("long_term_liabilities", "no dai han", ("tong no dai han",),
            ("330",), "balance_sheet", ("no dai han",),
            maturity="long", granularity="aggregate"),
    _metric("equity", "von chu so huu",
            ("tong von chu so huu", "nguon von chu so huu"),
            ("400",), "balance_sheet", ("von chu so huu",),
            ("von gop cua chu so huu",), context=("bang can doi ke toan",),
            granularity="aggregate"),
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
            ("10",), "income_statement", ("doanh thu thuan",),
            context=("bao cao ket qua hoat dong kinh doanh",
                     "bao cao ket qua kinh doanh"), gross_net="net"),
    _metric("cost_of_goods_sold", "gia von hang ban", ("gia von",),
            ("11",), "income_statement", ("gia von",), sign="absolute"),
    _metric("gross_profit", "loi nhuan gop",
            ("loi nhuan gop ve ban hang va cung cap dich vu",),
            ("20",), "income_statement", ("loi nhuan gop",)),
    _metric("financial_income", "doanh thu hoat dong tai chinh",
            ("doanh thu tai chinh",), ("21",), "income_statement",
            ("doanh thu", "tai chinh")),
    _metric("financial_expense", "chi phi tai chinh", ("chi phi hoat dong tai chinh",),
            ("22",), "income_statement", ("chi phi", "tai chinh"),
            sign="absolute"),
    _metric("interest_expense", "chi phi lai vay", ("lai tien vay", "lai vay"),
            ("23",), "income_statement", ("chi phi lai vay", "lai tien vay"),
            ("da tra", "thuc tra", "thanh toan"),
            row_aliases=("trong do chi phi lai vay",), sign="absolute"),
    _metric("selling_expense", "chi phi ban hang", (), ("25",),
            "income_statement", ("chi phi ban hang",), sign="absolute"),
    _metric("administrative_expense", "chi phi quan ly doanh nghiep", (), ("26",),
            "income_statement", ("chi phi quan ly doanh nghiep",), sign="absolute"),
    _metric("operating_profit", "loi nhuan thuan tu hoat dong kinh doanh",
            ("loi nhuan thuan hoat dong kinh doanh",), ("30",),
            "income_statement", ("loi nhuan thuan", "hoat dong kinh doanh")),
    _metric("other_income", "thu nhap khac", (), ("31",),
            "income_statement", ("thu nhap khac",)),
    _metric("other_expense", "chi phi khac", (), ("32",),
            "income_statement", ("chi phi khac",), sign="absolute"),
    _metric("other_profit", "loi nhuan khac", (), ("40",),
            "income_statement", ("loi nhuan khac",),
            context=("bao cao ket qua hoat dong kinh doanh",
                     "bao cao ket qua kinh doanh")),
    _metric("pretax_profit", "loi nhuan truoc thue",
            ("tong loi nhuan ke toan truoc thue", "loi nhuan ke toan truoc thue", "lntt"),
            ("50",), "income_statement",
            ("loi nhuan truoc thue", "loi nhuan ke toan truoc thue")),
    _metric("current_income_tax", "chi phi thue thu nhap doanh nghiep hien hanh",
            ("chi phi thue tndn hien hanh", "thue thu nhap doanh nghiep hien hanh",
             "chi phi thue thu nhap hien hanh", "chi phi thue hien hanh"),
            ("51",), "income_statement", ("thue", "thu nhap", "hien hanh"),
            ("ca nhan", "hoan lai"), sign="absolute"),
    _metric("deferred_income_tax", "chi phi thue thu nhap doanh nghiep hoan lai",
            ("chi phi thue tndn hoan lai", "thue thu nhap hoan lai"),
            ("52",), "income_statement", ("thue", "thu nhap", "hoan lai"),
            ("ca nhan", "hien hanh"), sign="absolute"),
    _metric("net_profit", "loi nhuan sau thue",
            ("loi nhuan sau thue thu nhap doanh nghiep", "loi nhuan thuan sau thue",
             "loi nhuan ke toan sau thue tndn", "loi nhuan thuan trong nam", "lnst"),
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


# Sector-specific line items.  Bank statements reuse many ordinary Vietnamese
# words ("tien gui", "vay", "du phong") for materially different concepts.
# Keeping the parent and child rows as separate keys lets schema linking reject
# a lexical superset such as "Tien gui va vay cac TCTD khac" when the question
# asks only for "Vay cac TCTD khac".
_BANK_LINE_ITEMS = [
    _metric("bank_cash", "tien mat vang bac da quy",
            ("tien mat", "tien mat vang bac"), statement="balance_sheet",
            granularity="aggregate"),
    _metric("central_bank_deposits", "tien gui tai ngan hang nha nuoc",
            ("tien gui tai nhnn", "tien gui tai nhnnvn"),
            statement="balance_sheet", granularity="detail"),
    _metric("interbank_assets_total", "tien gui va cho vay cac tctd khac",
            ("tien gui va cho vay cac to chuc tin dung khac",),
            statement="balance_sheet", granularity="aggregate"),
    _metric("interbank_deposits_asset", "tien gui tai cac tctd khac",
            ("tien gui cac tctd khac", "tien gui tai cac to chuc tin dung khac"),
            statement="balance_sheet", granularity="detail"),
    _metric("interbank_loans_asset", "cho vay cac tctd khac",
            ("cho vay cac to chuc tin dung khac",),
            statement="balance_sheet", granularity="detail"),
    _metric("customer_loans", "cho vay khach hang",
            ("du no cho vay khach hang", "tong du no cho vay", "du no cho vay"),
            statement="balance_sheet", gross_net="gross",
            granularity="aggregate"),
    _metric("customer_loan_provision_balance",
            "du phong rui ro cho vay khach hang",
            ("so du du phong rui ro cho vay khach hang",
             "du phong cho vay khach hang"),
            statement="balance_sheet", granularity="aggregate",
            sign="absolute"),
    _metric("customer_deposits", "tien gui cua khach hang",
            ("tien gui khach hang",), statement="balance_sheet",
            granularity="aggregate"),
    _metric("interbank_funding_total", "tien gui va vay cac tctd khac",
            ("tien gui va vay cac to chuc tin dung khac",),
            statement="balance_sheet", granularity="aggregate"),
    _metric("interbank_deposits_liability", "tien gui cua cac tctd khac",
            ("tien gui cac to chuc tin dung khac",),
            statement="balance_sheet", granularity="detail"),
    _metric("interbank_borrowings", "vay cac tctd khac",
            ("vay cac to chuc tin dung khac", "tien vay cac tctd khac"),
            statement="balance_sheet", granularity="detail"),
    _metric("valuable_papers_issued", "phat hanh giay to co gia",
            ("giay to co gia phat hanh",), statement="balance_sheet",
            granularity="aggregate"),
    _metric("available_for_sale_securities",
            "chung khoan dau tu san sang de ban",
            ("chung khoan san sang de ban",), statement="balance_sheet"),
    _metric("debt_securities", "chung khoan no",
            ("chung khoan no dau tu",), statement="balance_sheet"),
    _metric("term_deposits", "tien gui co ky han",
            ("tien gui ky han",), statement="balance_sheet"),
    _metric("savings_deposits", "tien gui tiet kiem", (),
            statement="balance_sheet", granularity="detail"),
    _metric("government_bonds", "trai phieu chinh phu", (),
            statement="balance_sheet", granularity="detail"),
    _metric("vamc_special_bonds", "trai phieu dac biet do vamc phat hanh",
            ("trai phieu dac biet vamc",), statement="balance_sheet",
            granularity="detail"),
    _metric("bank_interest_income", "thu nhap lai va cac khoan thu nhap tuong tu",
            ("thu nhap lai",), statement="income_statement"),
    _metric("bank_interest_expense", "chi phi lai va cac chi phi tuong tu",
            ("chi phi lai ngan hang",), statement="income_statement",
            sign="absolute"),
    _metric("net_interest_income", "thu nhap lai thuan",
            ("lai thuan tu hoat dong tin dung",),
            statement="income_statement", gross_net="net"),
    _metric("bank_service_income", "thu nhap tu hoat dong dich vu",
            ("thu nhap dich vu",), statement="income_statement"),
    _metric("bank_service_expense", "chi phi hoat dong dich vu",
            ("chi phi dich vu ngan hang",), statement="income_statement",
            sign="absolute"),
    _metric("net_service_income", "lai thuan tu hoat dong dich vu",
            ("lai lo thuan tu hoat dong dich vu",
             "ket qua thuan tu hoat dong dich vu"),
            statement="income_statement", gross_net="net"),
    _metric("bank_operating_income", "tong thu nhap hoat dong",
            ("tong thu nhap hoat dong ngan hang",),
            statement="income_statement", granularity="aggregate"),
    _metric("bank_operating_expense", "chi phi hoat dong",
            ("tong chi phi hoat dong",), statement="income_statement",
            granularity="aggregate", sign="absolute"),
    _metric("pre_provision_operating_profit",
            "loi nhuan thuan tu hoat dong kinh doanh truoc chi phi du phong rui ro tin dung",
            ("loi nhuan truoc du phong", "loi nhuan truoc chi phi du phong"),
            statement="income_statement", gross_net="net"),
    _metric("credit_provision_expense", "chi phi du phong rui ro tin dung",
            ("chi phi du phong tin dung", "chi phi du phong"),
            statement="income_statement", sign="absolute"),
    _metric("customer_loan_provision_expense",
            "trich lap du phong rui ro cho vay khach hang",
            ("chi phi trich lap du phong rui ro cho vay khach hang",
             "trich lap du phong cho vay khach hang"),
            statement="income_statement", granularity="aggregate",
            sign="absolute"),
    _metric("general_customer_loan_provision",
            "du phong chung cho vay khach hang",
            ("trich lap du phong chung cho vay khach hang",),
            statement="income_statement", granularity="detail",
            sign="absolute"),
    _metric("specific_customer_loan_provision",
            "du phong cu the cho vay khach hang",
            ("trich lap du phong cu the cho vay khach hang",),
            statement="income_statement", granularity="detail",
            sign="absolute"),
    _metric("performing_loans", "du no du tieu chuan",
            ("no du tieu chuan",), statement="balance_sheet",
            granularity="detail"),
    _metric("loans_to_economic_entities", "cho vay cac to chuc kinh te",
            ("cho vay doi voi cac to chuc kinh te va ca nhan trong nuoc",
             "cho vay to chuc kinh te va ca nhan trong nuoc"),
            statement="balance_sheet", granularity="detail"),
    _metric("deposit_and_loan_interest_income", "lai tien gui va cho vay",
            ("lai tien gui tien cho vay", "lai tien gui"),
            statement="income_statement"),
    _metric("forex_commitments", "cam ket giao dich hoi doai",
            ("cam ket ngoai hoi",), statement="other"),
]


_NOTE_LINE_ITEMS = [
    _metric("external_receivables", "cac khoan phai thu ben ngoai",
            ("khoan phai thu ben ngoai", "so du cac khoan phai thu ben ngoai"),
            statement="balance_sheet", granularity="detail"),
    _metric("merchandise_inventory", "hang hoa ton kho",
            ("gia tri hang hoa ton kho", "hang hoa ton kho cuoi ky"),
            statement="balance_sheet", row_aliases=("hang hoa",),
            context=("hang ton kho",), granularity="detail"),
    _metric("unearned_revenue_short_term",
            "doanh thu chua thuc hien ngan han",
            ("tong doanh thu chua thuc hien ngan han",), ("318",),
            statement="balance_sheet", maturity="short",
            granularity="aggregate"),
    _metric("real_estate_brokerage_expense",
            "chi phi moi gioi bat dong san",
            ("tong chi phi hoa hong moi gioi bat dong san",
             "chi phi hoa hong moi gioi bat dong san"),
            statement="balance_sheet", context=("chi phi phai tra ngan han",),
            maturity="short", granularity="detail", sign="absolute"),
    _metric("construction_payables_short_term",
            "chi phi xay dung phai tra ngan han",
            ("chi phi xay dung phai tra",), statement="balance_sheet",
            row_aliases=("chi phi xay dung",),
            context=("chi phi phai tra",), maturity="short",
            granularity="detail"),
    _metric("lpg_revenue", "doanh thu thuan tu san pham khi lpg",
            ("doanh thu thuan san pham khi lpg", "doanh thu lpg"),
            statement="income_statement", gross_net="net",
            granularity="detail"),
    _metric("related_party_service_revenue",
            "doanh thu cung cap dich vu cho cac ben lien quan",
            ("tong doanh thu cung cap dich vu cho cac ben lien quan",
             "doanh thu voi cac ben lien quan",
             "doanh thu voi cac ben lien quan chu yeu"),
            statement="income_statement", granularity="detail"),
    _metric("crown_saigon_trade_payable",
            "phai tra cho cong ty lien doanh tnhh crown sai gon",
            ("so du phai tra cong ty lien doanh tnhh crown sai gon",),
            statement="balance_sheet",
            row_aliases=("cong ty lien doanh tnhh crown sai gon",),
            context=("phai tra nguoi ban",), maturity="short",
            granularity="detail"),
    _metric("supplier_prepayments_long_term", "tra truoc cho nguoi ban dai han",
            ("tra truoc nguoi ban dai han",), statement="balance_sheet",
            maturity="long", granularity="detail"),
    _metric("other_receivables_short_term", "phai thu ngan han khac",
            ("cac khoan phai thu khac ngan han",), statement="balance_sheet",
            maturity="short"),
    _metric("related_party_other_receivables_short_term",
            "phai thu ngan han khac tu cac ben lien quan",
            ("cac khoan phai thu ngan han khac tu cac ben lien quan",
             "so du cuoi nam cac khoan phai thu ngan han khac tu cac ben lien quan"),
            statement="balance_sheet", maturity="short", granularity="detail"),
    _metric("prepaid_expenses", "chi phi tra truoc", (),
            statement="balance_sheet", granularity="aggregate"),
    _metric("prepaid_expenses_short_term", "chi phi tra truoc ngan han",
            ("chi phi tra truoc ngan han khac",), statement="balance_sheet",
            maturity="short", granularity="detail"),
    _metric("prepaid_expenses_long_term", "chi phi tra truoc dai han", (),
            statement="balance_sheet", maturity="long", granularity="detail"),
    _metric("income_tax_payable", "thue thu nhap doanh nghiep phai nop",
            ("thue tndn phai nop", "thue thu nhap phai tra",
             "thue thu nhap doanh nghiep phai tra"),
            statement="balance_sheet"),
    _metric("bonus_welfare_fund", "quy khen thuong phuc loi",
            ("quy khen thuong va phuc loi",), statement="balance_sheet"),
    _metric("related_party_short_term_borrowings",
            "vay ngan han phai tra cac ben lien quan",
            ("vay ngan han cac ben lien quan",
             "vay ngan han phai tra ben lien quan"),
            statement="balance_sheet", row_aliases=("vay ngan han cac ben lien quan",),
            maturity="short", granularity="detail"),
    _metric("bank_short_term_borrowings", "vay ngan han ngan hang",
            ("so du vay ngan han ngan hang",), statement="balance_sheet",
            row_aliases=("vay ngan han ngan hang",), maturity="short",
            granularity="detail"),
    _metric("investment_property_net", "gia tri con lai cua bat dong san dau tu",
            ("bat dong san dau tu",), statement="balance_sheet",
            gross_net="net"),
    _metric("investment_property_cost", "nguyen gia bat dong san dau tu",
            ("gia goc bat dong san dau tu",), statement="balance_sheet",
            gross_net="gross"),
    _metric("goodwill_net", "gia tri con lai cua loi the thuong mai",
            ("loi the thuong mai",), statement="balance_sheet",
            gross_net="net"),
    _metric("issued_share_capital", "von co phan da phat hanh",
            ("von co phan", "co phan da phat hanh"),
            statement="balance_sheet"),
    _metric("shares_outstanding", "so luong co phieu dang luu hanh",
            ("co phieu pho thong dang luu hanh", "so co phieu dang luu hanh"),
            statement="balance_sheet"),
    _metric("ownership_rate", "ty le so huu", ("phan tram so huu",),
            statement="other", granularity="detail"),
    _metric("voting_rate", "ty le quyen bieu quyet",
            ("ty le bieu quyet", "quyen bieu quyet"),
            statement="other", granularity="detail"),
    _metric("regular_bonds", "trai phieu thuong",
            ("tong trai phieu thuong",), statement="balance_sheet",
            granularity="aggregate"),
    _metric("regular_bonds_short_term", "trai phieu thuong ngan han", (),
            statement="balance_sheet", maturity="short", granularity="detail"),
    _metric("regular_bonds_long_term", "trai phieu thuong dai han", (),
            statement="balance_sheet", maturity="long", granularity="detail"),
    _metric("accrued_interest_payable", "lai vay phai tra",
            ("chi phi lai vay phai tra",), statement="balance_sheet"),
    _metric("related_party_receivables", "phai thu ben lien quan",
            ("phai thu tu cac ben lien quan",), statement="balance_sheet",
            granularity="detail"),
    _metric("related_party_payables", "phai tra ben lien quan",
            ("phai tra cho cac ben lien quan",), statement="balance_sheet",
            granularity="detail"),
    _metric("outside_services_expense", "chi phi dich vu mua ngoai",
            ("dich vu mua ngoai",), statement="income_statement",
            sign="absolute"),
    _metric("salary_expense", "chi phi luong",
            ("chi phi nhan cong", "quy luong", "chi phi nhan vien"),
            statement="income_statement", sign="absolute"),
    _metric("operating_lease_commitments", "cam ket cho thue hoat dong",
            ("cam ket thue hoat dong", "tien thue toi thieu trong tuong lai",
             "tong tien thue toi thieu phai tra theo cac hop dong thue hoat dong khong duoc huy ngang",
             "cac khoan tien thue toi thieu phai tra cho cac hop dong thue hoat dong khong duoc huy ngang"),
            statement="other", granularity="aggregate"),
    _metric("borrowings_total", "vay va no",
            ("tong no vay", "tong cac khoan vay", "no vay"),
            statement="balance_sheet", granularity="aggregate"),
    _metric("borrowings_long_term", "vay dai han",
            ("cac khoan vay dai han", "no vay dai han"), statement="balance_sheet",
            maturity="long"),
    _metric("bonds_issued", "trai phieu phat hanh",
            ("phat hanh trai phieu",), statement="balance_sheet"),
    _metric("long_term_financial_investments", "dau tu tai chinh dai han",
            ("cac khoan dau tu tai chinh dai han",),
            statement="balance_sheet", maturity="long",
            granularity="aggregate"),
    _metric("investments_in_subsidiaries", "dau tu vao cong ty con",
            ("dau tu vao cac cong ty con",), statement="balance_sheet",
            granularity="detail"),
    _metric("investments_in_associates", "dau tu vao cong ty lien ket",
            ("dau tu vao cac cong ty lien ket", "dau tu cong ty lien ket"),
            statement="balance_sheet", granularity="detail"),
    _metric("other_equity_investments", "dau tu gop von vao don vi khac",
            ("gop von vao don vi khac",), statement="balance_sheet"),
    _metric("contract_progress_receivables",
            "phai thu theo tien do ke hoach hop dong",
            ("phai thu theo tien do hop dong",), statement="balance_sheet"),
    _metric("buyer_advances", "nguoi mua tra tien truoc",
            ("nguoi mua tra truoc", "tam ung nhan tu khach hang"),
            statement="balance_sheet"),
    _metric("internal_payables", "cac khoan phai tra noi bo",
            ("phai tra noi bo",), statement="balance_sheet"),
    _metric("provisions_payable", "du phong phai tra",
            ("tong du phong phai tra",), statement="balance_sheet",
            sign="absolute"),
    _metric("warranty_provision", "du phong chi phi bao hanh",
            ("du phong bao hanh",), statement="balance_sheet",
            sign="absolute"),
    _metric("trading_securities_cost", "gia goc chung khoan kinh doanh",
            ("chung khoan kinh doanh theo gia goc",),
            statement="balance_sheet", gross_net="gross"),
    _metric("land_use_right_net", "gia tri con lai cua quyen su dung dat",
            ("quyen su dung dat con lai",), statement="balance_sheet",
            gross_net="net"),
    _metric("weighted_average_common_shares",
            "so co phieu pho thong binh quan gia quyen",
            ("so luong co phieu pho thong binh quan gia quyen",),
            statement="income_statement"),
    _metric("common_shareholder_profit",
            "loi nhuan phan bo cho co dong so huu co phieu pho thong",
            ("loi nhuan phan bo cho co dong pho thong",),
            statement="income_statement"),
    _metric("board_compensation", "thu lao hoi dong quan tri",
            ("thu lao hdqt", "thu lao thanh vien hdqt"),
            statement="other", sign="absolute"),
    _metric("management_compensation", "thu nhap ban tong giam doc va quan ly",
            ("thu nhap ban tong giam doc", "thu nhap ban dieu hanh"),
            statement="other", sign="absolute"),
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
    _metric("cost_inventory_ratio", "gia von hang ban tren hang ton kho",
            ("ty le giua gia von hang ban tong cong va gia goc hang ton kho cuoi nam",
             "gia von hang ban chia cho hang ton kho"),
            components=("cost_of_goods_sold", "inventory")),
    _metric("interest_pretax_ratio", "chi phi lai vay tren loi nhuan truoc thue",
            ("ty le giua chi phi lai vay va loi nhuan truoc thue",),
            components=("interest_expense", "pretax_profit")),
    _metric("inventory_days", "so ngay hang ton kho",
            ("365 lan hang ton kho binh quan tren gia von hang ban",
             "365 lan hang ton kho binh quan dau ky va cuoi ky tren gia von hang ban",
             "365 lan trung binh hang ton kho dau nam va cuoi nam tren gia von hang ban",
             "hang ton kho binh quan nhan 365 chia cho gia von hang ban",
             "hang ton kho binh quan nhan 365 roi chia cho gia von hang ban"),
            components=("inventory", "inventory", "cost_of_goods_sold"),
            component_year_offsets=(-1, 0, 0)),
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
    metric.key: metric
    for metric in (*_LINE_ITEMS, *_BANK_LINE_ITEMS, *_NOTE_LINE_ITEMS, *_DERIVED)
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


def extract_metric_qualifiers(text: str, keys: Iterable[str] = (),
                              include_defaults: bool = True) -> MetricQualifiers:
    """Extract the controlled semantic dimensions from a question or label."""
    value = norm(text)
    key_list = tuple(k for k in keys if k in METRICS)
    defaults = _qualifier_defaults(key_list) if include_defaults else MetricQualifiers()

    period = ""
    if any(p in value for p in ("so dau nam", "dau nam", "dau ky", "1 1")):
        period = "opening"
    elif any(p in value for p in ("so cuoi nam", "cuoi nam", "cuoi ky", "31 12",
                                  "den ngay", "tai ngay")):
        period = "closing"

    maturity = ""
    if "ngan han" in value:
        maturity = "short"
    elif "trung han" in value:
        maturity = "medium"
    elif "dai han" in value:
        maturity = "long"

    gross_net = ""
    if any(p in value for p in ("gia goc", "nguyen gia", "gia tri gop", "du no gop")):
        gross_net = "gross"
    elif any(p in value for p in ("gia tri thuan", "gia tri con lai", "sau du phong",
                                  "gia tri rong")):
        gross_net = "net"

    granularity = ""
    if any(p in value for p in ("tong cong", "tong gia tri", "tong so", "tong du no",
                                "tong thu nhap", "tong chi phi", "quy mo")):
        granularity = "aggregate"
    elif any(p in value for p in ("trong do", "cu the", "chi tiet", "doi voi",
                                  "tai cong ty", "voi cong ty")):
        granularity = "detail"

    stock_flow = ""
    if period or any(p in value for p in ("so du", "gia tri ghi so", "du no")):
        stock_flow = "stock"
    elif any(p in value for p in ("trong nam", "trong ky", "trich lap",
                                  "doanh thu", "thu nhap", "loi nhuan",
                                  "luu chuyen tien", "dong tien")):
        stock_flow = "flow"

    sign = ""
    signed_phrases = ("chenh lech", "tru di", "tang truong", "thay doi",
                      "tang giam", "cao hon", "thap hon", "nhieu hon", "it hon")
    signed_words = re.search(r"(?<![0-9a-z])(?:am|duong)(?![0-9a-z])", value)
    if any(p in value for p in signed_phrases) or signed_words:
        sign = "signed"
    elif any(p in value for p in ("chi phi", "gia von", "quy mo", "trich lap du phong")):
        sign = "absolute"

    return MetricQualifiers(
        stock_flow=stock_flow or defaults.stock_flow,
        gross_net=gross_net or defaults.gross_net,
        maturity=maturity or defaults.maturity,
        period=period or defaults.period,
        granularity=granularity or defaults.granularity,
        sign=sign or defaults.sign,
    )


def metric_schema_score(phrases: Iterable[str], label: str,
                        question: str = "") -> float:
    """Score canonical identity and qualifier agreement for one row label.

    Lexical similarity alone rates a parent row as a perfect match for its
    child.  Canonical disagreement must therefore be a larger penalty than any
    small token-overlap bonus used by the shortlist.
    """
    phrase_list = _dedupe(phrases)
    asked = set(metric_keys(phrase_list, expand_derived=False))
    labelled = set(metric_keys([label], expand_derived=False))
    score = 0.0
    if asked and labelled:
        score += 16.0 if asked & labelled else -36.0

    asked_q = extract_metric_qualifiers(
        " ".join((*phrase_list, norm(question))), asked)
    label_q = extract_metric_qualifiers(label, labelled)
    for field, reward, penalty in (
        ("stock_flow", 3.0, -18.0),
        ("gross_net", 6.0, -28.0),
        ("maturity", 8.0, -30.0),
        ("granularity", 5.0, -24.0),
    ):
        want = getattr(asked_q, field)
        have = getattr(label_q, field)
        if want and have:
            score += reward if want == have else penalty

    # Defaults identify the economic line, but an explicit qualifier is stronger
    # evidence inside note tables. Prefer "gia tri con lai" over an otherwise
    # valid generic parent when the question explicitly asks for the net figure.
    asked_explicit = extract_metric_qualifiers(
        " ".join((*phrase_list, norm(question))), include_defaults=False)
    label_explicit = extract_metric_qualifiers(label, include_defaults=False)
    if asked_explicit.gross_net:
        if label_explicit.gross_net:
            score += (8.0 if asked_explicit.gross_net == label_explicit.gross_net
                      else -24.0)
        else:
            score -= 18.0
    return score


def metric_uses_absolute_value(text: str, keys: Iterable[str] = ()) -> bool:
    return extract_metric_qualifiers(text, keys).sign == "absolute"


def _qualifier_defaults(keys: tuple[str, ...]) -> MetricQualifiers:
    def one(field: str) -> str:
        values = {getattr(METRICS[key].qualifiers, field) for key in keys
                  if getattr(METRICS[key].qualifiers, field)}
        return next(iter(values)) if len(values) == 1 else ""

    return MetricQualifiers(**{
        field: one(field) for field in MetricQualifiers.__dataclass_fields__
    })


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


def metric_evidence_components(key: str) -> tuple[tuple[str, int], ...]:
    """Return atomic metric operands with their relative fiscal-year offsets."""
    metric = get_metric(key)
    if not metric.is_derived:
        return ((key, 0),)
    offsets = metric.component_year_offsets or (0,) * len(metric.components)
    if len(offsets) != len(metric.components):
        raise ValueError(f"invalid component year offsets for metric {key}")
    out: list[tuple[str, int]] = []
    for component, offset in zip(metric.components, offsets):
        for atomic_key, nested_offset in metric_evidence_components(component):
            item = (atomic_key, int(offset) + nested_offset)
            if item not in out:
                out.append(item)
    return tuple(out)


@lru_cache(maxsize=32768)
def _metric_keys_cached(texts: tuple[str, ...],
                        expand_derived: bool) -> tuple[str, ...]:
    seen, out = set(), []
    for text in texts:
        for match in find_metrics(text):
            keys = (match.metric.components
                    if match.metric.is_derived and expand_derived
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
        expanded.extend(metric.row_variants[:max(1, aliases_per_metric)])
    return _dedupe(expanded)


def metric_context_matches(metric_key: str, context: str) -> bool:
    """Require note-table context only for metrics whose row label is ambiguous."""
    try:
        phrases = METRICS[metric_key].context_phrases
    except KeyError:
        return False
    context_norm = norm(context)
    return not phrases or any(phrase in context_norm for phrase in phrases)


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
