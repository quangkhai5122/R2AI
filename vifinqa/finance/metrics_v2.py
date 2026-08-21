"""Canonical metric dictionary v2 for difficult Vietnamese financial notes.

V1 covers common VAS line items. V2 adds typed profiles for bank disclosures
and note tables where a broad parent row contains several child rows. Profiles
are used for query variants and exact row rejection; they do not replace V1.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from ..utils.viet_text import norm

PROFILE_VERSION = "canonical_metric_v2_2026_08_18f"


@dataclass(frozen=True)
class MetricProfile:
    key: str
    label: str
    aliases: tuple[str, ...] = ()
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_codes: tuple[str, ...] = ()
    column_phrases: tuple[str, ...] = ()
    statement: str = "other"
    parent_keys: tuple[str, ...] = ()
    qualifiers: tuple[tuple[str, str], ...] = ()
    child_exact: bool = False

    @property
    def variants(self) -> tuple[str, ...]:
        # Required phrases are a row-validation contract, not aliases. Treating
        # a token such as "tien" as an alias activates unrelated profiles.
        return _dedupe((self.label, *self.aliases))

    def to_dict(self) -> dict:
        return asdict(self)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen, out = set(), []
    for value in values:
        value = norm(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _profile(key: str, label: str, aliases=(), required=(), forbidden=(),
             codes=(), columns=(), statement="other", parent=(), qualifiers=(), child=False):
    return MetricProfile(
        key=key, label=norm(label), aliases=_dedupe(aliases),
        required_phrases=_dedupe(required), forbidden_phrases=_dedupe(forbidden),
        expected_codes=tuple(str(c) for c in codes), column_phrases=_dedupe(columns),
        statement=statement,
        parent_keys=tuple(parent), qualifiers=tuple((norm(k), norm(v)) for k, v in qualifiers),
        child_exact=child,
    )


# High-precision profiles are listed before broad fallback profiles.
PROFILES: tuple[MetricProfile, ...] = (
    # Banking: TCTD, customer loans, provisions and bank income.
    _profile("bank_customer_loans_financial_asset_receivables",
              "cho vay va phai thu cua tai san tai chinh trong hoat dong cho vay khach hang",
              ("cho vay va phai thu trong hoat dong cho vay khach hang",
               "tai san tai chinh trong hoat dong cho vay khach hang"),
              required=("cho vay", "khach hang"),
              columns=("cho vay va phai thu",),
              forbidden=("du phong",), parent=("bank_customer_loans",), child=True),
    _profile("bank_customer_loans_gross", "cho vay khach hang gop",
              ("cho vay khach hang - gop", "du no cho vay khach hang gop",
               "cho vay khach hang gross"),
              required=("cho vay", "khach hang", "gop"), forbidden=("du phong",),
              qualifiers=(("gross_net", "gross"),), child=True),
    _profile("bank_customer_loans_net", "cho vay khach hang thuan",
              ("cho vay khach hang sau du phong", "du no cho vay khach hang thuan"),
              required=("cho vay", "khach hang", "thuan"), forbidden=("du phong", "gop"),
              qualifiers=(("gross_net", "net"),), child=True),
    _profile("bank_customer_loans", "cho vay khach hang",
              ("du no cho vay khach hang", "cho vay khach hang"),
              required=("cho vay", "khach hang"), forbidden=("du phong",)),
    _profile("bank_other_tctd_deposits", "tien gui tai cac tctd khac",
              ("tien gui tai tctd khac", "tien gui cac to chuc tin dung khac"),
              required=("tien gui", "tctd"), forbidden=("cho vay",), child=True),
    _profile("bank_other_tctd_loans", "cho vay cac tctd khac",
              ("cho vay tctd khac", "cho vay cac to chuc tin dung khac",
               "cap tin dung cho cac tctd khac"),
              required=("cho vay", "tctd"), forbidden=("khach hang",), child=True),
    _profile("bank_other_tctd_deposits_loans", "tien gui va cho vay cac tctd khac",
              ("tien gui va cho vay tctd khac", "tien gui va cho vay cac to chuc tin dung khac"),
              required=("tien gui", "cho vay", "tctd"), child=True),
    _profile("bank_customer_loan_provision", "du phong rui ro cho vay khach hang",
              ("du phong cho vay khach hang", "du phong rui ro tin dung cho vay khach hang"),
              required=("du phong", "cho vay khach hang"), child=True),
    _profile("bank_specific_loan_provision", "du phong cu the cho vay khach hang",
              ("trich lap du phong cu the cho vay khach hang",),
              required=("du phong", "cu the"), parent=("bank_customer_loan_provision",), child=True),
    _profile("bank_general_loan_provision", "du phong chung cho vay khach hang",
              ("trich lap du phong chung cho vay khach hang",),
              required=("du phong", "chung"), parent=("bank_customer_loan_provision",), child=True),
    _profile("bank_loan_provision_expense", "chi phi du phong rui ro tin dung",
              ("chi phi du phong", "chi phi trich lap du phong",
               "trich lap du phong rui ro tin dung", "chi phi du phong cho vay khach hang"),
              required=("du phong",), qualifiers=(("stock_flow", "flow"),), child=True),
    _profile("bank_preprovision_profit", "loi nhuan truoc chi phi du phong rui ro tin dung",
              ("loi nhuan truoc du phong", "loi nhuan truoc chi phi du phong"),
              required=("loi nhuan", "truoc", "du phong"), child=True),
    _profile("bank_net_interest_income", "thu nhap lai thuan",
              ("lai thuan", "thu nhap lai va cac khoan thu nhap tuong tu tru chi phi lai"),
              required=("lai", "thuan"), forbidden=("chi phi lai",), statement="bank_income"),
    _profile("bank_interest_income", "thu nhap lai",
              ("thu nhap lai va cac khoan thu nhap tuong tu", "lai tien gui"),
              required=("thu nhap lai",), forbidden=("chi phi lai", "lai thuan"), statement="bank_income"),
    _profile("bank_deposit_interest_expense", "chi phi lai tien gui",
              ("chi phi lai tien gui cua khach hang",),
              required=("chi phi", "lai", "tien gui"), statement="bank_income", child=True),
    _profile("bank_service_income", "lai thuan tu hoat dong dich vu",
              ("lai thuan hoat dong dich vu", "thu nhap thuan tu hoat dong dich vu"),
              required=("lai", "thuan", "dich vu"), statement="bank_income"),
    _profile("bank_customer_deposits", "tien gui cua khach hang",
              ("tien gui khach hang",), required=("tien gui", "khach hang"), child=True),
    _profile("bank_employee_expense", "chi phi cho nhan vien",
              ("chi phi nhan vien", "chi cho nhan vien", "chi phi luong nhan vien"),
              required=("chi phi", "nhan vien"), child=True),
    _profile("bank_net_profit_parent", "loi nhuan thuan cua ngan hang me",
              ("loi nhuan thuan sau thue cua ngan hang me", "loi nhuan thuan phan bo cho ngan hang me"),
              required=("loi nhuan", "ngan hang me"), child=True),
    _profile("bad_debt", "no xau", ("du no xau", "no duoi tieu chuan", "no nghi ngo"),
              required=("no xau",)),
    _profile("bad_debt_coverage", "ty le bao phu no xau",
              ("bao phu no xau",), required=("bao phu", "no xau")),
    _profile("credit_provision_coverage", "ty le du phong rui ro tin dung",
              ("du phong chung trong tong du phong rui ro", "du phong cu the trong tong du phong rui ro",
               "chi phi du phong rui ro tin dung tren loi nhuan truoc du phong"),
              required=("du phong",), qualifiers=(("stock_flow", "stock"),)),

    # Notes and standard statements.
    _profile("fixed_assets_tangible", "tai san co dinh huu hinh",
              ("tscd huu hinh",), required=("tai san co dinh", "huu hinh"),
              forbidden=("vo hinh",), child=True),
    _profile("fixed_assets_intangible", "tai san co dinh vo hinh",
              ("tscd vo hinh",), required=("tai san co dinh", "vo hinh"),
              forbidden=("huu hinh",), child=True),
    _profile("fixed_assets_net_value", "gia tri con lai tai san co dinh",
              ("gia tri con lai cua tai san co dinh", "gia tri thuan tai san co dinh", "gia tri con lai tscd"),
              required=("gia tri con lai", "tai san co dinh"), child=True),
    _profile("land_use_right_net", "gia tri con lai quyen su dung dat",
              ("gia tri thuan quyen su dung dat", "quyen su dung dat con lai"),
              required=("quyen su dung dat", "con lai"), child=True),
    _profile("investment_property_net", "gia tri con lai bat dong san dau tu",
              ("gia tri thuan bat dong san dau tu",),
              required=("bat dong san dau tu", "con lai"), child=True),
    _profile("construction_in_progress_detail", "xay dung co ban do dang",
              ("chi phi xay dung co ban do dang", "xay dung nha ga"),
              required=("xay dung", "do dang"), child=True),
    _profile("operating_lease_commitment", "cam ket thue hoat dong",
              ("tong cam ket cho thue hoat dong", "cam ket cho thue hoat dong"),
              required=("thue hoat dong",), qualifiers=(("aggregate_detail", "aggregate"),)),
    _profile("operating_lease_minimum_payment", "tien thue toi thieu thue hoat dong",
              ("tong so tien thue toi thieu", "tien thue toi thieu phai tra", "tien thue toi thieu phai nhan"),
              required=("thue toi thieu",), child=True),
    _profile("related_receivables_short", "phai thu ngan han ben lien quan",
              ("phai thu ngan han cac ben lien quan", "tong phai thu ngan han tu cac ben lien quan"),
              required=("phai thu", "ngan han", "lien quan"), child=True),
    _profile("related_receivables_long", "phai thu dai han ben lien quan",
              ("phai thu dai han cac ben lien quan", "tong phai thu dai han tu cac ben lien quan"),
              required=("phai thu", "dai han", "lien quan"), child=True),
    _profile("related_payables_short", "phai tra ngan han ben lien quan",
              ("phai tra ngan han cac ben lien quan",), required=("phai tra", "ngan han", "lien quan"), child=True),
    _profile("related_loan_long", "cho vay dai han ben lien quan",
              ("khoan cho vay dai han ben lien quan",), required=("cho vay", "dai han", "lien quan"), child=True),
    _profile("ownership_percentage", "ty le so huu",
              ("phan tram so huu", "ty trong so huu", "so huu cua"), required=("so huu",)),
    _profile("voting_rights", "quyen bieu quyet",
              ("ty le quyen bieu quyet", "ty le bieu quyet", "phan tram bieu quyet"), required=("bieu quyet",)),
    _profile("common_shares_outstanding", "co phieu pho thong dang luu hanh",
              ("so luong co phieu pho thong dang luu hanh", "co phieu dang luu hanh", "so luong co phieu dang luu hanh"),
              required=("co phieu", "luu hanh"), child=True),
    _profile("issued_share_capital", "von co phan da phat hanh",
              ("co phieu pho thong co quyen bieu quyet",), required=("co phan", "phat hanh"), child=True),
    _profile("board_remuneration", "thu lao hoi dong quan tri",
              ("tong thu lao hoi dong quan tri", "thu lao hdqt"), required=("thu lao", "hoi dong quan tri"), child=True),
    _profile("employee_compensation", "chi phi luong va cac khoan khac theo luong",
              ("chi phi cho nhan vien",), required=("chi phi", "luong"), child=True),
    _profile("bad_receivable_provision", "du phong phai thu kho doi",
              ("du phong phai thu ngan han kho doi", "du phong phai thu dai han kho doi"),
              required=("du phong", "phai thu", "kho doi"), child=True),
    _profile("provision_appropriation", "du phong trich lap trong nam",
              ("du phong duoc trich lap", "du phong trich lap trong nam", "so trich lap du phong trong nam", "chi phi trich lap du phong"),
              required=("trich lap", "du phong"), qualifiers=(("stock_flow", "flow"),), child=True),
    _profile("current_income_tax_payable", "thue thu nhap doanh nghiep phai nop",
              ("thue tndn phai nop", "so du thue thu nhap phai nop", "thue thu nhap phai tra"),
              required=("thue", "phai nop"), child=True),
    _profile("issued_valuable_papers", "giay to co gia phat hanh",
              ("tong giay to co gia phat hanh", "phat hanh giay to co gia"), required=("giay to co gia", "phat hanh"), child=True),
    _profile("derivative_contract_value", "cong cu tai chinh phai sinh",
              ("tong gia tri hop dong cong cu phai sinh", "cong cu phai sinh"), required=("phai sinh",), child=True),
    _profile("cash_and_equivalents", "tien va cac khoan tuong duong tien",
              ("tong tien mat va cac khoan tuong duong tien", "tien mat va vang"), required=("tien",), forbidden=("tien gui khach hang",)),
    _profile("other_income_v2", "thu nhap khac", ("tong thu nhap khac",), required=("thu nhap khac",)),
    _profile("other_expense_v2", "chi phi khac", ("tong chi phi khac",), required=("chi phi khac",)),
    _profile("interest_expense_payable", "lai vay phai tra",
              ("chi phi lai vay phai tra", "lai phai tra"), required=("lai", "phai tra"), child=True),
    _profile("foreign_currency_position", "trang thai ngoai te",
              ("so du ngoai te", "tai san tien te", "cong no tien te"), required=("ngoai te",)),
    _profile("investment_at_cost", "gia goc khoan dau tu",
              ("gia tri dau tu", "gia goc chung khoan kinh doanh"), required=("gia goc", "dau tu"), child=True),
    _profile("other_financial_assets", "tai san tai chinh khac",
              ("tai san tai chinh chiu rui ro tin dung", "gia tri tai san tai chinh"), required=("tai san tai chinh",), child=True),
)


# Additional note/direct profiles discovered from the 330 unresolved questions.
# These stay deliberately narrow; generic words such as "von", "cac" and
# "so du" are not registered because they describe context, not a metric.
_EXTRA_PROFILES: tuple[MetricProfile, ...] = (
    _profile("bonus_welfare_fund", "quy khen thuong phuc loi",
              ("quy khen thuong va phuc loi", "quy khen thuong phuc loi"),
              required=("quy", "khen thuong", "phuc loi"), child=True),
    _profile("penalty_expense", "chi phi phat", ("chi phi tien phat",), required=("chi phi", "phat"), child=True),
    _profile("employee_remuneration", "thu lao thanh vien hoi dong quan tri", ("thu lao cua thanh vien hdqt", "thu lao chu tich hdqt", "thu lao ong chu tich"), required=("thu lao",), child=True),
    _profile("labor_cost", "chi phi nhan cong", ("chi phi lao dong", "chi phi nhan vien"), required=("chi phi", "nhan"), child=True),
    _profile("borrowings_and_debt", "vay va no", ("vay va no thue tai chinh", "vay va phat hanh trai phieu"), required=("vay", "no"), child=True),
    _profile("long_term_borrowing_related", "vay dai han ben lien quan", ("vay dai han voi", "vay dai han cong ty me"), required=("vay dai han",), child=True),
    _profile("long_term_supplier_prepayment", "tra truoc cho nguoi ban dai han", ("tong so tra truoc cho nguoi ban dai han",), required=("tra truoc", "nguoi ban", "dai han"), child=True),
    _profile("net_investment_contribution", "dau tu gop von vao don vi khac", ("gia tri thuan dau tu gop von vao don vi khac", "khoan dau tu gop von vao don vi khac"), required=("dau tu", "gop von", "don vi khac"), child=True),
    _profile("aircraft_dry_lease_revenue", "doanh thu cho thue kho tau bay", ("doanh thu tu cho thue kho tau bay", "doanh thu cho thue kho may bay"), required=("doanh thu", "thue", "tau bay"), child=True),
    _profile("contract_progress_receivable", "phai thu theo tien do ke hoach hop dong", ("phai thu theo tien do ke hoach", "phai thu theo tien do hop dong"), required=("phai thu", "tien do", "ke hoach"), child=True),
    _profile("insurance_related_receivable", "phai thu tu bao viet nhan tho", ("khoan phai thu tu bao viet nhan tho",), required=("phai thu", "nhan tho"), child=True),
    _profile("goodwill_net_value", "gia tri con lai loi the thuong mai", ("gia tri con lai cua loi the thuong mai", "gia tri thuan loi the thuong mai"), required=("loi the thuong mai", "con lai"), child=True),
    _profile("central_management_fee_receivable", "phai thu phi quan ly tap trung", ("so du phai thu phi quan ly tap trung",), required=("phai thu", "phi quan ly"), child=True),
    _profile("vamc_special_bond", "trai phieu dac biet vamc", ("trai phieu dac biet do vamc phat hanh", "trai phieu dac biet do vamc"), required=("trai phieu", "vamc"), child=True),
    _profile("medium_term_notes_bonds", "ky phieu trai phieu trung han", ("ky phieu va trai phieu trung han", "trai phieu trung han"), required=("trai phieu", "trung han"), child=True),
    _profile("fast_transfer_receivable", "phai thu nghiep vu chuyen tien nhanh", ("phai thu trong nghiep vu chuyen tien nhanh",), required=("phai thu", "chuyen tien nhanh"), child=True),
    _profile("domestic_enterprise_person_loans", "cho vay to chuc kinh te va ca nhan trong nuoc", ("cho vay doi voi cac to chuc kinh te va ca nhan trong nuoc", "cho vay cac to chuc kinh te va ca nhan"), required=("cho vay", "to chuc", "ca nhan"), child=True),
    _profile("afs_securities", "chung khoan dau tu san sang de ban", ("chung khoan dau tu san sang ban", "chung khoan san sang de ban"), required=("chung khoan", "san sang de ban"), child=True),
    _profile("standard_loans", "du no du tieu chuan", ("du no tieu chuan", "no du tieu chuan"), required=("du no", "tieu chuan"), child=True),
    _profile("industry_customer_loans", "du no cho vay theo nganh", ("du no cho vay nganh thuong mai dich vu", "du no cho vay theo nganh nghe kinh doanh", "cho vay khach hang nganh"), required=("cho vay", "nganh"), child=True),
    _profile("associate_investment_book_value", "gia tri ghi so dau tu cong ty lien ket", ("tong gia tri ghi so dau tu vao cong ty lien ket", "ghi so dau tu vao cong ty lien ket"), required=("ghi so", "dau tu", "cong ty lien ket"), child=True),
    _profile("direct_fund_contribution", "von gop truc tiep vao quy dau tu", ("khoan muc von gop truc tiep vao quy", "von gop truc tiep vao quy dau tu gia tri"), required=("von gop", "truc tiep", "quy dau tu"), child=True),
    _profile("other_receivables_short", "phai thu khac ngan han", ("tong so du phai thu khac ngan han",), required=("phai thu", "khac", "ngan han"), child=True),
    _profile("debt_securities", "chung khoan no", ("chung khoan no cua", "du no chung khoan"), required=("chung khoan no",), child=True),
    _profile("fx_commitment", "cam ket giao dich hoi doai", ("so du cam ket giao dich hoi doai",), required=("cam ket", "hoi doai"), child=True),
    _profile("bank_interest_expense", "chi phi lai va cac chi phi tuong tu", ("chi phi lai", "chi phi lai va cac chi phi tuong tu"), required=("chi phi", "lai"), forbidden=("lai vay da tra",), statement="bank_income", child=True),
    _profile("current_income_tax_expense", "chi phi thue thu nhap doanh nghiep hien hanh", ("chi phi thue thu nhap hien hanh", "chi phi thue hien hanh", "chi phi thue tndn hien hanh"), required=("chi phi", "thue", "hien hanh"), child=True),
    _profile("investment_property_cost", "nguyen gia bat dong san dau tu", ("nguyen gia bat dong san dau tu",), required=("nguyen gia", "bat dong san dau tu"), child=True),
    _profile("other_prepaid_expense_short", "chi phi tra truoc ngan han khac", ("chi phi tra truoc ngan han khac",), required=("chi phi tra truoc", "ngan han"), child=True),
    _profile("total_shares", "so luong co phan", ("tong so luong co phan", "so luong co phieu"), required=("so luong", "co phan"), child=True),
    _profile("warranty_provision_short", "du phong chi phi bao hanh ngan han", ("du phong bao hanh ngan han",), required=("du phong", "bao hanh", "ngan han"), child=True),
    _profile("term_deposit_short", "tien gui co ky han ngan han", ("tien gui co ky han", "tien gui tiet kiem"), required=("tien gui", "ky han"), child=True),
    _profile("term_deposit_foreign", "tien gui co ky han bang ngoai te", ("tien gui co ky han ngoai te",), required=("tien gui", "ky han", "ngoai te"), child=True),
    _profile("individual_customer_deposits", "tien gui cua ca nhan", ("tien gui cua ca nhan va cac doi tuong khac", "tien gui ca nhan"), required=("tien gui", "ca nhan"), child=True),
    _profile("total_funds", "tong nguon von", ("tong cong nguon von", "nguon von"), required=("nguon von",), child=True),
    _profile("operating_expense_total", "tong chi phi hoat dong", ("chi phi hoat dong", "tong chi phi hoat dong"), required=("chi phi", "hoat dong"), child=True),
    _profile("individual_income", "thu nhap ca nhan", ("thu nhap cua", "thu nhap ong"), required=("thu nhap",), child=True),
    _profile("customer_advance_short", "nguoi mua tra truoc ngan han", ("khach hang tra truoc ngan han", "tien tra truoc ngan han cua khach hang"), required=("tra truoc", "khach hang"), child=True),
    _profile("customer_real_estate_advance", "tam ung nhan tu khach hang mua bat dong san", ("tien tam ung nhan tu khach hang mua bat dong san",), required=("tam ung", "khach hang", "bat dong san"), child=True),
    _profile("deferred_tax_expense", "chi phi thue thu nhap hoan lai", ("chi phi thue hoan lai", "thu nhap thue thu nhap hoan lai"), required=("thue", "hoan lai"), child=True),
    _profile("ordinary_bonds", "trai phieu thuong", ("tong trai phieu thuong",), required=("trai phieu thuong",), child=True),
    _profile("accrued_dividend_income", "du thu co tuc loi nhuan duoc chia", ("du thu co tuc", "du thu loi nhuan duoc chia"), required=("du thu", "co tuc"), child=True),
    _profile("investment_subsidiaries", "dau tu vao cong ty con", ("tu vao cac cong ty con", "nguyen gia dau tu vao cong ty con"), required=("dau tu", "cong ty con"), child=True),
    _profile("loan_maturity_medium", "vay trung han", ("so du vay trung han",), required=("vay", "trung han"), child=True),
    _profile("internal_payables", "cac khoan phai tra noi bo", ("phai tra noi bo",), required=("phai tra", "noi bo"), child=True),
    _profile("provision_payable", "du phong phai tra", ("tong cong du phong phai tra",), required=("du phong", "phai tra"), child=True),
    _profile("accounting_profit_after_tax", "loi nhuan ke toan sau thue tndn", ("loi nhuan ke toan sau thue",), required=("loi nhuan", "sau thue"), child=True),
    _profile("total_customer_loans", "tong du no cho vay", ("du no cho vay", "tong cho vay"), required=("du no", "cho vay"), child=True),
    _profile("unsecured_long_term_loans", "cho vay dai han khong co tai san dam bao", ("cac khoan cho vay dai han khong co tai san dam bao",), required=("cho vay", "dai han", "khong co tai san"), child=True),
    _profile("financial_obligations", "nghia vu no tai chinh", ("tong cong nghia vu no tai chinh",), required=("nghia vu", "no tai chinh"), child=True),
    _profile("fair_value_fvtpl", "gia tri hop ly tai san tai chinh fvtpl", ("gia tri hop ly cua tai san tai chinh loai fvtpl", "tai san tai chinh loai fvtpl"), required=("fvtpl",), child=True),
    _profile("economic_interest", "ty le loi ich kinh te", ("loi ich kinh te", "ty le loi ich"), required=("loi ich", "kinh te"), child=True),
    _profile("vat_payable", "thue gia tri gia tang phai nop", ("thue gia tri gia tang phai nop", "thue gtgt phai nop"), required=("thue", "gia tri gia tang", "phai nop"), child=True),
    _profile("finance_lease_principal_short", "no goc thue tai chinh ky han duoi mot nam", ("no goc thue tai chinh ky han duoi 1 nam",), required=("thue tai chinh", "duoi"), child=True),
    _profile("financial_assets_total", "tong tai san tai chinh", ("so tai san tai chinh", "tong so tai san tai chinh"), required=("tai san tai chinh",), child=True),
    _profile("cash_national_bank", "tien gui tai ngan hang nha nuoc", ("tien gui tai nhnn", "tien gui tai ngan hang nha nuoc viet nam"), required=("tien gui", "ngan hang nha nuoc"), child=True),
    _profile("loan_shareholder_major", "cho vay co dong lon", ("tien ngan hang cho vay doi voi co dong lon",), required=("cho vay", "co dong lon"), child=True),
    _profile("related_payable_third_party", "phai tra nguoi ban ben thu ba", ("phai tra nguoi ban la ben thu ba", "nha cung cap"), required=("phai tra", "nguoi ban"), child=True),
    _profile("customer_other_receivable", "phai thu tu khach hang khac", ("phai thu khach hang khac",), required=("phai thu", "khach hang khac"), child=True),
    _profile("economic_organization_loans", "cho vay cac to chuc kinh te", ("vay cac to chuc kinh te", "cho vay to chuc kinh te"), required=("cho vay", "to chuc kinh te"), child=True),
)

PROFILES = (*PROFILES, *_EXTRA_PROFILES)

_BY_KEY = {profile.key: profile for profile in PROFILES}



# These profiles describe balance-sheet/note balances even when their label
# contains a word such as "chi phi". Do not let that lexical token alone turn
# an ending balance into a flow; explicit period/transaction language still
# wins below.
_STOCK_PROFILE_KEYS = frozenset({
    "current_income_tax_payable", "construction_in_progress_detail",
    "other_prepaid_expense_short", "short_term_prepaid_expense",
    "other_prepaid_expense", "long_term_prepaid_expense",
    "fixed_assets_tangible", "fixed_assets_intangible", "fixed_assets_net_value",
    "investment_at_cost", "long_term_investment_cost", "investment_property_cost",
    "investment_property_cost_v2", "investment_property_net_v2",
    "afs_securities", "customer_advance_short", "other_short_advance_customer",
    "common_shares_outstanding", "total_shares", "issued_share_capital",
    "borrowings_and_debt", "long_term_borrowing_total", "financial_debt_total",
    "total_customer_loans", "industry_customer_loans", "bank_customer_loans",
    "gross_receivables", "related_receivables_short", "related_receivables_long",
    "related_payables_short", "related_payable_third_party", "loan_maturity_medium",
    "finance_lease_principal_short", "land_use_right_net", "goodwill_net_value",
})


def _explicit_flow(text: str) -> bool:
    q = norm(text)
    return bool(re.search(
        r"\b(trong nam|phat sinh|da nop|da tra|thuc nop|luu chuyen|"
        r"tang|giam|bien dong|chuyen sang|mua|ban|doanh thu|thu nhap|"
        r"trich lap|chiem|dong gop)\b", q
    ))


def _profile_adjusted_flags(profile: MetricProfile, text: str) -> dict[str, str]:
    flags = qualifier_flags(text)
    if (profile.key in _STOCK_PROFILE_KEYS
            and flags.get("stock_flow") == "flow"
            and not _explicit_flow(text)):
        flags["stock_flow"] = "stock"
    if (profile.key == "current_income_tax_payable"
            and re.search(r"\b(phai tra|phai nop)\b", norm(text))):
        flags["stock_flow"] = "stock"
    return flags


def qualifier_flags(text: str) -> dict[str, str]:
    """Extract structured qualifier axes from a question or row label."""
    q = norm(text)
    flags: dict[str, str] = {}
    if re.search(r"\b(so du|tai ngay|cuoi nam|cuoi ky|dau nam|dau ky|gia tri con lai)\b", q):
        flags["stock_flow"] = "stock"
    elif re.search(r"\b(chi phi|trich lap|doanh thu|luu chuyen|mua|ban|phat sinh|thu nhap)\b", q):
        flags["stock_flow"] = "flow"
    if re.search(r"\b(gop|gross|tong gia tri)\b", q):
        flags["gross_net"] = "gross"
    elif re.search(r"\b(thuan|rong|net|gia tri con lai|gia tri thuan)\b", q):
        flags["gross_net"] = "net"
    if "ngan han" in q:
        flags["term"] = "short"
    elif "dai han" in q:
        flags["term"] = "long"
    if re.search(r"\b(dau nam|dau ky|so dau|opening)\b", q):
        flags["period"] = "opening"
    elif re.search(r"\b(cuoi nam|cuoi ky|so cuoi|closing)\b", q):
        flags["period"] = "closing"
    if re.search(r"\b(tong|gop|toan bo|aggregate)\b", q):
        flags["aggregate_detail"] = "aggregate"
    elif re.search(r"\b(chi tiet|theo nganh|doi voi|tai|voi cong ty|detail)\b", q):
        flags["aggregate_detail"] = "detail"
    if re.search(r"\b(gia tri tuyet doi|tri tuyet doi|absolute)\b", q):
        flags["sign"] = "absolute"
    elif re.search(r"\b(am|duong|signed)\b", q):
        flags["sign"] = "signed"
    return flags


def infer_profiles(texts: Iterable[str]) -> list[MetricProfile]:
    """Return high-precision v2 profiles mentioned by any input phrase."""
    combined = norm(" ".join(str(text or "") for text in texts))
    if not combined:
        return []
    found = []
    for profile in PROFILES:
        aliases = sorted(profile.variants, key=len, reverse=True)
        best = next((alias for alias in aliases if alias in combined), "")
        if not best:
            continue
        if any(forbidden in combined for forbidden in profile.forbidden_phrases):
            # A forbidden phrase only blocks a broad profile. A longer exact
            # child alias can still match its own profile later in the loop.
            if best == profile.label:
                continue
        found.append((len(best), profile))
    found.sort(key=lambda item: (-item[0], item[1].key))
    suppressed = {parent for _length, profile in found for parent in profile.parent_keys}
    selected, seen = [], set()
    for _length, profile in found:
        if profile.key in suppressed:
            continue
        if profile.key not in seen:
            selected.append(profile)
            seen.add(profile.key)
    return selected


def profile_keys(texts: Iterable[str]) -> list[str]:
    return [profile.key for profile in infer_profiles(texts)]


def expand_variants_v2(phrases: Iterable[str], question: str = "",
                       aliases_per_profile: int = 4) -> list[str]:
    """Append v2 aliases without deleting the caller's original phrases."""
    originals = list(_dedupe(phrases))
    profiles = infer_profiles([*originals, question])
    expanded = list(originals)
    for profile in profiles:
        expanded.extend(profile.variants[:max(1, aliases_per_profile)])
    return list(_dedupe(expanded))


def profiles_from_route(route: dict) -> list[MetricProfile]:
    keys = route.get("metric_profile_keys") or []
    if keys:
        return [_BY_KEY[key] for key in keys if key in _BY_KEY]
    return infer_profiles([
        route.get("metric_norm", ""), route.get("metric_wide", ""),
        *(route.get("metric_variants") or []), route.get("question", ""),
    ])


def _contains_all(text: str, phrases: Iterable[str]) -> bool:
    text = norm(text)
    return all(norm(phrase) in text for phrase in phrases if norm(phrase))


def row_profile_match(profile: MetricProfile, question: str, label: str,
                      code: str = "", col_name: str = "",
                      qualifier_text: str | None = None) -> tuple[bool, float, str]:
    """Hard profile gate plus a deterministic score adjustment.

    ``child_exact`` profiles reject a broad parent row unless the row carries
    every required child concept. This is the main protection against selecting
    "cho vay khách hàng" when the question asks for "dự phòng cụ thể".
    """
    q, row, column = norm(question), norm(label), norm(col_name)
    if profile.expected_codes and str(code).strip() in profile.expected_codes:
        code_bonus = 14.0
    else:
        code_bonus = 0.0
    if any(forbidden in row and forbidden not in q
           for forbidden in profile.forbidden_phrases):
        return False, -100.0, "forbidden phrase"
    required = [phrase for phrase in profile.required_phrases if phrase]
    if profile.column_phrases and not any(phrase in column for phrase in profile.column_phrases):
        return False, -100.0, "missing exact column phrase"
    alias_hit = max((len(alias) for alias in profile.variants if alias in row), default=0)
    required_hit = bool(required) and _contains_all(row, required)
    if not alias_hit and not required_hit:
        return False, -100.0, "missing canonical phrase"
    if profile.child_exact and not alias_hit and not required_hit:
        return False, -100.0, "missing exact child phrase"

    asked_flags = _profile_adjusted_flags(profile, q)
    qualifier_row = norm(qualifier_text) if qualifier_text is not None else row
    row_flags = _profile_adjusted_flags(profile, qualifier_row)
    if (profile.key == "current_income_tax_payable"
            and re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", qualifier_row)
            and not re.search(r"\b(phat sinh|da nop|da tra|trong nam)\b", qualifier_row)):
        row_flags["stock_flow"] = "stock"
    if (profile.key == "current_income_tax_payable"
            and re.search(r"\b(phai tra|phai nop)\b", q)):
        # Bank tax-note rows often say only "Thuế TNDN"; the payable meaning
        # is carried by the question and closing-date column.
        required_hit = True
        alias_hit = max(alias_hit, len(profile.label))
    for axis, expected in profile.qualifiers:
        wanted = asked_flags.get(axis, expected)
        present = row_flags.get(axis)
        if present and present != wanted:
            return False, -100.0, f"qualifier mismatch {axis}"
    for axis in ("stock_flow", "gross_net", "term", "period"):
        wanted, present = asked_flags.get(axis), row_flags.get(axis)
        if wanted and present and wanted != present:
            return False, -100.0, f"qualifier mismatch {axis}"

    bonus = code_bonus + min(18.0, alias_hit / 4.0)
    if required and _contains_all(row, required):
        bonus += 10.0
    return True, bonus, "profile match"


def best_row_profile(route: dict, label: str, code: str = "", col_name: str = "",
                     qualifier_text: str | None = None):
    """Return best compatible profile and adjustment for one candidate cell."""
    profiles = profiles_from_route(route)
    if not profiles:
        return None, 0.0, "no profile"
    accepted = []
    reasons = []
    for profile in profiles:
        ok, bonus, reason = row_profile_match(
            profile, route.get("question", ""), label, code, col_name,
            qualifier_text=qualifier_text)
        if ok:
            accepted.append((bonus, profile, reason))
        else:
            reasons.append(f"{profile.key}:{reason}")
    if not accepted:
        return None, -100.0, "; ".join(reasons[:3])
    bonus, profile, reason = max(accepted, key=lambda item: (item[0], item[1].key))
    return profile, bonus, reason


# Final direct/note coverage batch. The profiles below are intentionally
# phrase-specific and are not used to infer a metric from generic context words.
_MORE_PROFILES: tuple[MetricProfile, ...] = (
    _profile("external_service_expense", "chi phi dich vu mua ngoai",
              ("tong chi phi dich vu mua ngoai",), required=("chi phi", "dich vu", "mua ngoai"), child=True),
    _profile("salary_fund", "quy luong", ("tong quy luong", "tong quỹ lương"), required=("quy luong",), child=True),
    _profile("borrowing_interest_expense", "lai vay", ("chi phi lai vay", "lai vay phai tra"), required=("lai vay",), child=True),
    _profile("tax_expense_income", "thue thu nhap doanh nghiep", ("chi phi thue thu nhap", "thue thu nhap"), required=("thue thu nhap",), child=True),
    _profile("net_financial_income", "loi nhuan thuan tu hoat dong tai chinh",
              ("lai rong tu hoat dong tai chinh", "ket qua thuan tu hoat dong tai chinh", "ket qua hoat dong tai chinh rong", "loi nhuan thuần từ hoạt động tài chính"),
              required=("hoat dong tai chinh",), child=True),
    _profile("other_operating_income_net", "thu nhap thuan tu hoat dong khac",
              ("ket qua thuan tu hoat dong khac", "thu nhap tu hoat dong khac"), required=("hoat dong khac",), child=True),
    _profile("basic_eps_v2", "lai co ban tren moi co phieu",
              ("lai tren moi co phieu", "lai co ban tren mot co phieu", "loi nhuan tren moi co phieu co ban", "eps co ban"), required=("co phieu",), child=True),
    _profile("net_profit_margin_v2", "suat loi nhuan rong",
              ("suat loi nhuan ròng", "ty suat loi nhuan rong", "ty suat loi nhuan net"), required=("loi nhuan rong",), child=True),
    _profile("foreign_exchange_gain_loss", "lai lo chenh lech ty gia hoi doai",
              ("lo chenh lech ty gia", "lai chenh lech ty gia hoi doai", "lo ty gia"), required=("ty gia",), child=True),
    _profile("dividend_received", "co tuc nhan duoc",
              ("co tuc nhan duoc tu cac khoan dau tu", "co tuc loi nhuan duoc chia", "du thu co tuc loi nhuan duoc chia"), required=("co tuc",), child=True),
    _profile("dividend_paid", "tien chi tra co tuc",
              ("muc tien chi tra co tuc", "suat chi tra co tuc"), required=("co tuc", "chi tra"), child=True),
    _profile("long_term_prepaid_expense", "chi phi tra truoc dai han",
              ("tong chi phi tra truoc dai han", "chi phi thue tra truoc dai han"), required=("chi phi tra truoc", "dai han"), child=True),
    _profile("short_term_prepaid_expense", "chi phi tra truoc ngan han",
              ("tong chi phi tra truoc ngan han",), required=("chi phi tra truoc", "ngan han"), child=True),
    _profile("other_prepaid_expense", "chi phi tra truoc",
              ("chi phi cho phan bo", "chi phi cong cu dung cu cho phan bo"), required=("chi phi tra truoc",), child=True),
    _profile("investment_property_depreciation", "trich khau hao bat dong san dau tu",
              ("trich khau hao bat dong san dau tu cho thue", "chi phi khau hao bat dong san dau tu"), required=("khau hao", "bat dong san dau tu"), child=True),
    _profile("aircraft_maintenance_prepaid", "chi phi bao duong tau bay",
              ("chi phi tra truoc chi phi bao duong tau bay",), required=("bao duong", "tau bay"), child=True),
    _profile("gross_receivables", "gia goc no phai thu",
              ("tong gia goc no phai thu", "gia goc cac khoan phai thu"), required=("gia goc", "phai thu"), child=True),
    _profile("other_receivables_long", "phai thu khac dai han",
              ("phai thu khac dai han tong cong", "khoan phai thu khac dai han"), required=("phai thu", "khac", "dai han"), child=True),
    _profile("customer_advance_short", "nguoi mua tra truoc",
              ("nguoi mua tra truoc ngan han", "khach hang tra truoc ngan han"), required=("mua", "tra truoc"), child=True),
    _profile("long_term_investment_cost", "gia goc cac khoan dau tu",
              ("gia goc khoan dau tu", "gia goc dau tu vao cong ty lien ket"), required=("gia goc", "dau tu"), child=True),
    _profile("subsidary_investment_cost", "nguyen gia dau tu vao cong ty con",
              ("nguyen gia dau tu vao cong ty con", "dau tu vao cac cong ty con"), required=("dau tu", "cong ty con"), child=True),
    _profile("fair_value_investment", "gia tri hop ly khoan dau tu",
              ("gia tri hop ly cua khoan dau tu vao", "gia tri hop ly khoan dau tu vao"), required=("gia tri hop ly", "dau tu"), child=True),
    _profile("department_revenue", "doanh thu bo phan",
              ("tong doanh thu bo phan", "doanh thu cung cap dich vu cho cac ben lien quan"), required=("doanh thu",), child=True),
    _profile("construction_work_in_progress_cost", "chi phi san xuat kinh doanh do dang",
              ("chi phi san xuat va kinh doanh do dang", "chi phi san xuat kinh doanh do dang trong hoat dong xay lap"), required=("chi phi", "do dang"), child=True),
    _profile("raw_material_cost", "nguyen lieu vat lieu gia goc",
              ("nguyen lieu vat lieu", "chi phi nguyen lieu vat lieu", "chi phi nguyen vat lieu"), required=("nguyen lieu", "vat lieu"), child=True),
    _profile("production_business_cost", "chi phi san xuat va kinh doanh",
              ("tong chi phi san xuat va kinh doanh",), required=("chi phi", "san xuat", "kinh doanh"), child=True),
    _profile("transport_fuel_cost", "chi phi nhien lieu",
              ("chi phi mua khi", "chi phi mua khi tu cac chu mo", "chi phi van chuyen"), required=("chi phi",), child=True),
    _profile("construction_service_revenue", "nguon thu tu dich vu xay dung",
              ("doanh thu tu dich vu xay dung",), required=("dich vu xay dung",), child=True),
    _profile("other_customer_debt", "phai thu tu khach hang khac",
              ("phai thu khach hang khac", "khoan phai thu khach hang khac"), required=("phai thu", "khach hang"), child=True),
    _profile("other_related_payables", "phai tra ngan han khac ben lien quan",
              ("phai tra ngan han khac voi ben lien quan", "phai tra ngan han khac cac ben lien quan"), required=("phai tra", "ngan han", "lien quan"), child=True),
    _profile("total_related_transactions", "tong giao dich voi ben lien quan",
              ("tong gia tri giao dich voi ben lien quan", "giao dich voi ben lien quan"), required=("giao dich", "lien quan"), child=True),
    _profile("outside_receivables", "phai thu ben ngoai",
              ("so du cac khoan phai thu ben ngoai", "khoan phai thu ben ngoai"), required=("phai thu", "ben ngoai"), child=True),
    _profile("custody_assets", "tai san va chung tu giu ho",
              ("tai san va chung tu giu ho bao quan", "tai san giu ho"), required=("giu ho",), child=True),
    _profile("unearned_revenue_short", "doanh thu chua thuc hien ngan han",
              ("tong doanh thu chua thuc hien ngan han", "doanh thu chua thuc hien ngắn hạn"), required=("doanh thu chua thuc hien", "ngan han"), child=True),
    _profile("taxes_payable_total", "thue va cac khoan phai nop nha nuoc",
              ("tong thue va cac khoan phai nop nha nuoc", "thue va cac khoan phai nop"), required=("thue", "phai nop"), child=True),
    _profile("corporate_tax_paid", "thue thu nhap doanh nghiep da nop",
              ("thue tndn da nop", "thue thu nhap da nop", "tien thue da nop"), required=("thue", "da nop"), child=True),
    _profile("employee_average_income", "thu nhap binh quan nhan vien",
              ("thu nhap binh quan thang tren moi nhan vien", "thu nhap binh quan nam cua nhan vien", "thu nhap binh quan thang nguoi"), required=("thu nhap binh quan",), child=True),
    _profile("long_term_borrowing_total", "tong no vay dai han",
              ("tong du no vay dai han", "no vay dai han tu ngan hang", "vay va phat hanh trai phieu dai han"), required=("vay", "dai han"), child=True),
    _profile("government_bond_maturity", "trai phieu chinh phu giu den ngay dao han",
              ("trai phieu chinh phu", "dau tu trai phieu chinh phu"), required=("trai phieu chinh phu",), child=True),
    _profile("investment_financial_long_term", "dau tu tai chinh dai han",
              ("dau tu tai chinh dai han", "dau tu dai han"), required=("dau tu", "dai han"), child=True),
    _profile("lending_interest_accrued", "lai du thu cho vay",
              ("lai du thu tu cho vay", "lai cho vay chua thu duoc", "lai du thu tien gui co ky han"), required=("lai", "du thu"), child=True),
    _profile("lending_interest_revenue", "doanh thu lai tu hoat dong ngan hang",
              ("doanh thu lãi từ hoạt động ngân hàng", "thu nhap lai"), required=("doanh thu", "lai"), child=True),
    _profile("derivative_asset_value", "gia tri tai san phai sinh",
              ("san phai sinh", "tai san phai sinh", "gia tri tai san tai chinh phai sinh"), required=("phai sinh",), child=True),
    _profile("var_value", "gia tri rui ro var",
              ("gia tri rui ro var 1 ngay", "var 1 ngay danh muc co phieu niem yet"), required=("var",), child=True),
    _profile("actual_tax_rate", "thue suat thuc te",
              ("thue suat thuc te trung binh",), required=("thue suat thuc te",), child=True),
    _profile("liquidity_gap", "chenh lech thanh khoan rong",
              ("chenh lech thanh khoan rong tong", "thanh khoan rong trong han", "muc chenh thanh khoan rong"), required=("thanh khoan", "rong"), child=True),
    _profile("lc_commitment", "cam ket lc",
              ("so du cam ket l/c", "cam ket tin dung l/c", "cam ket lc"), required=("cam ket",), child=True),
    _profile("other_bank_tax_cost", "chi phi thue le phi va phi",
              ("chi phi thue le phi va phi",), required=("chi phi", "le phi"), child=True),
    _profile("loan_due_within_year", "vay dai han den han trong nam",
              ("khoan vay dai han den han trong nam",), required=("vay dai han", "den han"), child=True),
    _profile("listed_trading_securities", "chung khoan kinh doanh da niem yet",
              ("chung khoan kinh doanh niem yet",), required=("chung khoan kinh doanh", "niem yet"), child=True),
    _profile("investment_techcombank", "dau tu vao techcombank",
              ("khoan dau tu vao techcombank",), required=("dau tu", "techcombank"), child=True),
    _profile("share_capital_v2", "von co phan",
              ("von co phan cua", "von co phan da phat hanh"), required=("von co phan",), child=True),
    _profile("capital_charter", "von dieu le",
              ("von dieu le",), required=("von dieu le",), child=True),
    _profile("provision_other_asset", "du phong tai san co khac",
              ("du phong tai san co khac",), required=("du phong", "tai san co khac"), child=True),
    _profile("loan_related_debt", "du no tien vay cac ben lien quan",
              ("du no tien vay cac ben lien quan", "no vay cac ben lien quan"), required=("vay", "lien quan"), child=True),
)

PROFILES = (*PROFILES, *_MORE_PROFILES)
_BY_KEY = {profile.key: profile for profile in PROFILES}

_FINAL_PROFILES_A: tuple[MetricProfile, ...] = (
    _profile("fuel_stabilization_fund", "quy binh on gia xang dau", ("tai khoan tien gui quy binh on gia xang dau", "so du quy binh on gia xang dau"), required=("quy binh on gia",), child=True),
    _profile("investment_property_net_v2", "con lai bat dong san dau tu", ("con lai cua bat dong san dau tu", "gia tri con lai bat dong san dau tu"), required=("con lai", "bat dong san dau tu"), child=True),
    _profile("direct_fund_contribution_v2", "von gop truc tiep vao quy dau tu", ("khoan muc von gop truc tiep vao quy dau tu gia tri",), required=("von gop", "truc tiep", "quy dau tu"), child=True),
    _profile("bond_total", "tong gia tri trai phieu", ("tong trai phieu", "tong gia tri trai phieu"), required=("trai phieu",), child=True),
    _profile("issued_bond_to_broker", "trai phieu phat hanh cho cong ty chung khoan", ("trai phieu phat hanh den chung khoan thanh",), required=("trai phieu", "phat hanh"), child=True),
    _profile("management_board_income", "thu nhap ban tong giam doc va quan ly", ("thu nhap ban tong doc va quan ly", "thu nhap ban tong giam doc"), required=("thu nhap", "quan ly"), child=True),
    _profile("named_board_remuneration", "thu lao thanh vien hoi dong quan tri", ("thu lao ong nguyen hanh phuc", "thu lao ong le phuoc vu", "thu lao cua thanh vien hdqt", "thu lao chu tich hdqt"), required=("thu lao",), child=True),
    _profile("related_service_purchase", "mua dich vu voi ben lien quan", ("mua dich vu voi cong ty",), required=("mua dich vu",), child=True),
    _profile("collateral_deposit_provision", "du phong cac khoan dat coc ky quy", ("du phong cac khoan dat coc ky quy",), required=("du phong", "dat coc", "ky quy"), child=True),
    _profile("attributable_common_profit", "loi nhuan phan bo cho co dong pho thong", ("loi nhuan phan bo cho co dong so huu co phieu pho thong",), required=("loi nhuan", "phan bo", "co dong"), child=True),
    _profile("named_related_loan", "cho vay cong ty doi tac", ("cho vay nha hoa binh", "vay nha hoa binh", "cho vay doi tac doanh nghiep"), required=("cho vay",), child=True),
    _profile("foreign_currency_usd_asset", "tai san bang do la my usd", ("tai san bang do la my", "tien mat ngoai te"), required=("do la my",), child=True),
    _profile("deposit_certificate", "chung chi tien gui", (), required=("chung chi", "tien gui"), child=True),
    _profile("bank_upas_refinancing", "cho vay tai tai tro nghiep vu upas l/c", ("vay tai tai tro nghiep vu upas l/c",), required=("upas",), child=True),
    _profile("pledge_deposit_short", "ky cuoc ky quy ngan han", ("khoan ky cuoc ky quy ngan han", "cam co ky cuoc ky quy ngan han"), required=("ky cuoc", "ky quy", "ngan han"), child=True),
    _profile("profit_distribution", "loi nhuan thuan phan bo", ("loi nhuan thuan phan bo cho co dong",), required=("loi nhuan", "phan bo"), child=True),
    _profile("other_loan_receivables_long", "phai thu ve cho vay dai han", ("so du phai thu ve cho vay dai han", "gia tri thuan khoan phai thu ve cho vay dai han"), required=("phai thu", "cho vay", "dai han"), child=True),
    _profile("term_deposit_vnd", "tien gui co ky han bang vnd", ("tien gui co ky han bang vnd", "tien gui co ky han ngan han"), required=("tien gui", "ky han"), child=True),
    _profile("loan_economic_individual_domestic", "cho vay to chuc kinh te va ca nhan", ("vay cac to chuc kinh te", "cho vay cac doi tac doanh nghiep"), required=("cho vay", "to chuc"), child=True),
    _profile("other_short_advance_customer", "tien tra truoc ngan han cua khach hang", ("so du tien tra truoc ngan han cua khach hang",), required=("tra truoc", "ngan han", "khach hang"), child=True),
)
PROFILES = (*PROFILES, *_FINAL_PROFILES_A)
_BY_KEY = {profile.key: profile for profile in PROFILES}


_PROFILE_PATCH_B: tuple[MetricProfile, ...] = (
    _profile("unearned_revenue_short_v2", "doanh thu chua thuc hien ngan han", ("tong doanh thu chua thuc hien ngan han",), required=("doanh thu chua thuc hien", "ngan han"), child=True),
    _profile("investment_property_cost_v2", "nguyen gia bat dong san dau tu", (), required=("nguyen gia", "bat dong san dau tu"), child=True),
    _profile("lease_minimum_commitment_v2", "tien thue toi thieu theo hop dong thue hoat dong", ("khoan tien thue toi thieu theo hop dong thue hoat", "tong khoan tien thue toi thieu theo hop dong thue hoat dong"), required=("tien thue", "toi thieu", "hop dong"), child=True),
    _profile("receivable_progress_plan", "phai thu theo tien do ke hoach", ("phai thu theo tien do ke hoach hop dong",), required=("phai thu", "tien do", "ke hoach"), child=True),
    _profile("welfare_fund_payable", "phai tra cua quy khen thuong phuc loi", ("so tien phai tra cua quy khen thuong phuc loi",), required=("phai tra", "quy khen thuong"), child=True),
    _profile("construction_payable_short", "chi phi xay dung phai tra ngan han", ("chi phi xay dung phai tra ngan han", "khoan phai tra ngan han cho dien luc viet nam"), required=("chi phi", "xay dung", "phai tra"), child=True),
    _profile("named_related_payable", "phai tra cho cong ty lien doanh", ("phai tra cho cong ty lien doanh tnhh crown sai gon", "phai tra ngan han khac voi cong ty con"), required=("phai tra",), child=True),
    _profile("real_estate_commission_expense", "chi phi hoa hong moi gioi bat dong san", ("tong chi phi hoa hong moi gioi bat dong san",), required=("chi phi", "hoa hong", "moi gioi"), child=True),
    _profile("building_depreciation_expense", "chi phi khau hao nha cua", ("chi phi khau hao nha cua trong nam",), required=("khau hao", "nha cua"), child=True),
    _profile("gas_cylinder_expense", "chi phi vo binh gas", ("chi phi vo binh binh quan", "chi phi vo binh gas binh quan"), required=("chi phi", "vo binh"), child=True),
    _profile("real_estate_development_cost", "chi phi xay dung va phat trien bat dong san", ("chi phi xay dung va phat trien bat dong san",), required=("chi phi", "xay dung", "bat dong san"), child=True),
    _profile("raw_material_net_value", "gia tri thuan nguyen vat lieu", ("gia tri thuan cua nguyen vat lieu", "gia tri thuan nguyen lieu vat lieu"), required=("gia tri thuan", "nguyen vat lieu"), child=True),
    _profile("related_customer_receivables", "phai thu khach hang ben lien quan", ("phai thu khach hang tu cac ben lien quan",), required=("phai thu", "khach hang", "lien quan"), child=True),
    _profile("related_receivable_payable_net", "rong phai thu phai tra ben lien quan", ("so du rong khoan phai thu phai tra ngan han voi ben lien quan",), required=("phai thu", "phai tra", "lien quan"), child=True),
    _profile("bad_debt_receivables_ratio", "no kho doi tren tong phai thu khach hang", ("ty le no kho doi tren tong phai thu khach hang",), required=("no kho doi", "phai thu"), child=True),
    _profile("interest_sensitivity_gap", "chenh lech nhay cam voi lai suat noi bang", ("muc chenh nhay cam voi lai suat noi bang",), required=("nhay cam", "lai suat"), child=True),
    _profile("investment_dividend_yield", "suat sinh loi tu co tuc dau tu", ("ty suat sinh loi tu co tuc dau tu",), required=("sinh loi", "co tuc"), child=True),
    _profile("financial_debt_total", "no tai chinh", ("tong no tai chinh", "no vay tong"), required=("no tai chinh",), child=True),
    _profile("retained_profit_after_tax", "loi nhuan ke toan sau thue tndn", ("loi nhuan ke toan sau thue",), required=("loi nhuan", "sau thue"), child=True),
    _profile("short_term_advance", "tam ung ngan han", ("khoan tam ung ngan han", "tam ung cho nong dan"), required=("tam ung",), child=True),
    _profile("other_financial_activity_result", "ket qua hoat dong tai chinh rong", ("loi nhuan thuan tu hoat dong tai chinh", "lai rong tu hoat dong tai chinh"), required=("tai chinh",), child=True),
    _profile("financial_asset_total", "tong so tai san tai chinh", ("tong tai san tai chinh", "so tai san tai chinh"), required=("tai san tai chinh",), child=True),
    _profile("loan_interest_accrued", "lai cho vay chua thu duoc", ("lai cho vay chua thu duoc", "lai du thu cho vay"), required=("lai", "cho vay", "thu duoc"), child=True),
    _profile("bank_customer_loan_industry_share", "ty trong du no cho vay nganh", ("ti trong du no nganh cong nghiep che bien che tao", "ty trong du no cho vay theo nganh"), required=("du no", "nganh"), child=True),
    _profile("foreign_currency_off_balance", "ngoai te ghi nhan ngoai bang", ("ty trong ngoai te usd trong tong du luong ngoai te ghi nhan ngoai bang",), required=("ngoai te", "ngoai bang"), child=True),
    _profile("market_value_derivative", "gia tri tai san phai sinh", ("gia tri tai san phai sinh", "san phai sinh"), required=("phai sinh",), child=True),
    _profile("non_deductible_tax_expense", "chi phi khong duoc tru khi tinh thue tndn", ("chi phi khong duoc tru khi tinh thue tndn",), required=("chi phi", "khong duoc tru", "thue"), child=True),
    _profile("foreign_currency_loan_ratio", "ty trong khoan vay bang usd", ("ty trong khoan vay bang usd trong tong khoan vay dai han",), required=("khoan vay", "usd"), child=True),
    _profile("transaction_related_party_total", "tong giao dich voi ben lien quan", ("tong gia tri giao dich voi ben lien quan",), required=("giao dich", "lien quan"), child=True),
    _profile("taxes_payable_current", "thue tndn hien hanh phai nop", ("muc thue tndn hien hanh phai nop",), required=("thue tndn", "phai nop"), child=True),
)
PROFILES = (*PROFILES, *_PROFILE_PATCH_B)
_BY_KEY = {profile.key: profile for profile in PROFILES}


def primary_profile_keys(primary_texts: Iterable[str], question: str = "") -> list[str]:
    """Pick target profiles before selector/filter profiles.

    When extraction returns a generic span (``cac``, ``so du``, ``tien``), use
    the text before a selector clause as the target context.
    """
    primary = norm(" ".join(str(x or "") for x in primary_texts))
    context = norm(question)
    generic = {"", "cac", "so du", "tien", "von", "doanh thu", "moc", "so luong"}
    target_context = primary
    if primary in generic or len(primary.split()) <= 1:
        cuts = [context.find(marker) for marker in (" nam co ", " tai nam co ", " nam nao ", " trong nam co ", " trong cac nam ")]
        cuts = [cut for cut in cuts if cut > 0]
        target_context = context[:min(cuts)] if cuts else context
    candidates = []
    for profile in PROFILES:
        best_target = max((len(alias) for alias in profile.variants if alias and alias in target_context), default=0)
        best_primary = max((len(alias) for alias in profile.variants if alias and alias in primary), default=0)
        best_context = max((len(alias) for alias in profile.variants if alias and alias in context), default=0)
        required_target = sum(1 for phrase in profile.required_phrases if phrase in target_context)
        required_context = sum(1 for phrase in profile.required_phrases if phrase in context)
        if best_primary:
            score = 30000 + best_primary * 10 + required_target * 20 + int(profile.child_exact)
        elif best_target >= 8 and required_target >= min(2, len(profile.required_phrases)):
            score = 20000 + best_target * 10 + required_target * 20 + int(profile.child_exact)
        elif best_context >= 12 and required_context >= min(2, len(profile.required_phrases)):
            score = 1000 + best_context * 2 + required_context * 10 + int(profile.child_exact)
        else:
            continue
        candidates.append((score, profile.key))
    if not candidates:
        return []
    candidates.sort(reverse=True)
    best = candidates[0][0]
    return [key for score, key in candidates if score == best]
