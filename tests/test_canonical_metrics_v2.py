from vifinqa.finance.metrics_v2 import (
    PROFILE_VERSION,
    best_row_profile,
    expand_variants_v2,
    infer_profiles,
    profile_keys,
    qualifier_flags,
    row_profile_match,
)
from vifinqa.retrieval.shortlist import build_shortlist


def _table(rows):
    import pandas as pd
    return [{
        "var": "df1",
        "report_id": "ACB_financial_statements_2024_consolidated",
        "table_pos": 10,
        "report_year": 2024,
        "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }]


def _row(row, label, value, code="", col=1, col_name="2024", unit=1e6):
    return {"row": row, "label": label, "code": code, "col": col,
            "col_name": col_name, "value": value, "unit_scale": unit}


def test_v2_profile_version_is_explicit():
    assert PROFILE_VERSION == "canonical_metric_v2_2026_08_18f"

def test_bank_specific_provision_suppresses_parent_profile():
    profiles = infer_profiles([
        "chi phi trich lap du phong cu the cho vay khach hang"
    ])
    keys = [profile.key for profile in profiles]
    assert "bank_specific_loan_provision" in keys
    assert "bank_customer_loan_provision" not in keys


def test_bank_tctd_variants_expand_abbreviation_and_full_name():
    variants = expand_variants_v2(["tien gui tai cac tctd khac"])
    assert "tien gui cac to chuc tin dung khac" in variants
    assert profile_keys(["tien gui tai cac tctd khac"]) == [
        "bank_other_tctd_deposits"
    ]


def test_qualifier_axes_cover_requested_contract():
    flags = qualifier_flags(
        "Giá trị tuyệt đối số dư cuối năm khoản vay dài hạn gộp"
    )
    assert flags["stock_flow"] == "stock"
    assert flags["gross_net"] == "gross"
    assert flags["term"] == "long"
    assert flags["period"] == "closing"
    assert flags["sign"] == "absolute"


def test_child_profile_rejects_broad_parent_row():
    profile = infer_profiles(["du phong cu the cho vay khach hang"])[0]
    ok_parent, _, _ = row_profile_match(
        profile, "Dự phòng cụ thể cho vay khách hàng",
        "Dự phòng rủi ro cho vay khách hàng",
    )
    ok_child, bonus, _ = row_profile_match(
        profile, "Dự phòng cụ thể cho vay khách hàng",
        "Trong đó: Dự phòng cụ thể cho vay khách hàng",
    )
    assert not ok_parent
    assert ok_child
    assert bonus > 0


def test_shortlist_profile_gate_selects_child_not_parent():
    route = {
        "question": "Số dư dự phòng cụ thể cho vay khách hàng cuối năm 2024",
        "metric_profile_keys": ["bank_specific_loan_provision"],
    }
    tables = _table([
        _row(1, "Dự phòng rủi ro cho vay khách hàng", 100),
        _row(2, "Trong đó: Dự phòng cụ thể cho vay khách hàng", 60),
        _row(3, "Trong đó: Dự phòng chung cho vay khách hàng", 40),
    ])
    cands = build_shortlist(
        tables, ["du phong cu the cho vay khach hang"], [2024],
        route=route, min_score=35,
    )
    assert cands
    assert "cụ thể" in cands[0].label


def test_stock_question_rejects_flow_expense_row():
    route = {
        "question": "Số dư dự phòng rủi ro cho vay khách hàng cuối năm 2024",
        "metric_profile_keys": ["bank_customer_loan_provision"],
    }
    profile, _, reason = best_row_profile(
        route, "Chi phí dự phòng rủi ro tín dụng trong năm", ""
    )
    assert profile is None
    assert "stock_flow" in reason or "canonical" in reason

def test_financial_asset_profile_requires_column_phrase():
    from vifinqa.finance.metrics_v2 import row_profile_match
    profile = infer_profiles(["cho vay va phai thu cua tai san tai chinh trong hoat dong cho vay khach hang"])[0]
    ok_row, _, _ = row_profile_match(
        profile,
        "cho vay va phai thu cua tai san tai chinh trong hoat dong cho vay khach hang",
        "Cho vay khách hàng",
        col_name="Số cuối năm",
    )
    ok_cell, bonus, _ = row_profile_match(
        profile,
        "cho vay va phai thu cua tai san tai chinh trong hoat dong cho vay khach hang",
        "Cho vay khách hàng",
        col_name="Cho vay và phải thu Triệu đồng",
    )
    assert not ok_row
    assert ok_cell
    assert bonus > 0

def test_note_profile_groups_cover_unresolved_phrases():
    assert "penalty_expense" in profile_keys(["Chi phí phạt"])
    assert "ordinary_bonds" in profile_keys(["Tổng trái phiếu thường"])
    assert "cash_national_bank" in profile_keys(["Tiền gửi tại Ngân hàng Nhà nước Việt Nam"])
    assert "economic_interest" in profile_keys(["Tỷ lệ lợi ích kinh tế"])
    assert "vat_payable" in profile_keys(["Thuế giá trị gia tăng phải nộp"])


def test_bank_and_note_child_profiles_do_not_use_generic_parent_aliases():
    assert profile_keys(["Dư nợ đủ tiêu chuẩn"]) == ["standard_loans"]
    assert profile_keys(["Tổng dư nợ cho vay"]) == ["total_customer_loans"]
    assert "financial_obligations" in profile_keys(["Nghĩa vụ nợ tài chính"])



def test_stock_profiles_do_not_confuse_cost_labels_with_flow():
    from vifinqa.finance.metrics_v2 import _BY_KEY
    cases = [
        ("other_prepaid_expense", "Chi phí trả trước ngắn hạn khác"),
        ("construction_in_progress_detail", "Chi phí xây dựng cơ bản dở dang"),
        ("investment_at_cost", "Giá gốc chứng khoán kinh doanh"),
        ("afs_securities", "Chứng khoán đầu tư sẵn sàng để bán"),
        ("current_income_tax_payable", "Thuế thu nhập doanh nghiệp phải nộp"),
    ]
    for key, label in cases:
        profile = _BY_KEY[key]
        ok, _bonus, reason = row_profile_match(
            profile, label, label, "", "Số cuối năm"
        )
        assert ok, (key, reason)


def test_flow_expense_still_rejected_by_stock_profiles():
    from vifinqa.finance.metrics_v2 import _BY_KEY
    profile = _BY_KEY["other_prepaid_expense"]
    ok, _bonus, reason = row_profile_match(
        profile, "Số dư chi phí trả trước cuối năm",
        "Chi phí trả trước phát sinh trong năm", "", "Năm nay"
    )
    assert not ok
    assert "stock_flow" in reason


def test_context_does_not_contaminate_row_stock_flow():
    from vifinqa.finance.metrics_v2 import _BY_KEY, best_row_profile
    route = {
        "question": "Số dư thuế thu nhập doanh nghiệp phải trả cuối năm 2020",
        "metric_profile_keys": ["current_income_tax_payable", "tax_expense_income"],
    }
    profile, _bonus, reason = best_row_profile(
        route,
        "Thuế thu nhập doanh nghiệp phải nộp 31/12/2020",
        "",
        "31/12/2020",
        qualifier_text="Thuế thu nhập doanh nghiệp 31/12/2020",
    )
    assert profile is not None, reason
    assert profile.key == "current_income_tax_payable"
