import io

import pandas as pd

from vifinqa.codegen.exact_cell import resolve_exact_cell
from vifinqa.codegen.executor import run_code


def _table(rows, context):
    return {"var": "df1", "report_id": "AAA_financial_statements_2024_separate",
            "table_pos": 1, "report_year": 2024,
            "context": context,
            "csv_text": pd.DataFrame(rows).to_csv(index=False)}


def _row(label, col, col_name, value, code=""):
    return {"row": 1, "label": label, "code": code, "col": col,
            "col_name": col_name, "value": value, "unit_scale": 1.0}


def test_exact_cell_matches_entity_row_and_metric_column():
    route = {"question": "Tỷ lệ biểu quyết của Xí nghiệp Liên doanh Visorutex là bao nhiêu %?",
             "metric_variants": ["bieu quyet cua xi nghiep lien doanh visorutex",
                                  "quyen bieu quyet"],
             "metric_profile_keys": ["voting_rights"], "output_type": "percent",
             "unit_scale": 1.0, "years": [2024], "plan": {"op": "lookup"}}
    table = _table([
        _row("- Xí nghiệp Liên doanh Visorutex", 2, "Tỷ lệ lợi ích", 27.78),
        _row("- Xí nghiệp Liên doanh Visorutex", 3, "Tỷ lệ biểu quyết", 31.25),
    ], "Đầu tư vào Công ty liên doanh liên kết thông tin chi tiết")
    cell = resolve_exact_cell(route, [table])
    assert cell is not None
    assert cell.col_name == "Tỷ lệ biểu quyết"
    assert cell.value == 31.25


def test_exact_cell_matches_metric_column_in_asset_note():
    route = {"question": "Giá trị còn lại của bất động sản đầu tư là bao nhiêu tỷ đồng?",
             "metric_variants": ["gia tri con lai bat dong san dau tu"],
             "metric_profile_keys": ["investment_property_net_v2"],
             "output_type": "number", "unit_scale": 1e9, "years": [2024],
             "plan": {"op": "lookup"}}
    table = _table([
        _row("Số cuối năm", 1, "Nguyên giá", 400e9),
        _row("Số cuối năm", 2, "Hao mòn lũy kế", 40e9),
        _row("Số cuối năm", 3, "Giá trị còn lại", 360e9),
    ], "10. Bất động sản đầu tư")
    cell = resolve_exact_cell(route, [table])
    assert cell is not None
    assert cell.col_name == "Giá trị còn lại"
    assert cell.value == 360e9


def test_exact_cell_rejects_missing_canonical_profile():
    route = {"question": "Một chỉ tiêu là bao nhiêu?", "metric_variants": ["chi tieu"],
             "output_type": "number", "unit_scale": 1.0, "years": [2024]}
    table = _table([_row("Một chỉ tiêu", 1, "2024", 1)], "note")
    assert resolve_exact_cell(route, [table]) is None


def test_exact_cell_rejects_nested_route():
    route = {"question": "Năm có tỷ lệ biểu quyết cao nhất là năm nào?",
             "metric_variants": ["ty le bieu quyet"],
             "metric_profile_keys": ["voting_rights"], "output_type": "year",
             "unit_scale": 1.0, "years": [2023, 2024],
             "plan": {"op": "ranking"}}
    table = _table([_row("Công ty A", 1, "Tỷ lệ biểu quyết", 51.0)], "Đầu tư")
    assert resolve_exact_cell(route, [table]) is None


def test_exact_cell_matches_tax_payable_date_column():
    route = {"question": "Số dư thuế thu nhập doanh nghiệp phải trả cuối năm 2024",
             "metric_variants": ["thue thu nhap phai tra", "thue thu nhap doanh nghiep"],
             "metric_profile_keys": ["tax_expense_income", "current_income_tax_payable"],
             "output_type": "number", "unit_scale": 1e6, "years": [2024],
             "plan": {"op": "lookup"}}
    table = _table([
        _row("Thuế thu nhập doanh nghiệp", 1, "Số phát sinh trong năm", 700),
        _row("Thuế thu nhập doanh nghiệp", 2, "31/12/2024", 250),
    ], "Thuyết minh có chi phí thuế thu nhập trong năm và số dư phải trả")
    cell = resolve_exact_cell(route, [table])
    assert cell is not None
    assert cell.col_name == "31/12/2024"
    assert cell.value == 250


def test_exact_cell_rejects_wrong_explicit_year():
    route = {"question": "Chứng khoán đầu tư sẵn sàng để bán cuối năm 2017",
             "metric_variants": ["chung khoan dau tu san sang de ban"],
             "metric_profile_keys": ["afs_securities"], "output_type": "number",
             "unit_scale": 1e6, "years": [2017], "plan": {"op": "lookup"}}
    table = _table([
        _row("Chứng khoán đầu tư sẵn sàng để bán", 1, "31/12/2016", 100),
    ], "Chứng khoán đầu tư sẵn sàng để bán")
    assert resolve_exact_cell(route, [table]) is None


def test_exact_cell_rejects_prepayment_for_payable_question():
    route = {"question": "Số dư phải trả nhà cung cấp cuối năm 2024",
             "metric_variants": ["phai tra nha cung cap"],
             "metric_profile_keys": ["related_payable_third_party"],
             "output_type": "number", "unit_scale": 1e9, "years": [2024],
             "plan": {"op": "lookup"}}
    table = _table([
        _row("Trả trước nhà cung cấp A", 1, "31/12/2024", 100),
    ], "Phải trả và trả trước nhà cung cấp")
    assert resolve_exact_cell(route, [table]) is None
