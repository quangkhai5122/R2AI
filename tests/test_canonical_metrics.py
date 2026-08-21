from vifinqa.finance.metrics import (
    METRICS,
    code_expectation,
    expand_metric_variants,
    extract_metric_qualifiers,
    find_metrics,
    get_metric,
    metric_schema_score,
    metric_uses_absolute_value,
    metric_keys,
)


def test_corporate_registry_has_unique_keys_and_expected_schema():
    assert len(METRICS) == len(set(METRICS))
    assert get_metric("net_revenue").codes == ("10",)
    assert get_metric("net_profit").statement == "income_statement"
    assert "chua phan phoi" in get_metric("net_profit").forbidden_phrases
    assert get_metric("cfo").statement == "cash_flow"


def test_equity_resolves_to_the_aggregate_vas_400_line():
    assert get_metric("equity").codes == ("400",)
    codes, mismatch = code_expectation(
        ["von chu so huu"], "VON CHU SO HUU (400 = 410 + 430)")
    assert codes == {"400"}
    assert not mismatch


def test_alias_maps_to_canonical_line_item():
    matches = find_metrics("Lợi nhuận thuần sau thuế năm 2024")
    assert [match.metric.key for match in matches] == ["net_profit"]


def test_derived_metric_expands_to_atomic_components():
    variants = expand_metric_variants(["bien loi nhuan rong"])
    assert "loi nhuan sau thue" in variants
    assert "doanh thu thuan" in variants
    assert metric_keys(["bien loi nhuan rong"]) == ["net_profit", "net_revenue"]


def test_quick_ratio_expansion_uses_shared_components():
    variants = expand_metric_variants(["he so thanh toan nhanh"])
    assert {"tai san ngan han", "hang ton kho", "no ngan han"} <= set(variants)


def test_vas_code_expectation_uses_label_specific_metric():
    codes, mismatch = code_expectation(
        ["doanh thu thuan"], "Doanh thu thuần về bán hàng và cung cấp dịch vụ")
    assert codes == {"10"}
    assert not mismatch


def test_retained_earnings_is_not_generic_net_profit():
    codes, mismatch = code_expectation(
        ["loi nhuan sau thue"], "Lợi nhuận sau thuế chưa phân phối")
    assert not codes
    assert mismatch


def test_attributable_profit_requires_explicit_qualifier():
    _codes, mismatch = code_expectation(
        ["loi nhuan sau thue"], "Lợi nhuận sau thuế thuộc về cổ đông công ty mẹ")
    assert mismatch

    codes, explicit_mismatch = code_expectation(
        ["loi nhuan sau thue thuoc ve co dong"],
        "Lợi nhuận sau thuế thuộc về cổ đông công ty mẹ")
    assert codes == {"60"}
    assert not explicit_mismatch

    _codes, generic_mismatch = code_expectation(
        ["loi nhuan sau thue thuoc ve co dong"], "Loi nhuan sau thue")
    assert generic_mismatch


def test_qualified_note_metric_does_not_expand_to_aggregate():
    phrase = "khoan no ngan han voi ben lien quan"
    variants = expand_metric_variants([phrase])
    assert variants == [phrase]

    _codes, mismatch = code_expectation([phrase], "I. No ngan han")
    assert mismatch


def test_interest_coverage_is_not_blocked_by_thanh_toan_word():
    variants = expand_metric_variants(["he so kha nang thanh toan lai vay"])
    assert "loi nhuan truoc thue" in variants
    assert "chi phi lai vay" in variants


def test_bank_parent_and_child_metrics_are_distinct():
    assert metric_keys(["tien gui va vay cac TCTD khac"], False) == [
        "interbank_funding_total"]
    assert metric_keys(["vay cac TCTD khac"], False) == [
        "interbank_borrowings"]
    assert metric_schema_score(
        ["vay cac TCTD khac"], "Tien gui va vay cac TCTD khac") < -20
    assert metric_schema_score(
        ["vay cac TCTD khac"], "Vay cac TCTD khac") > 10


def test_provision_stock_and_flow_metrics_are_distinct():
    assert metric_keys(["so du du phong rui ro cho vay khach hang"], False) == [
        "customer_loan_provision_balance"]
    assert metric_keys(["trich lap du phong rui ro cho vay khach hang"], False) == [
        "customer_loan_provision_expense"]


def test_structured_qualifiers_cover_requested_dimensions():
    q = extract_metric_qualifiers(
        "Tong gia tri thuan vay dai han dau nam, tinh theo tri tuyet doi")
    assert q.stock_flow == "stock"
    assert q.gross_net == "net"
    assert q.maturity == "long"
    assert q.period == "opening"
    assert q.granularity == "aggregate"


def test_expense_metric_uses_absolute_value():
    assert metric_uses_absolute_value("chi phi lai vay", ["interest_expense"])
