from vifinqa.finance.metrics import (
    METRICS,
    code_expectation,
    expand_metric_variants,
    find_metrics,
    get_metric,
    metric_keys,
)


def test_corporate_registry_has_unique_keys_and_expected_schema():
    assert len(METRICS) == len(set(METRICS))
    assert get_metric("net_revenue").codes == ("10",)
    assert get_metric("net_profit").statement == "income_statement"
    assert "chua phan phoi" in get_metric("net_profit").forbidden_phrases
    assert get_metric("cfo").statement == "cash_flow"


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
