from vifinqa.finance.metrics import (
    METRICS,
    code_expectation,
    expand_metric_variants,
    extract_metric_qualifiers,
    find_metrics,
    get_metric,
    metric_schema_score,
    metric_uses_absolute_value,
    metric_evidence_components,
    metric_keys,
)


def test_corporate_registry_has_unique_keys_and_expected_schema():
    assert len(METRICS) == len(set(METRICS))
    assert get_metric("net_revenue").codes == ("10",)
    assert get_metric("net_profit").statement == "income_statement"
    assert "chua phan phoi" in get_metric("net_profit").forbidden_phrases
    assert get_metric("cfo").statement == "cash_flow"


def test_extended_balance_sheet_metrics_use_standard_vas_codes():
    assert get_metric("supplier_prepayments_long_term").codes == ("212",)
    assert get_metric("investments_in_associates").codes == ("252",)
    assert get_metric("other_equity_investments").codes == ("253",)
    assert get_metric("other_receivables_short_term").codes == ("136",)
    assert get_metric("total_capital").codes == ("440",)


def test_exact_lookup_v26_dictionary_recognizes_literal_note_metrics():
    cases = {
        "so du quy binh on gia xang dau": "fuel_price_stabilization_fund",
        "tong so du phai thu khac ngan han": "other_receivables_short_term",
        "tong so luong co phan": "total_shares",
        "gia tri tai san tai chinh chiu lai suat co dinh":
            "fixed_rate_financial_assets",
        "du thu co tuc loi nhuan duoc chia":
            "accrued_dividend_profit_receivable",
        "tong gia goc no phai thu": "gross_receivables",
        "tien tra truoc ngan han cua khach hang":
            "buyer_advances_short_term",
        "gia tri con lai cua tai san vo hinh": "intangible_fixed_assets",
        "gia tri hop ly cua tai san tai chinh fvtpl":
            "financial_assets_fvtpl_fair_value",
        "so du chi phi cho phan bo": "deferred_allocation_expense",
        "loi nhuan tren moi co phieu dang luu hanh": "basic_eps",
        "tong von gop": "contributed_capital",
    }

    for phrase, expected in cases.items():
        assert metric_keys([phrase], False) == [expected]

    assert get_metric("total_shares").column_phrases == (
        "so luong co phan",)
    assert get_metric("gross_receivables").column_phrases == ("gia goc",)


def test_equity_resolves_to_the_aggregate_vas_400_line():
    assert get_metric("equity").codes == ("400",)
    codes, mismatch = code_expectation(
        ["von chu so huu"], "VON CHU SO HUU (400 = 410 + 430)")
    assert codes == {"400"}
    assert not mismatch


def test_alias_maps_to_canonical_line_item():
    matches = find_metrics("Lợi nhuận thuần sau thuế năm 2024")
    assert [match.metric.key for match in matches] == ["net_profit"]


def test_net_profit_accepts_statement_wording_profit_for_the_year():
    metric = get_metric("net_profit")
    assert "loi nhuan thuan trong nam" in metric.variants
    assert "loi nhuan thuan trong nam" in metric.required_phrases


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


def test_intangible_fixed_assets_do_not_collapse_to_fixed_assets_parent():
    phrase = "gia tri con lai cua tai san co dinh vo hinh"

    assert metric_keys([phrase], False) == ["intangible_fixed_assets"]
    assert metric_schema_score(
        [phrase], "Tai san co dinh vo hinh", phrase) > 0
    assert metric_schema_score(
        [phrase], "X Tai san co dinh", phrase) < 0


def test_note_dictionary_v3_recognizes_exact_year_ranking_metrics():
    cases = {
        "so du cac khoan phai thu ben ngoai": "external_receivables",
        "gia tri hang hoa ton kho cuoi ky": "merchandise_inventory",
        "tong doanh thu chua thuc hien ngan han": "unearned_revenue_short_term",
        "tong chi phi hoa hong moi gioi bat dong san":
            "real_estate_brokerage_expense",
        "chi phi xay dung phai tra ngan han":
            "construction_payables_short_term",
        "doanh thu thuan tu san pham khi LPG": "lpg_revenue",
        "tong doanh thu cung cap dich vu cho cac ben lien quan":
            "related_party_service_revenue",
        "phai tra cho Cong ty Lien doanh TNHH Crown Sai Gon":
            "crown_saigon_trade_payable",
    }

    for phrase, expected in cases.items():
        assert metric_keys([phrase], False) == [expected]


def test_compositional_v3_note_qualifiers_are_distinct_metrics():
    keys = metric_keys([
        "vay ngắn hạn phải trả các bên liên quan",
        "vay ngắn hạn ngân hàng",
        "phải thu ngắn hạn khác từ các bên liên quan",
    ], expand_derived=False)

    assert "related_party_short_term_borrowings" in keys
    assert "bank_short_term_borrowings" in keys
    assert "related_party_other_receivables_short_term" in keys


def test_compositional_v3_derived_metrics_expand_operands():
    keys = metric_keys([
        "365 lần hàng tồn kho bình quân trên giá vốn hàng bán",
        "tỷ lệ giữa chi phí lãi vay và lợi nhuận trước thuế",
    ])

    assert "inventory" in keys
    assert "cost_of_goods_sold" in keys
    assert "interest_expense" in keys
    assert "pretax_profit" in keys


def test_inventory_days_declares_opening_and_closing_inventory_periods():
    assert metric_evidence_components("inventory_days") == (
        ("inventory", -1),
        ("inventory", 0),
        ("cost_of_goods_sold", 0),
    )


def test_v7_period_formulas_declare_all_atomic_periods():
    assert metric_evidence_components("accrual_average_assets") == (
        ("net_profit", 0),
        ("cfo", 0),
        ("total_assets", -1),
        ("total_assets", 0),
    )
    assert metric_evidence_components("operating_leverage") == (
        ("operating_profit", -1),
        ("operating_profit", 0),
        ("net_revenue", -1),
        ("net_revenue", 0),
    )


def test_v9_note_parent_child_metrics_remain_distinct():
    cases = {
        "du no cho vay nganh bat dong san": "real_estate_customer_loans",
        "vay bang usd": "usd_long_term_borrowings",
        "chi phi lai tien gui": "deposit_interest_expense",
        "du phong chung trong du phong rui ro cho vay khach hang":
            "general_customer_loan_provision_balance",
    }

    for phrase, expected in cases.items():
        assert metric_keys([phrase], False) == [expected]


def test_v9_derived_note_ratios_expand_exact_parent_and_child():
    assert metric_evidence_components("real_estate_customer_loan_share") == (
        ("real_estate_customer_loans", 0),
        ("customer_loans", 0),
    )
    assert metric_evidence_components(
        "general_provision_total_loan_provision_share") == (
            ("general_customer_loan_provision_balance", 0),
            ("customer_loan_provision_balance", 0),
        )
    assert metric_evidence_components(
        "borrowings_cash_and_deposits_ratio") == (
            ("borrowings_total", 0),
            ("cash_on_hand", 0),
            ("bank_deposits", 0),
        )


def test_v16_note_temporal_metrics_expand_exact_operands():
    assert metric_keys(["ty le bao phu no xau"]) == [
        "customer_loan_provision_balance", "substandard_loans",
        "doubtful_loans", "loss_loans",
    ]
    assert metric_keys(["doanh thu chua thuc hien cuoi ky"]) == [
        "unearned_revenue_short_term", "unearned_revenue_long_term",
    ]
    assert metric_keys(["ty le hao mon luy ke tscd huu hinh"]) == [
        "tangible_fixed_assets_accumulated_depreciation",
        "tangible_fixed_assets_cost",
    ]
    assert metric_keys(
        ["phai tra ngan han khac voi cong ty con"], expand_derived=False
    ) == ["subsidiary_other_payables_short_term"]


def test_v17_note_ratio_metrics_expand_exact_parent_and_child():
    cases = {
        "ty trong cho vay ngan han trong tong du no cho vay khach hang": (
            "customer_loans_short_term", "customer_loans"),
        "ty trong chung chi tien gui co ky han duoi 12 thang": (
            "certificate_deposits_under_12_months",
            "certificates_of_deposit_total"),
        "ty trong doanh thu tu khu vuc lao so voi tong doanh thu toan cong ty": (
            "laos_geographic_revenue", "geographic_revenue_total"),
        "ty trong ngoai te usd trong tong du luong ngoai te ghi nhan ngoai bang can doi ke toan": (
            "off_balance_usd_balance", "off_balance_foreign_currency_total"),
        "ty trong gia von cho thue dai han dat va co so ha tang tren tong gia von hang ban va dich vu cung cap": (
            "land_infrastructure_rental_cogs", "cost_of_goods_sold"),
    }
    for phrase, expected in cases.items():
        root = metric_keys([phrase], expand_derived=False)
        assert len(root) == 1
        assert tuple(key for key, _ in metric_evidence_components(root[0])) == expected


def test_note_row_aliases_do_not_become_generic_question_aliases():
    assert metric_keys(["hang hoa"], False) == []
    assert metric_keys(["chi phi xay dung"], False) == []
    assert "hang hoa" in get_metric("merchandise_inventory").row_variants
    assert "chi phi xay dung" in get_metric(
        "construction_payables_short_term").row_variants


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


def test_operating_lease_future_receipts_map_to_schedule_metric():
    assert metric_keys([
        "tien thue phai thu trong tuong lai den han duoi 1 nam"
    ], False) == ["operating_lease_commitments"]
