from vifinqa.codegen.atomic_slots import plan_atomic_slots


def _route(tickers=("AAA",), years=(2021,), op="ranking", metric="metric"):
    facts = [
        {"ticker": ticker, "year": year, "doc_type": "consolidated",
         "metric": metric, "role": "value"}
        for ticker in tickers for year in years
    ]
    return {
        "tickers": list(tickers), "years": list(years),
        "doc_type": "consolidated", "metric_norm": metric,
        "plan": {"op": op, "facts": facts},
    }


def test_quick_ratio_expands_three_role_aware_components_per_period():
    slots, trace = plan_atomic_slots(
        "Năm nào có hệ số thanh toán nhanh thấp nhất?",
        _route(years=(2021, 2022)),
    )
    assert len(slots) == 6
    assert {s["metric"] for s in slots} == {
        "tai san ngan han", "hang ton kho", "no ngan han",
    }
    assert trace["families"] == ["quick_ratio"]
    assert trace["roles"] == {"rank": 4, "denominator": 2}


def test_nested_rank_then_next_year_projection_adds_both_formula_families():
    slots, trace = plan_atomic_slots(
        "Hệ số dòng tiền hoạt động trên nợ ngắn hạn vào năm sau năm có "
        "hệ số thanh toán nhanh thấp nhất giai đoạn 2021-2022",
        _route(years=(2021, 2022)),
    )
    assert trace["families"] == [
        "quick_ratio", "operating_cashflow_to_current_debt",
    ]
    projected = [s for s in slots if s["role"] == "project"]
    assert {s["year"] for s in projected} == {2022, 2023}
    assert all(s["period_role"] == "next_period" for s in projected)


def test_multiple_entities_expand_de_and_interest_coverage_components():
    slots, trace = plan_atomic_slots(
        "Chênh lệch hệ số khả năng thanh toán lãi vay giữa nhóm có tỷ số D/E cao",
        _route(tickers=("AAA", "BBB"), op="difference"),
    )
    assert set(trace["families"]) == {"debt_to_equity", "interest_coverage"}
    assert len(slots) == 8
    assert {s["metric"] for s in slots} == {
        "no phai tra", "von chu so huu", "loi nhuan truoc thue", "chi phi lai vay",
    }


def test_plain_count_changes_value_role_to_filter_without_metric_rewrite():
    slots, trace = plan_atomic_slots(
        "Có bao nhiêu công ty có lưu chuyển tiền thuần dương?",
        _route(tickers=("AAA", "BBB"), op="count", metric="luu chuyen tien thuan"),
    )
    assert [s["role"] for s in slots] == ["filter", "filter"]
    assert trace["families"] == []


def test_beginning_period_role_is_preserved_for_plain_lookup():
    slots, trace = plan_atomic_slots(
        "Số dư đầu năm 2024 là bao nhiêu?", _route(op="lookup"),
    )
    assert slots[0]["period_role"] == "beginning"
    assert trace["period_roles"] == {"beginning": 1}

