from types import SimpleNamespace

from vifinqa.retrieval.retrieve import _apply_quota, _retrieval_metric_variants
from vifinqa.retrieval.shortlist import (candidate_matches_metric,
                                         candidate_matches_requirement,
                                         requirement_linking_variants,
                                         requirement_specificity_key)
from vifinqa.router.evidence import build_evidence_requirements, evidence_coverage


def _route(question, metric="doanh thu thuan", variants=None, facts=None):
    return SimpleNamespace(
        question=question,
        metric_norm=metric,
        metric_variants=variants or [metric],
        plan={"facts": facts or []},
    )


def test_quick_ratio_retrieval_variants_include_formula_components():
    route = _route(
        "Có bao nhiêu doanh nghiệp vừa có hệ số thanh toán nhanh lớn hơn 1 lần?"
    )

    variants = _retrieval_metric_variants(route)

    assert "tai san ngan han" in variants
    assert "hang ton kho" in variants
    assert "no ngan han" in variants


def test_debt_to_equity_retrieval_variants_include_formula_components():
    route = _route(
        "Doanh nghiệp nào có hệ số nợ phải trả trên vốn chủ sở hữu nhỏ hơn 1 lần?"
    )

    variants = _retrieval_metric_variants(route)

    assert "no phai tra" in variants
    assert "von chu so huu" in variants


def test_retrieval_variants_include_plan_fact_metrics():
    route = _route(
        "Tính tỷ lệ CFO trên doanh thu thuần của VNM",
        facts=[
            {"ticker": "VNM", "year": 2024, "doc_type": "consolidated",
             "metric": "luu chuyen tien thuan tu hoat dong kinh doanh"},
            {"ticker": "VNM", "year": 2024, "doc_type": "consolidated",
             "metric": "doanh thu thuan"},
        ],
    )

    variants = _retrieval_metric_variants(route)

    assert "luu chuyen tien thuan tu hoat dong kinh doanh" in variants
    assert "doanh thu thuan" in variants


def test_nested_ranking_builds_every_entity_operand_requirement():
    question = (
        "Năm 2016, hệ số thanh toán hiện hành của doanh nghiệp có hệ số "
        "nợ phải trả trên vốn chủ sở hữu cao nhất là bao nhiêu lần?"
    )

    requirements = build_evidence_requirements(
        question,
        ["VNM", "DLG", "MBS"],
        [2016],
        "consolidated",
    )

    assert len(requirements) == 12
    keys = {requirement.metric_key for requirement in requirements}
    assert keys == {"current_assets", "current_liabilities", "liabilities", "equity"}


def test_inventory_days_requires_prior_year_opening_inventory():
    requirements = build_evidence_requirements(
        "Mức tăng của tỷ lệ 365 lần hàng tồn kho bình quân đầu kỳ và cuối kỳ "
        "trên giá vốn hàng bán từ năm 2021 đến 2022",
        ["HPG"], [2021, 2022], "consolidated",
    )
    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert ("inventory", 2020) in periods
    assert ("inventory", 2021) in periods
    assert ("inventory", 2022) in periods
    assert ("cost_of_goods_sold", 2020) not in periods
    assert ("cost_of_goods_sold", 2021) in periods
    assert ("cost_of_goods_sold", 2022) in periods


def test_quota_prefers_best_exact_table_for_each_requirement_when_budget_allows():
    route = SimpleNamespace(
        plan={"facts": [{}, {}]},
        evidence_requirements=[
            {"requirement_id": "AAA|2024|gross_profit"},
            {"requirement_id": "AAA|2024|net_revenue"},
        ],
    )
    candidates = [
        {
            "report_id": "AAA_2024", "table_pos": 1, "score": 10.0,
            "requirement_hits": [
                "AAA|2024|gross_profit", "AAA|2024|net_revenue"
            ],
            "requirement_scores": {
                "AAA|2024|gross_profit": 80.0,
                "AAA|2024|net_revenue": 80.0,
            },
        },
        {
            "report_id": "AAA_2024", "table_pos": 2, "score": 100.0,
            "requirement_hits": ["AAA|2024|gross_profit"],
            "requirement_scores": {"AAA|2024|gross_profit": 95.0},
        },
        {
            "report_id": "AAA_2024", "table_pos": 3, "score": 90.0,
            "requirement_hits": ["AAA|2024|net_revenue"],
            "requirement_scores": {"AAA|2024|net_revenue": 95.0},
        },
    ]

    selected = _apply_quota(candidates, route, depth=2)

    assert [candidate["table_pos"] for candidate in selected] == [2, 3]
    assert evidence_coverage(route.evidence_requirements, selected)["complete"]


def test_requirement_match_rejects_lexically_similar_wrong_metric():
    candidate = SimpleNamespace(label="Phải trả ngắn hạn khác", code="319")

    assert not candidate_matches_metric(candidate, "current_assets")


def test_requirement_preserves_and_enforces_named_counterparty():
    requirements = build_evidence_requirements(
        "Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai của HNG",
        ["HNG"], [2017], "separate",
        ["vay dai han voi hoang anh gia lai"],
    )
    requirement = requirements[0].to_dict()
    parent = SimpleNamespace(label="Vay dài hạn", code="")
    child = SimpleNamespace(
        label="Công ty Cổ phần HoàngAnh Gia Lai Công ty mẹ Vay dài hạn", code="")

    assert requirement["metric_variants"][0] == "vay dai han voi hoang anh gia lai"
    assert not candidate_matches_requirement(parent, requirement)
    assert candidate_matches_requirement(child, requirement)


def test_requirement_cache_fingerprint_only_tracks_named_detail():
    base = {"metric_key": "borrowings_long_term"}
    hag = {**base, "metric_variants": ["vay dai han voi hoang anh gia lai"]}
    alpha = {**base, "metric_variants": ["vay dai han voi cong ty alpha"]}
    generic_a = {**base, "metric_variants": ["vay dai han cuoi nam 2024"]}
    generic_b = {**base, "metric_variants": ["cac khoan vay dai han"]}

    assert requirement_specificity_key(hag) != requirement_specificity_key(alpha)
    assert requirement_specificity_key(generic_a) == ()
    assert requirement_specificity_key(generic_b) == ()


def test_generic_requirement_linking_drops_noisy_source_phrase():
    requirement = {
        "metric_key": "voting_rate",
        "metric_variants": [
            "tong so phai tra sau 12 thang cua cong ty co ty le quyen bieu quyet",
            "ty le quyen bieu quyet",
        ],
    }

    variants = requirement_linking_variants(requirement)

    assert variants[0] == "ty le quyen bieu quyet"
    assert all("phai tra sau 12 thang" not in value for value in variants)


def test_named_detail_requirement_linking_keeps_source_phrase():
    requirement = {
        "metric_key": "borrowings_long_term",
        "metric_variants": [
            "vay dai han voi hoang anh gia lai",
            "vay dai han",
        ],
    }

    assert requirement_linking_variants(requirement)[0] == (
        "vay dai han voi hoang anh gia lai"
    )
