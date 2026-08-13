from types import SimpleNamespace

from vifinqa.retrieval.retrieve import _retrieval_metric_variants


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
