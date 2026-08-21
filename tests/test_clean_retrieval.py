from types import SimpleNamespace

from vifinqa.clean.retrieval import CleanRetrievalConfig, canonicalize_route


def _route(question, metric, facts=None):
    return SimpleNamespace(
        question=question,
        metric_norm=metric,
        metric_variants=[metric],
        plan={"facts": facts or []},
    )


def test_clean_config_is_source_only_and_row_rerank_defaults_off():
    config = CleanRetrievalConfig()
    config.validate()
    assert config.canonical_registry
    assert config.component_expansion
    assert not config.row_rerank


def test_quick_ratio_expands_to_atomic_components():
    route = _route("Hệ số thanh toán nhanh lớn hơn 1 lần", "he so thanh toan nhanh")
    variants, keys, _qualifiers = canonicalize_route(route)
    assert "current_assets" in keys or "quick_ratio" in keys
    assert "tai san ngan han" in variants
    assert "hang ton kho" in variants
    assert "no ngan han" in variants


def test_plan_fact_metrics_are_included_without_question_ids():
    route = _route(
        "Tính CFO trên doanh thu thuần",
        "cfo tren doanh thu thuan",
        facts=[{"metric": "luu chuyen tien thuan tu hoat dong kinh doanh"},
               {"metric": "doanh thu thuan"}],
    )
    variants, _keys, _qualifiers = canonicalize_route(route)
    assert "luu chuyen tien thuan tu hoat dong kinh doanh" in variants
    assert "doanh thu thuan" in variants

