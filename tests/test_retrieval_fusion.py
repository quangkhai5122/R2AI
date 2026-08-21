from types import SimpleNamespace

import unittest

from vifinqa.retrieval.fusion import rank_positions, reciprocal_rank_fusion
from vifinqa.retrieval.retrieve import (
    _candidate_record,
    _dense_table_scores,
    _load_route_source,
    _ranked_mapping,
    _schema_table_scores,
    retrieve_for_route,
)


class _FakeEncoder:
    def similarity(self, queries, labels):
        assert "doanh thu thuan" in queries
        return {
            label: (0.93 if "revenue" in label.lower() else 0.18)
            for label in labels
        }


class _NoTouchStore:
    def tables_of(self, *_args, **_kwargs):
        raise AssertionError("empty route should not read the store")


def test_rank_positions_deduplicates_at_first_occurrence():
    assert rank_positions(["a", "b", "a", "c"]) == {"a": 1, "b": 2, "c": 3}


def test_rrf_rewards_cross_channel_agreement():
    scores, provenance = reciprocal_rank_fusion(
        {"bm25": ["a", "b", "c"], "dense": ["b", "c", "a"]},
        rank_constant=60,
    )

    assert scores["b"] > scores["a"]
    assert provenance["b"] == {"bm25": 2, "dense": 1}


def test_rrf_channel_weight_can_disable_a_ranking():
    scores, provenance = reciprocal_rank_fusion(
        {"bm25": ["a", "b"], "dense": ["b", "a"]},
        weights={"dense": 0.0},
    )

    assert scores["a"] > scores["b"]
    assert "dense" not in provenance["a"]


def test_dense_row_channel_can_rescue_semantic_alias():
    grids = [
        [["label", "2024"], ["Net revenue from sales", "100"]],
        [["label", "2024"], ["Administrative expenses", "20"]],
    ]

    scores = _dense_table_scores(grids, ["doanh thu thuan"], _FakeEncoder())
    order = _ranked_mapping(scores, limit=10, minimum=0.35)

    assert order == [0]
    assert abs(scores[0] - 0.93) < 1e-9


def test_legacy_is_default_and_empty_route_is_unchanged():
    route = SimpleNamespace(report_ids=[], tickers=[])

    assert retrieve_for_route(route, _NoTouchStore()) == []
    assert retrieve_for_route(route, _NoTouchStore(), retrieval_mode="rrf") == []


def test_unknown_retrieval_mode_fails_fast():
    route = SimpleNamespace(report_ids=[], tickers=[])
    with unittest.TestCase().assertRaisesRegex(ValueError, "unknown retrieval_mode"):
        retrieve_for_route(route, _NoTouchStore(), retrieval_mode="dense-only")

def test_schema_channel_scores_direct_grid_without_tidy_dataframe():
    route = SimpleNamespace(years=[2024])
    grids = [
        [["CHỈ TIÊU", "Mã số", "2024"], ["Doanh thu thuần", "10", "100"]],
        [["CHỈ TIÊU", "Mã số", "2024"], ["Chi phí quản lý", "26", "20"]],
    ]

    scores = _schema_table_scores(route, grids, ["doanh thu thuan"], [0, 1])

    assert scores[0] > scores[1]


def test_schema_code_normalization_does_not_strip_all_zeroes():
    route = SimpleNamespace(years=[2024])
    grids = [[
        ["CHỈ TIÊU", "Mã số", "2024"],
        ["Doanh thu thuần", "10", "100"],
    ]]

    scores = _schema_table_scores(route, grids, ["doanh thu thuan"], [0])

    assert scores[0] > 100.0


def test_rrf_candidate_exposes_schema_score_and_channel_provenance():
    meta = {
        "report_id": "AAA_financial_statements_2024_consolidated",
        "ticker": "AAA", "table_pos": 3, "page": 5,
        "unit_scale": 1e6, "unit_source": "header", "n_rows": 10,
    }
    candidate = _candidate_record(
        meta, score=0.04, bm25_score=3.0, row_score=91.0,
        label_match=88.0, dense_score=0.9, retrieval_mode="rrf",
        channel_ranks={"bm25": 2, "schema_row": 1},
    )

    assert candidate["row_score"] == 91.0
    assert candidate["channel_ranks"] == {"bm25": 2, "schema_row": 1}

def test_route_source_validates_and_fingerprints_frozen_routes(tmp_path):
    path = tmp_path / "frozen.jsonl"
    path.write_text(
        '{"id": 1, "question": "Q1", "route": {"tickers": ["AAA"]}}\n',
        encoding="utf-8",
    )

    routes, info = _load_route_source(path, [{"id": 1, "question": "Q1"}])

    assert routes[1]["route"]["tickers"] == ["AAA"]
    assert info["enabled"] is True
    assert len(info["sha256"]) == 64


def test_route_source_rejects_question_drift(tmp_path):
    path = tmp_path / "frozen.jsonl"
    path.write_text(
        '{"id": 1, "question": "old", "route": {"tickers": ["AAA"]}}\n',
        encoding="utf-8",
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "question mismatch"):
        _load_route_source(path, [{"id": 1, "question": "new"}])

