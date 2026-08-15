from __future__ import annotations

from vifinqa.codegen.atomic_slots import plan_atomic_slots
from vifinqa.retrieval.shortlist import Candidate, _metric_grounding


def _route(*, metric: str = "metric", tickers=("AAA",), years=(2024,),
           output_type: str = "number", op: str = "lookup") -> dict:
    facts = [
        {"ticker": ticker, "year": year, "doc_type": "consolidated",
         "metric": metric, "role": "value"}
        for ticker in tickers for year in years
    ]
    return {
        "tickers": list(tickers), "years": list(years),
        "doc_type": "consolidated", "metric_norm": metric,
        "output_type": output_type, "plan": {"op": op, "facts": facts},
    }


def _candidate(*, label: str, col_name: str, report_year: int = 2024,
               code: str = "", value: float = 1.0) -> Candidate:
    return Candidate(
        var="df1", report_id=f"AAA_financial_statements_{report_year}_consolidated",
        table_pos=1, row=1, label=label, code=code, col=1,
        col_name=col_name, value=value, unit_scale=1.0, score=80.0,
        lexical=80.0, semantic=0.0, ticker="AAA", report_year=report_year,
    )


def _fact(metric: str, *, year: int = 2024, period: str = "same_period",
          anchors=()) -> dict:
    return {
        "ticker": "AAA", "year": year, "metric": metric,
        "period_role": period, "semantic_anchors": list(anchors),
        "route_grounded": True,
    }


def test_planner_marks_nested_question_missing_sga_leaves_ungrounded():
    question = (
        "Trong nhom AAA va BBB, cong ty co ty le SG&A cao nhat co "
        "bien loi nhuan rong bao nhieu phan tram?"
    )
    slots, trace = plan_atomic_slots(
        question, _route(tickers=("AAA", "BBB"), metric="bien loi nhuan rong",
                         output_type="percent", op="ranking"),
    )
    assert slots
    assert not trace["planner_guard"]["ok"]
    assert set(trace["planner_guard"]["missing"]) == {
        "sga:chi phi ban hang", "sga:chi phi quan ly doanh nghiep",
    }
    assert not any(slot["route_grounded"] for slot in slots)


def test_planner_strips_average_ratio_prefix_into_atomic_metrics():
    metric = (
        "trung binh ty le chi phi du phong rui ro tin dung tren "
        "loi nhuan truoc du phong"
    )
    slots, trace = plan_atomic_slots(
        "Gia tri trung binh ty le chi phi du phong rui ro tin dung tren "
        "loi nhuan truoc du phong la bao nhieu?",
        _route(metric=metric, output_type="percent", op="average"),
    )
    assert {slot["metric"] for slot in slots} == {
        "chi phi du phong rui ro tin dung", "loi nhuan truoc du phong",
    }
    assert trace["planner_guard"]["ok"]


def test_31_december_phrase_is_ending_not_beginning():
    slots, _ = plan_atomic_slots(
        "So du den ngay 31 thang 12 nam 2024 la bao nhieu?",
        _route(metric="so du"),
    )
    assert slots[0]["period_role"] == "ending"


def test_temporal_grounding_rejects_report_plus_one_current_column():
    candidate = _candidate(label="Doanh thu thuan", col_name="Nam nay",
                           report_year=2022)
    ok, _, reason = _metric_grounding(
        candidate, _fact("doanh thu thuan", year=2021), "",
    )
    assert not ok
    assert reason == "year_mismatch:relative=2022"


def test_context_date_disambiguates_comparison_table_year():
    candidate = _candidate(
        label="Cho vay khach hang - gop",
        col_name="Chua qua han va chua phai lap du phong",
        report_year=2022,
    )
    ok, _, reason = _metric_grounding(
        candidate,
        _fact("vay khach hang gop chua qua han va chua phai lap du phong",
              year=2021, period="ending"),
        "Tai ngay 31 thang 12 nam 2021",
    )
    assert ok, reason


def test_entity_anchor_requires_the_named_company_not_partial_tokens():
    wrong = _candidate(
        label="Cong ty Co phan 3F Viet - san pham thit",
        col_name="Ty le loi ich kinh te",
    )
    fact = _fact("loi ich kinh te", anchors=("thuc pham 3f viet",))
    ok, _, reason = _metric_grounding(wrong, fact, "")
    assert not ok
    assert reason.startswith("entity_anchor_missing:")

    exact = _candidate(
        label="Cong ty TNHH Thuc pham 3F Viet",
        col_name="Ty le loi ich kinh te",
    )
    assert _metric_grounding(exact, fact, "")[0]


def test_metric_domain_guards_reject_profit_segment_and_restatement_aliases():
    fact = _fact("chi phi du phong rui ro tin dung")
    cases = [
        (_candidate(
            label="Loi nhuan truoc chi phi du phong rui ro tin dung",
            col_name="Nam nay"), "profit_not_provision"),
        (_candidate(label="Chi phi du phong rui ro tin dung",
                    col_name="Mien Bac"), "segment_not_total"),
        (_candidate(label="Chi phi du phong rui ro tin dung",
                    col_name="So da trinh bay"), "presentation_state_ambiguous"),
    ]
    for candidate, reason in cases:
        ok, _, got = _metric_grounding(candidate, fact, "")
        assert not ok
        assert reason in got


def test_ending_balance_column_layout_is_allowed_but_beginning_is_not():
    fact = _fact("con lai cua quyen su dung dat", year=2018, period="ending")
    ending = _candidate(label="So du cuoi nam",
                        col_name="Quyen su dung dat trieu dong",
                        report_year=2018)
    assert _metric_grounding(ending, fact, "")[0]

    beginning = _candidate(label="So du dau nam",
                           col_name="Quyen su dung dat trieu dong",
                           report_year=2018)
    ok, _, reason = _metric_grounding(beginning, fact, "")
    assert not ok
    assert reason == "period_contradiction:expected_ending"
