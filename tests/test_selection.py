"""Tests for structured selection (model picks cells, we write the pandas).

Motivation measured on submission #12 (Qwen 7B writing pandas itself):
35% of queries had no column filter, 15% skipped the unit conversion, 90%
omitted regex=False — 175 new answers moved the leaderboard by ~2 questions.
These tests pin that the synthesiser cannot reproduce any of those mistakes.
"""
import unittest

import pandas as pd

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.selection import (Selection, parse_selection, synthesize,
                                       confidence)


class _C:
    """Minimal stand-in for retrieval.shortlist.Candidate."""

    def __init__(self, var="df1", label="Doanh thu thuần", col=1,
                 col_name="2023", value=100.0, unit_scale=1e6, score=80.0,
                 row=1, fact_year=None):
        self.var, self.label, self.col, self.col_name = var, label, col, col_name
        self.value, self.unit_scale, self.score = value, unit_scale, score
        self.row, self.fact_year = row, fact_year
        self.report_id, self.table_pos, self.code = "AAA_2023", 1, "10"


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        s = parse_selection('{"op": "lookup", "operands": [2]}')
        self.assertEqual((s.op, s.operands), ("lookup", [2]))

    def test_fenced_json_with_prose(self):
        s = parse_selection('Sure!\n```json\n{"op":"difference","operands":[1,3]}\n```\nDone')
        self.assertEqual((s.op, s.operands), ("difference", [1, 3]))

    def test_scalar_operand_is_wrapped(self):
        self.assertEqual(parse_selection('{"op":"lookup","operands":1}').operands, [1])

    def test_alternative_keys(self):
        self.assertEqual(parse_selection('{"op":"lookup","idx":[4]}').operands, [4])

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_selection("I cannot answer this question."))
        self.assertIsNone(parse_selection(""))


class ValidationTests(unittest.TestCase):
    def test_rejects_out_of_range_operand(self):
        self.assertIn("out of range", Selection("lookup", [9]).valid_for(3))

    def test_rejects_unknown_op(self):
        self.assertIn("unknown op", Selection("frobnicate", [1]).valid_for(3))

    def test_rejects_too_few_operands(self):
        self.assertIn("needs exactly 2", Selection("difference", [1]).valid_for(3))

    def test_rejects_extra_fixed_arity_operands(self):
        error = Selection("lookup", [1, 2]).valid_for(2)
        self.assertIn("exactly 1", error)

    def test_rejects_duplicate_operands(self):
        self.assertIn("duplicate", Selection("sum", [1, 1]).valid_for(2))


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.route = {"unit_scale": 1e9, "output_type": "number"}

    def test_lookup_applies_unit_conversion(self):
        # cell is 100 (triệu VND); question asks tỷ đồng -> 100*1e6/1e9 = 0.1
        c = _C(value=100.0, unit_scale=1e6)
        ans, q, err = synthesize(Selection("lookup", [1]), [c], self.route)
        self.assertIsNone(err)
        self.assertAlmostEqual(ans, 0.1, places=6)
        self.assertIn("/ 1e+09", q)

    def test_generated_query_uses_exact_stable_row_and_column(self):
        c = _C(label="Doanh thu (thuần)", row=7, col=3)
        _a, q, _e = synthesize(Selection("lookup", [1]), [c], self.route)
        self.assertIn("['row'] == 7", q)
        self.assertIn("['col'] == 3", q)
        self.assertNotIn("str.contains", q)
        compile(q, "<q>", "eval")          # single expression, grader-compatible

    def test_exact_row_avoids_substring_collision(self):
        c = _C(label="Các khoản tương đương tiền", row=5, col=3,
               value=2.0, unit_scale=1.0)
        answer, query, error = synthesize(
            Selection("lookup", [1]), [c],
            {"unit_scale": 1.0, "output_type": "number"},
        )
        df = pd.DataFrame([
            {"row": 3, "label": "Tiền và các khoản tương đương tiền",
             "col": 3, "value": 6.0},
            {"row": 5, "label": "Các khoản tương đương tiền",
             "col": 3, "value": 2.0},
        ])
        replay = run_code(query, {"df1": df})
        self.assertIsNone(error)
        self.assertEqual(answer, 2.0)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(replay["value"], 2.0)

    def test_growth_is_percent_and_ordered_end_first(self):
        end = _C(var="df1", value=120.0, unit_scale=1.0)
        base = _C(var="df2", value=100.0, unit_scale=1.0)
        ans, _q, err = synthesize(Selection("growth_pct", [1, 2]), [end, base],
                                  {"unit_scale": 1.0, "output_type": "percent"})
        self.assertIsNone(err)
        self.assertAlmostEqual(ans, 20.0)

    def test_ratio_returns_percent_not_fraction(self):
        num = _C(var="df1", value=9.0, unit_scale=1.0)
        den = _C(var="df2", value=100.0, unit_scale=1.0)
        ans, _q, _e = synthesize(Selection("ratio", [1, 2]), [num, den],
                                 {"unit_scale": 1.0, "output_type": "percent"})
        self.assertAlmostEqual(ans, 9.0)

    def test_ratio_times_is_not_scaled(self):
        a = _C(var="df1", value=9.0, unit_scale=1.0)
        b = _C(var="df2", value=3.0, unit_scale=1.0)
        ans, _q, _e = synthesize(Selection("ratio_times", [1, 2]), [a, b],
                                 {"unit_scale": 1.0, "output_type": "ratio"})
        self.assertAlmostEqual(ans, 3.0)

    def test_ratio_query_parenthesises_scaled_operands(self):
        a = _C(var="df1", row=1, value=9.0, unit_scale=1e6)
        b = _C(var="df2", row=2, value=3.0, unit_scale=1e6)
        answer, query, error = synthesize(
            Selection("ratio_times", [1, 2]), [a, b],
            {"unit_scale": 1.0, "output_type": "ratio"},
        )
        dfs = {
            "df1": pd.DataFrame([{"row": 1, "col": 1, "value": 9.0}]),
            "df2": pd.DataFrame([{"row": 2, "col": 1, "value": 3.0}]),
        }
        replay = run_code(query, dfs)
        self.assertIsNone(error)
        self.assertEqual(answer, 3.0)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(replay["value"], 3.0)

    def test_ranking_max_and_min(self):
        cs = [_C(var="df1", value=1.0, unit_scale=1e9),
              _C(var="df2", value=7.0, unit_scale=1e9),
              _C(var="df3", value=4.0, unit_scale=1e9)]
        hi, _q, _e = synthesize(Selection("ranking_max", [1, 2, 3]), cs, self.route)
        lo, _q2, _e2 = synthesize(Selection("ranking_min", [1, 2, 3]), cs, self.route)
        self.assertAlmostEqual(hi, 7.0)
        self.assertAlmostEqual(lo, 1.0)

    def test_year_output_projects_fact_year_not_report_value(self):
        cs = [_C(var="df1", row=1, value=10.0, unit_scale=1.0,
                 fact_year=2019),
              _C(var="df2", row=2, value=30.0, unit_scale=1.0,
                 fact_year=2021),
              _C(var="df3", row=3, value=20.0, unit_scale=1.0,
                 fact_year=2020)]
        answer, query, error = synthesize(
            Selection("ranking_max", [1, 2, 3]), cs,
            {"unit_scale": 1.0, "output_type": "year",
             "years": [2019, 2021, 2020]},
        )
        dfs = {
            c.var: pd.DataFrame([{"row": c.row, "col": c.col,
                                  "value": c.value}]) for c in cs
        }
        replay = run_code(query, dfs)
        self.assertIsNone(error)
        self.assertEqual(answer, 2021.0)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(replay["value"], 2021.0)

    def test_year_projection_rejects_ambiguous_duplicate_years(self):
        cs = [_C(var="df1", fact_year=2020),
              _C(var="df2", fact_year=2020)]
        answer, _query, error = synthesize(
            Selection("argmax", [1, 2]), cs,
            {"unit_scale": 1.0, "output_type": "year", "years": [2020]},
        )
        self.assertIsNone(answer)
        self.assertIn("distinct years", error)

    def test_year_projection_rejects_distinct_indices_for_same_stable_cell(self):
        cs = [_C(var="df1", row=4, col=2, fact_year=2020),
              _C(var="df1", row=4, col=2, fact_year=2024)]
        answer, query, error = synthesize(
            Selection("ranking_max", [1, 2]), cs,
            {"unit_scale": 1.0, "output_type": "year",
             "years": [2020, 2024]},
        )
        self.assertIsNone(answer)
        self.assertEqual(query, "")
        self.assertIn("duplicate stable cells", error)

    def test_count_is_grounded_in_selected_cells(self):
        cs = [_C(var="df1", row=1, value=5.0, unit_scale=1.0),
              _C(var="df2", row=2, value=8.0, unit_scale=1.0)]
        answer, query, error = synthesize(
            Selection("count", [1, 2]), cs,
            {"unit_scale": 1.0, "output_type": "count"},
        )
        dfs = {
            c.var: pd.DataFrame([{"row": c.row, "col": c.col,
                                  "value": c.value}]) for c in cs
        }
        replay = run_code(query, dfs)
        self.assertIsNone(error)
        self.assertEqual(answer, 2.0)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(replay["value"], 2.0)

    def test_count_rejects_distinct_indices_for_same_stable_cell(self):
        cs = [_C(var="df1", row=4, col=2),
              _C(var="df1", row=4, col=2)]
        answer, query, error = synthesize(
            Selection("count", [1, 2]), cs,
            {"unit_scale": 1.0, "output_type": "count"},
        )
        self.assertIsNone(answer)
        self.assertEqual(query, "")
        self.assertIn("duplicate stable cells", error)

    def test_count_requires_count_output_type(self):
        answer, _query, error = synthesize(
            Selection("count", [1]), [_C()],
            {"unit_scale": 1.0, "output_type": "number"},
        )
        self.assertIsNone(answer)
        self.assertIn("requires output_type=count", error)

    def test_ratio_times_requires_ratio_output_type(self):
        answer, _query, error = synthesize(
            Selection("ratio_times", [1, 2]), [_C(), _C(var="df2")],
            {"unit_scale": 1.0, "output_type": "percent"},
        )
        self.assertIsNone(answer)
        self.assertIn("requires output_type=ratio", error)

    def test_average_and_sum(self):
        cs = [_C(var="df1", value=2.0, unit_scale=1e9),
              _C(var="df2", value=4.0, unit_scale=1e9)]
        s, _q, _e = synthesize(Selection("sum", [1, 2]), cs, self.route)
        a, _q2, _e2 = synthesize(Selection("average", [1, 2]), cs, self.route)
        self.assertAlmostEqual(s, 6.0)
        self.assertAlmostEqual(a, 3.0)

    def test_division_by_zero_is_reported_not_raised(self):
        num = _C(var="df1", value=1.0, unit_scale=1.0)
        den = _C(var="df2", value=0.0, unit_scale=1.0)
        ans, _q, err = synthesize(Selection("ratio", [1, 2]), [num, den], self.route)
        self.assertIsNone(ans)
        self.assertIn("zero", err)

    def test_percent_lookup_scales_a_ratio_cell(self):
        c = _C(label="Tỷ lệ sở hữu", value=0.9, unit_scale=1.0)
        ans, q, _e = synthesize(Selection("lookup", [1]), [c],
                                {"unit_scale": 1.0, "output_type": "percent"})
        self.assertAlmostEqual(ans, 90.0)      # organizer-confirmed: 90 not 0.9
        self.assertIn("* 100", q)

    def test_hard_typed_magnitude_guard_rejects_unit_explosion(self):
        c = _C(label="Tỷ lệ", value=2_000_000.0, unit_scale=1.0)
        answer, query, error = synthesize(
            Selection("lookup", [1]), [c],
            {"unit_scale": 1.0, "output_type": "percent"},
        )
        self.assertIsNone(answer)
        self.assertEqual(query, "")
        self.assertIn("hard limit", error)


class ConfidenceTests(unittest.TestCase):
    def test_confidence_follows_weakest_pick(self):
        cs = [_C(score=90.0), _C(score=64.0)]
        c = confidence(Selection("difference", [1, 2]), cs, 10.0,
                       {"output_type": "number"})
        self.assertAlmostEqual(c, 64.0)

    def test_implausible_unit_lowers_confidence(self):
        cs = [_C(score=90.0)]
        good = confidence(Selection("lookup", [1]), cs, 90.0, {"output_type": "percent"})
        bad = confidence(Selection("lookup", [1]), cs, 0.9, {"output_type": "percent"})
        self.assertLess(bad, good)


if __name__ == "__main__":
    unittest.main()
