"""Tests for the P1 additions: extractive metric, shortlist, decomposition,
formula registry and answer-unit normalisation."""
import unittest

import pandas as pd

from vifinqa.codegen.formulas import REGISTRY, describe_for_prompt, get
from vifinqa.codegen.units import (check_answer_unit, cell_is_already_percent,
                                   percent_from_cell)
from vifinqa.retrieval.shortlist import (
    _numeric_header_year,
    _period_kind,
    build_shortlist,
    render_shortlist,
)
from vifinqa.router.decompose import build_plan, detect_op, evidence_budget
from vifinqa.router.metric_phrase import extract_metric
from vifinqa.utils.viet_text import label_metric_score


class MetricPhraseTests(unittest.TestCase):
    def test_strips_quantity_prefix_and_entity_tail(self):
        mp = extract_metric(
            "Số dư trả trước cho người bán của CTCP Phát triển Bất động sản Văn "
            "Phú (VPI) cuối năm 2024 tính bằng đồng là bao nhiêu?",
            ["cong ty co phan phat trien bat dong san van phu"], ["VPI"])
        self.assertEqual(mp.core, "tra truoc cho nguoi ban")

    def test_beats_the_old_subtractive_phrase_on_the_real_regression(self):
        # measured: the old phrase scored 63 against this gold row (< 78 cut)
        gold = "Trả trước cho người bán"
        old = "so du tra truoc cho nguoi ban tinh dong"
        mp = extract_metric("Số dư trả trước cho người bán của X cuối năm 2024 "
                            "tính bằng đồng là bao nhiêu?", [], [])
        self.assertGreater(max(label_metric_score(gold, v) for v in mp.variants()),
                           label_metric_score(gold, old))

    def test_keeps_qualifier_in_wide_form(self):
        mp = extract_metric("Tổng số tiền trả trước cho người bán ngắn hạn của Y "
                            "cuối năm 2023 là bao nhiêu?", [], [])
        self.assertIn("ngan han", mp.wide)

    def test_does_not_eat_doanh_thu(self):
        mp = extract_metric("Doanh thu thuần của ABC năm 2023 là bao nhiêu tỷ đồng?",
                            [], ["ABC"])
        self.assertTrue(mp.core.startswith("doanh thu"), mp.core)


class DecomposeTests(unittest.TestCase):
    def test_detects_ops(self):
        self.assertEqual(detect_op("Lợi nhuận sau thuế năm 2023 tăng trưởng bao "
                                   "nhiêu phần trăm so với 2022?"), "growth_pct")
        self.assertEqual(detect_op("Chênh lệch doanh thu giữa A và B"), "difference")
        self.assertEqual(detect_op("Công ty nào có tổng tài sản lớn nhất"), "ranking")

    def test_tong_cong_ty_is_not_an_aggregation(self):
        self.assertEqual(
            detect_op("Trả trước cho người bán của Tổng Công ty cổ phần X "
                      "cuối năm 2023 là bao nhiêu?"), "lookup")

    def test_growth_expands_to_prior_year(self):
        plan = build_plan("Doanh thu năm 2023 tăng trưởng bao nhiêu phần trăm?",
                          ["AAA"], [2023], "consolidated", "doanh thu")
        self.assertEqual(plan.op, "growth_pct")
        self.assertEqual({f.year for f in plan.facts}, {2023, 2022})

    def test_budget_grows_with_facts(self):
        one = build_plan("Doanh thu 2023", ["A"], [2023], "consolidated", "m")
        many = build_plan("Chênh lệch giữa A và B năm 2023", ["A", "B"], [2023],
                          "consolidated", "m")
        self.assertLess(evidence_budget(one), evidence_budget(many))
        self.assertLessEqual(evidence_budget(many), 12)

    def test_year_ranking_plan_declares_dimension_projection_and_direction(self):
        plan = build_plan(
            "Doanh thu thuan thap nhat trong cac nam 2022-2024 la nam nao?",
            ["AAA"], [2022, 2023, 2024], "consolidated", "doanh thu thuan",
            output_type="year",
        )
        self.assertEqual(plan.op, "ranking")
        self.assertEqual(plan.dimension, "year")
        self.assertEqual(plan.projection, "year")
        self.assertEqual(plan.direction, "min")
        self.assertEqual({fact.year for fact in plan.facts}, {2022, 2023, 2024})


class FormulaTests(unittest.TestCase):
    def test_growth_is_in_percent_units(self):
        self.assertAlmostEqual(REGISTRY["growth_pct"].fn(120.0, 100.0), 20.0)

    def test_ratio_is_in_percent_units(self):
        self.assertAlmostEqual(REGISTRY["ratio"].fn(9.0, 10.0), 90.0)

    def test_ratio_times_is_not_scaled(self):
        self.assertAlmostEqual(REGISTRY["ratio_times"].fn(9.0, 3.0), 3.0)

    def test_describe_mentions_multiple_operands(self):
        self.assertIn("BOTH", describe_for_prompt("difference", 2))
        self.assertEqual(get("nope").name, "lookup")


class UnitTests(unittest.TestCase):
    def test_ratio_cell_becomes_percent(self):
        self.assertAlmostEqual(percent_from_cell(0.9, "Tỷ lệ sở hữu", ""), 90.0)

    def test_percent_cell_is_left_alone(self):
        self.assertAlmostEqual(percent_from_cell(90.0, "Tỷ lệ sở hữu", ""), 90.0)

    def test_percent_marked_column_is_left_alone(self):
        self.assertTrue(cell_is_already_percent("Biến động", "%", 0.5))

    def test_warns_on_ratio_shaped_percent_answer(self):
        self.assertIn("RATIO", check_answer_unit(0.9, "percent"))
        self.assertIsNone(check_answer_unit(90.0, "percent"))

    def test_year_and_count_ranges(self):
        self.assertIsNone(check_answer_unit(2023, "year"))
        self.assertIsNotNone(check_answer_unit(15.5, "year"))


class ShortlistTests(unittest.TestCase):
    def test_non_finite_header_value_is_not_a_year(self):
        self.assertIsNone(_numeric_header_year(float("inf")))

    def test_ocr_value_headers_keep_leftmost_current_period_semantics(self):
        self.assertEqual(
            _period_kind("Giá gốc/Số có khả năng trả nợ"), "current")
        self.assertEqual(_period_kind("Đơn vị tính: VND"), "current")

    def test_balance_sheet_dates_distinguish_opening_and_closing_periods(self):
        self.assertEqual(_period_kind("1/1/2024 VND"), "prior")
        self.assertEqual(_period_kind("01/01/2024 VND"), "prior")
        self.assertEqual(_period_kind("31/12/2024 VND"), "current")

    def test_start_of_report_year_serves_as_prior_year_closing_balance(self):
        rows = [
            {"row": 1, "label": "Hàng tồn kho", "code": "140", "col": 1,
             "col_name": "31/12/2024 VND", "value": 200.0, "unit_scale": 1.0},
            {"row": 1, "label": "Hàng tồn kho", "code": "140", "col": 2,
             "col_name": "1/1/2024 VND", "value": 100.0, "unit_scale": 1.0},
        ]
        tables = [{
            "var": "df1", "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024, "table_pos": 1,
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]

        candidates = build_shortlist(tables, ["hang ton kho"], [2023])

        self.assertEqual(candidates[0].col_name, "1/1/2024 VND")
        self.assertEqual(candidates[0].value, 100.0)

    def test_years_split_into_header_row_are_attached_to_value_columns(self):
        rows = [
            {"row": 1, "label": "Ma so Thuyet minh", "code": "", "col": 3,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": 3092022.0, "unit_scale": 1.0},
            {"row": 1, "label": "Ma so Thuyet minh", "code": "", "col": 4,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": 3092021.0, "unit_scale": 1.0},
            {"row": 5, "label": "Gia von hang ban va dich vu cung cap",
             "code": "11", "col": 3,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": -200.0, "unit_scale": 1.0},
            {"row": 5, "label": "Gia von hang ban va dich vu cung cap",
             "code": "11", "col": 4,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": -180.0, "unit_scale": 1.0},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2022_consolidated",
            "report_year": 2022,
            "table_pos": 10,
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]

        current = build_shortlist(tables, ["gia von hang ban"], [2022])
        prior = build_shortlist(tables, ["gia von hang ban"], [2021])

        self.assertEqual((current[0].col, current[0].value), (3, -200.0))
        self.assertIn("2022", current[0].col_name)
        self.assertEqual((prior[0].col, prior[0].value), (4, -180.0))
        self.assertIn("2021", prior[0].col_name)

    def setUp(self):
        rows = [
            {"row": 1, "label": "Trả trước cho người bán", "code": "132",
             "col": 1, "col_name": "31/12/2024", "value": 100.0, "unit_scale": 1e6},
            {"row": 2, "label": "Trả trước cho người bán dài hạn", "code": "212",
             "col": 1, "col_name": "31/12/2024", "value": 5.0, "unit_scale": 1e6},
            {"row": 3, "label": "Chi phí quản lý doanh nghiệp", "code": "26",
             "col": 1, "col_name": "31/12/2024", "value": 7.0, "unit_scale": 1e6},
        ]
        self.tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                        "csv_text": pd.DataFrame(rows).to_csv(index=False)}]

    def test_ranks_the_matching_row_first(self):
        cands = build_shortlist(self.tables, ["tra truoc cho nguoi ban"], [2024])
        self.assertTrue(cands)
        self.assertEqual(cands[0].label, "Trả trước cho người bán")

    def test_qualifier_mismatch_is_penalised(self):
        cands = build_shortlist(self.tables, ["tra truoc cho nguoi ban ngan han"], [2024])
        labels = [c.label for c in cands]
        self.assertIn("Trả trước cho người bán", labels)
        self.assertNotIn("Trả trước cho người bán dài hạn", labels)

    def test_render_is_compact_and_safe_when_empty(self):
        self.assertIn("no candidate row", render_shortlist([]))
        self.assertIn("df1", render_shortlist(
            build_shortlist(self.tables, ["tra truoc cho nguoi ban"], [2024])))

    def test_canonical_child_beats_lexical_parent(self):
        rows = [
            {"row": 1, "label": "Tien gui va vay cac TCTD khac", "code": "",
             "col": 1, "col_name": "So cuoi nam", "value": 100.0,
             "unit_scale": 1e6},
            {"row": 2, "label": "Vay cac TCTD khac", "code": "",
             "col": 1, "col_name": "So cuoi nam", "value": 40.0,
             "unit_scale": 1e6},
        ]
        tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                   "report_year": 2024,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["vay cac TCTD khac"], [2024],
            question="So du vay cac TCTD khac cuoi nam 2024")
        self.assertEqual(cands[0].label, "Vay cac TCTD khac")

    def test_intangible_fixed_asset_child_beats_parent(self):
        rows = [
            {"row": 1, "label": "X Tai san co dinh", "code": "220", "col": 1,
             "col_name": "So cuoi nam", "value": 100.0, "unit_scale": 1e6},
            {"row": 2, "label": "Tai san co dinh vo hinh", "code": "227", "col": 1,
             "col_name": "So cuoi nam", "value": 40.0, "unit_scale": 1e6},
        ]
        tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                   "report_year": 2024,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["gia tri con lai cua tai san co dinh vo hinh"], [2024],
            question="Gia tri con lai cua tai san co dinh vo hinh tong cong")
        self.assertEqual(cands[0].label, "Tai san co dinh vo hinh")

    def test_named_counterparty_child_beats_generic_parent(self):
        rows = [
            {"row": 1, "label": "Vay dai han", "code": "", "col": 1,
             "col_name": "31/12/2017", "value": 100.0, "unit_scale": 1e3},
            {"row": 2,
             "label": "Cong ty Co phan HoangAnh Gia Lai Cong ty me Vay dai han",
             "code": "", "col": 1, "col_name": "31/12/2017", "value": 40.0,
             "unit_scale": 1e3},
        ]
        tables = [{"var": "df1", "report_id": "HNG_2017_separate", "table_pos": 3,
                   "report_year": 2017,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["vay dai han voi hoang anh gia lai"], [2017],
            question="Vay dai han voi Hoang Anh Gia Lai cuoi nam 2017")
        self.assertEqual(cands[0].row, 2)

    def test_opening_qualifier_selects_opening_column(self):
        rows = [
            {"row": 1, "label": "No ngan han", "code": "310", "col": 1,
             "col_name": "So cuoi nam", "value": 200.0, "unit_scale": 1e6},
            {"row": 1, "label": "No ngan han", "code": "310", "col": 2,
             "col_name": "So dau nam", "value": 120.0, "unit_scale": 1e6},
        ]
        tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                   "report_year": 2024,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["no ngan han"], [2024],
            question="No ngan han dau nam 2024 la bao nhieu?")
        self.assertEqual(cands[0].col_name, "So dau nam")


if __name__ == "__main__":
    unittest.main()
