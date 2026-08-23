import io
import unittest

import pandas as pd

from vifinqa.codegen.formula_solver import (
    build_compositional_ranking_plan,
    requires_formula_solver,
    try_formula_answer,
)


def _table(var, report_id, rows, table_pos=1,
           context="Báo cáo kết quả hoạt động kinh doanh Bảng cân đối kế toán"):
    return {"var": var, "report_id": report_id, "table_pos": table_pos,
            "report_year": int(report_id.split("_")[-2]),
            "context": context,
            "csv_text": pd.DataFrame(rows).to_csv(index=False)}


def _row(label, value, code="", col=1, col_name="2024", unit=1e9, row=1):
    return {"row": row, "label": label, "code": code, "col": col,
            "col_name": col_name, "value": value, "unit_scale": unit}


def _route(question, tickers, output_type="number", op="lookup",
           metric="doanh thu thuan", years=None):
    return {
        "question": question,
        "tickers": tickers,
        "years": years or [2024],
        "doc_type": "consolidated",
        "metric_norm": metric,
        "metric_variants": [metric],
        "output_type": output_type,
        "plan": {"op": op, "facts": []},
    }


def _eval(query, tables):
    env = {"float": float, "round": round, "min": min, "max": max, "abs": abs}
    env.update({t["var"]: pd.read_csv(io.StringIO(t["csv_text"])) for t in tables})
    return eval(query, env, {})


class FormulaSolverTests(unittest.TestCase):
    def test_count_direct_money_threshold(self):
        q = "Có bao nhiêu công ty có cam kết thuê hoạt động lớn hơn 40 tỷ đồng năm 2024?"
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated",
                   [_row("Cam kết thuê hoạt động", 50.0)]),
            _table("df2", "BBB_financial_statements_2024_consolidated",
                   [_row("Cam kết thuê hoạt động", 40.0)]),
            _table("df3", "CCC_financial_statements_2024_consolidated",
                   [_row("Cam kết thuê hoạt động", 100.0)]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB", "CCC"], "count", "count",
                   "cam ket thue hoat dong"),
            tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_count_corporate_income_tax_rejects_personal_income_tax(self):
        q = "Có bao nhiêu công ty có thuế thu nhập doanh nghiệp lớn hơn 50 tỷ đồng?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Thuế thu nhập cá nhân", 100.0),
        ])]
        ca = try_formula_answer(
            _route(q, ["AAA"], "count", "count", "thue thu nhap doanh nghiep"),
            tables)
        self.assertFalse(ca.ok)

    def test_count_rejects_implausible_double_unit_scaling(self):
        q = "Có bao nhiêu công ty có dòng tiền hoạt động lớn hơn 1 nghìn tỷ đồng?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Dòng tiền thuần từ hoạt động kinh doanh", 932_000_000_000.0,
                 unit=1e9),
        ])]
        ca = try_formula_answer(
            _route(q, ["AAA"], "count", "count",
                   "dong tien thuan tu hoat dong kinh doanh"), tables)
        self.assertFalse(ca.ok)

    def test_count_conditions_inside_top_revenue_population(self):
        q = ("Trong 2 doanh nghiệp có doanh thu thuần lớn nhất năm 2024, "
             "có bao nhiêu doanh nghiệp vừa có hệ số thanh toán nhanh lớn hơn 1 lần, "
             "vừa có hệ số nợ phải trả trên vốn chủ sở hữu nhỏ hơn 1,5 lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 300.0)], table_pos=1),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 100.0, row=1),
                _row("Hàng tồn kho", 20.0, row=2),
                _row("Nợ ngắn hạn", 50.0, row=3),
                _row("Nợ phải trả", 120.0, row=4),
                _row("Vốn chủ sở hữu", 100.0, row=5),
            ], table_pos=2),
            _table("df3", "BBB_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 200.0)], table_pos=1),
            _table("df4", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 70.0, row=1),
                _row("Hàng tồn kho", 30.0, row=2),
                _row("Nợ ngắn hạn", 50.0, row=3),
                _row("Nợ phải trả", 80.0, row=4),
                _row("Vốn chủ sở hữu", 100.0, row=5),
            ], table_pos=2),
            _table("df5", "CCC_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 100.0)], table_pos=1),
            _table("df6", "CCC_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 100.0, row=1),
                _row("Hàng tồn kho", 10.0, row=2),
                _row("Nợ ngắn hạn", 50.0, row=3),
                _row("Nợ phải trả", 20.0, row=4),
                _row("Vốn chủ sở hữu", 100.0, row=5),
            ], table_pos=2),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB", "CCC"], "count", "count"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)
        self.assertIn("df5", ca.pandas_query)  # revenue evidence for top-N filter

    def test_ranking_by_debt_equity_ratio(self):
        q = "Hệ số nợ phải trả trên vốn chủ sở hữu lớn nhất năm 2024 là bao nhiêu lần?"
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 120.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 80.0, row=1),
                _row("Vốn chủ sở hữu", 40.0, row=2),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB"], "ratio", "ranking"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_year_ranking_returns_argmax_year_and_reads_every_period(self):
        q = ("Doanh thu thuần của AAA cao nhất trong các năm 2022, 2023 và "
             "2024 là vào năm nào?")
        tables = [
            _table("df1", "AAA_financial_statements_2022_consolidated",
                   [_row("Doanh thu thuần", 100.0, code="10", col_name="2022")]),
            _table("df2", "AAA_financial_statements_2023_consolidated",
                   [_row("Doanh thu thuần", 180.0, code="10", col_name="2023")]),
            _table("df3", "AAA_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 120.0, code="10", col_name="2024")]),
        ]
        route = _route(
            q, ["AAA"], "year", "ranking", "doanh thu thuan",
            years=[2022, 2023, 2024])
        route["plan"].update(
            dimension="year", projection="year", direction="max")

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)
        self.assertTrue(all(var in ca.pandas_query for var in ("df1", "df2", "df3")))
        self.assertIn("formula_year_ranking", ca.detail)

    def test_year_ranking_treats_nam_nao_co_as_direct(self):
        q = "Năm nào có số dư tiền cao nhất trong các năm 2023 và 2024?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 100.0,
                         code="110", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 120.0,
                         code="110", col_name="2024")]),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "tien va cac khoan tuong duong tien", [2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)

    def test_year_ranking_does_not_read_viet_nam_co_as_nested(self):
        q = ("Ngân hàng Đầu tư và Phát triển Việt Nam có các khoản phải thu "
             "bên ngoài lớn nhất vào năm nào trong các năm 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Các khoản phải thu bên ngoài", 100.0,
                         col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Các khoản phải thu bên ngoài", 120.0,
                         col_name="2024")]),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "cac khoan phai thu ben ngoai", [2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)

    def test_year_ranking_does_not_read_cong_ty_co_phan_as_nested(self):
        q = ("Tổng tiền của Công ty cổ phần Bluemarq Group cao nhất vào năm "
             "nào trong các năm 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 100.0,
                         code="110", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 120.0,
                         code="110", col_name="2024")]),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "tien va cac khoan tuong duong tien", [2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)

    def test_year_ranking_uses_note_context_for_named_counterparty(self):
        q = ("Năm nào có số dư phải trả cho Công ty Liên doanh TNHH Crown "
             "Sài Gòn cao nhất trong các năm 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 900.0,
                         col_name="2023")], 1, "Đầu tư vào công ty liên kết"),
            _table("df2", "AAA_financial_statements_2023_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 100.0,
                         col_name="2023")], 2, "Phải trả người bán ngắn hạn"),
            _table("df3", "AAA_financial_statements_2024_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 800.0,
                         col_name="2024")], 1, "Đầu tư vào công ty liên kết"),
            _table("df4", "AAA_financial_statements_2024_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 120.0,
                         col_name="2024")], 2, "Phải trả người bán ngắn hạn"),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "phai tra cho cong ty lien doanh tnhh crown sai gon",
                       [2023, 2024])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)
        self.assertNotIn("df1", ca.pandas_query)
        self.assertNotIn("df3", ca.pandas_query)

    def test_year_ranking_supports_argmin(self):
        q = "Tổng tài sản của AAA thấp nhất trong giai đoạn 2022-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2022_consolidated",
                   [_row("Tổng tài sản", 100.0, code="270", col_name="2022")]),
            _table("df2", "AAA_financial_statements_2023_consolidated",
                   [_row("Tổng tài sản", 80.0, code="270", col_name="2023")]),
            _table("df3", "AAA_financial_statements_2024_consolidated",
                   [_row("Tổng tài sản", 120.0, code="270", col_name="2024")]),
        ]
        route = _route(
            q, ["AAA"], "year", "ranking", "tong tai san",
            years=[2022, 2023, 2024])
        route["plan"].update(
            dimension="year", projection="year", direction="min")

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)

    def test_year_ranking_supports_exact_formula_operands(self):
        q = "Biên lợi nhuận gộp của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated", [
                _row("Lợi nhuận gộp", 20.0, code="20", col_name="2023", row=1),
                _row("Doanh thu thuần", 100.0, code="10", col_name="2023", row=2),
            ]),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 30.0, code="20", col_name="2024", row=1),
                _row("Doanh thu thuần", 100.0, code="10", col_name="2024", row=2),
            ]),
        ]
        route = _route(
            q, ["AAA"], "year", "ranking", "bien loi nhuan gop",
            years=[2023, 2024])
        route["plan"].update(
            dimension="year", projection="year", direction="max")

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2024.0)

    def test_year_ranking_fails_closed_on_missing_period(self):
        q = "Doanh thu thuần của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [_table(
            "df1", "AAA_financial_statements_2024_consolidated",
            [_row("Doanh thu thuần", 120.0, code="10", col_name="2024")],
        )]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "doanh thu thuan",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_year_ranking_rejects_prior_value_when_current_cell_is_dash(self):
        q = "Năm nào hàng hóa có giá trị lớn nhất trong các năm 2024 và 2025?"
        tables = [
            _table(
                "df1", "AAA_financial_statements_2024_consolidated",
                [_row("Hàng hóa", 800.0, col_name="Đơn vị tính: VND")],
                context="9. Hàng tồn kho",
            ),
            _table(
                "df2", "AAA_financial_statements_2025_consolidated",
                [_row("Hàng hóa -", 1_220_617_073.0,
                      col_name="Đơn vị tính: VND")],
                context="9. Hàng tồn kho",
            ),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "hang hoa",
                   years=[2024, 2025]),
            tables,
        )

        self.assertFalse(ca.ok)

    def test_year_ranking_fails_closed_on_tie(self):
        q = "Doanh thu thuần của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Doanh thu thuần", 100.0, code="10", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 100.0, code="10", col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "doanh thu thuan",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)
        self.assertIn("tie", ca.detail)

    def test_year_ranking_rejects_parent_row_for_child_metric(self):
        q = ("Tài sản cố định vô hình của AAA cao nhất trong các năm 2023-2024 "
             "là năm nào?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Tài sản cố định", 100.0, code="220", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Tài sản cố định", 120.0, code="220", col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "tai san co dinh vo hinh",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_year_ranking_rejects_detail_row_for_aggregate_metric(self):
        q = "Tổng vốn chủ sở hữu của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Quỹ khác thuộc vốn chủ sở hữu", 100.0,
                         col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Quỹ khác thuộc vốn chủ sở hữu", 120.0,
                         col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "von chu so huu",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_year_ranking_does_not_drop_ratio_denominator(self):
        q = ("Năm nào AAA có tỷ trọng giá vốn cho thuê đất trên tổng giá vốn "
             "hàng bán cao nhất trong các năm 2023-2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Giá vốn hàng bán", 100.0, code="11", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Giá vốn hàng bán", 120.0, code="11", col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "gia von hang ban",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_direct_gross_margin(self):
        q = "Biên lợi nhuận gộp của AAA năm 2024 là bao nhiêu phần trăm?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận gộp", 30.0, row=1),
            _row("Doanh thu thuần", 120.0, row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "percent", "margin"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 25.0)

    def test_formula_rejects_segment_column_without_period_evidence(self):
        q = "Biên lợi nhuận gộp của AAA năm 2024 là bao nhiêu phần trăm?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận gộp", 30.0, code="20", col_name="Nội địa", row=1),
            _row("Doanh thu thuần", 100.0, code="10", col_name="Nội địa", row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "percent", "margin"), tables)
        self.assertFalse(ca.ok)

    def test_nested_ranking_selects_company_then_answers_other_formula(self):
        q = ("Năm 2024, hệ số thanh toán hiện hành của doanh nghiệp có "
             "hệ số nợ phải trả trên vốn chủ sở hữu cao nhất là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 200.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Nợ phải trả", 120.0, row=3),
                _row("Vốn chủ sở hữu", 100.0, row=4),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 90.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Nợ phải trả", 180.0, row=3),
                _row("Vốn chủ sở hữu", 100.0, row=4),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB"], "ratio", "ranking"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 0.9)
        self.assertEqual(_eval(ca.pandas_query, tables), 0.9)
        self.assertIn("formula_nested", ca.detail)

    def test_nested_ranking_does_not_fall_back_when_answer_formula_is_missing(self):
        q = ("Năm 2024, hệ số thanh toán hiện hành của doanh nghiệp có "
             "hệ số nợ phải trả trên vốn chủ sở hữu cao nhất là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 120.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 180.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "ratio", "ranking")

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)
        self.assertTrue(requires_formula_solver(route))

    def test_nested_ranking_by_cfo_over_operating_profit(self):
        q = ("Năm 2024, trong hai công ty AAA và BBB, doanh nghiệp có tỷ lệ "
             "lưu chuyển tiền thuần từ hoạt động kinh doanh trên lợi nhuận thuần "
             "từ hoạt động kinh doanh thấp nhất có biên lợi nhuận ròng bao nhiêu %?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 20.0, row=1),
                _row("Lợi nhuận thuần từ hoạt động kinh doanh", 10.0, code="30", row=2),
                _row("Lợi nhuận sau thuế", 30.0, code="60", row=3),
                _row("Doanh thu thuần", 100.0, code="10", row=4),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 50.0, row=1),
                _row("Lợi nhuận thuần từ hoạt động kinh doanh", 100.0, code="30", row=2),
                _row("Lợi nhuận sau thuế", 10.0, code="60", row=3),
                _row("Doanh thu thuần", 100.0, code="10", row=4),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB"], "percent", "ranking"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_nested_top_n_group_share_refuses(self):
        q = ("Trong 3 công ty AAA, BBB và CCC, 2 doanh nghiệp có biên lợi nhuận "
             "gộp cao nhất nắm giữ bao nhiêu phần trăm tổng tiền của cả nhóm?")
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 30.0, code="20", row=1),
                _row("Doanh thu thuần", 100.0, code="10", row=2),
            ]) for i, ticker in enumerate(("AAA", "BBB", "CCC"), 1)
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB", "CCC"], "percent",
                                       "ranking"), tables)
        self.assertFalse(ca.ok)

    def test_nested_ranking_selects_year_then_answers_interest_coverage(self):
        q = ("Trong giai đoạn 2023-2024, vào năm AAA có tỷ số D/E cao nhất, "
             "hệ số khả năng thanh toán lãi vay là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated", [
                _row("Nợ phải trả", 80.0, col_name="2023", row=1),
                _row("Vốn chủ sở hữu", 100.0, col_name="2023", row=2),
                _row("Lợi nhuận trước thuế", 30.0, col_name="2023", row=3),
                _row("Chi phí lãi vay", 10.0, col_name="2023", row=4),
            ]),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 180.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
                _row("Lợi nhuận trước thuế", 20.0, row=3),
                _row("Chi phí lãi vay", 10.0, row=4),
            ]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "ratio", "ranking", years=[2023, 2024]), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 3.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 3.0)

    def test_typed_nested_plan_separates_selector_and_projection_modes(self):
        q = ("Từ năm 2023 đến 2024, trong nhóm AAA và BBB, tại doanh nghiệp có "
             "tỷ lệ phần trăm tăng doanh thu thuần cao nhất, tỷ lệ phần trăm "
             "tăng của tổng chi phí bán hàng và chi phí quản lý doanh nghiệp "
             "là bao nhiêu %?")
        route = _route(q, ["AAA", "BBB"], "percent", "ranking",
                       years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.dimension, "entity")
        self.assertEqual(plan.selector.match.spec.name, "net_revenue")
        self.assertEqual(plan.selector.mode, "growth")
        self.assertEqual(plan.projection.match.spec.name, "sga_expense")
        self.assertEqual(plan.projection.mode, "growth")

    def test_typed_nested_plan_finds_selector_after_extreme_phrase(self):
        q = ("Trong nhóm AAA và BBB, doanh nghiệp có mức tăng lớn nhất từ năm "
             "2023 đến 2024 của tỷ lệ 365 lần hàng tồn kho bình quân đầu kỳ "
             "và cuối kỳ trên giá vốn hàng bán có tỷ lệ lợi nhuận gộp trên "
             "doanh thu thuần thay đổi bao nhiêu điểm phần trăm?")
        route = _route(q, ["AAA", "BBB"], "percentage_point", "ranking",
                       years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.selector.match.spec.name, "inventory_days")
        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual(plan.projection.match.spec.name, "gross_margin")
        self.assertEqual(plan.projection.mode, "delta")

    def test_typed_nested_inventory_days_selects_then_projects_margin_change(self):
        q = ("Trong nhóm AAA và BBB, doanh nghiệp có mức tăng lớn nhất từ năm "
             "2023 đến 2024 của tỷ lệ 365 lần hàng tồn kho bình quân đầu kỳ "
             "và cuối kỳ trên giá vốn hàng bán có tỷ lệ lợi nhuận gộp trên "
             "doanh thu thuần thay đổi bao nhiêu điểm phần trăm?")
        tables = []
        values = {
            "AAA": {2022: (100, 100, 20), 2023: (100, 100, 20),
                    2024: (200, 100, 30)},
            "BBB": {2022: (100, 100, 20), 2023: (100, 100, 20),
                    2024: (110, 100, 25)},
        }
        index = 1
        for ticker, by_year in values.items():
            for year, (inventory, cogs, gross_profit) in by_year.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated",
                    [
                        _row("Hàng tồn kho", inventory, code="140",
                             col_name=str(year), row=1),
                        _row("Giá vốn hàng bán", cogs, code="11",
                             col_name=str(year), row=2),
                        _row("Lợi nhuận gộp", gross_profit, code="20",
                             col_name=str(year), row=3),
                        _row("Doanh thu thuần", 100, code="10",
                             col_name=str(year), row=4),
                    ],
                ))
                index += 1
        route = _route(q, ["AAA", "BBB"], "percentage_point", "ranking",
                       years=[2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)
        self.assertIn("formula_nested_v3", ca.detail)

    def test_interest_coverage_rejects_cash_flow_interest_paid(self):
        q = "Hệ số khả năng thanh toán lãi vay của AAA năm 2024 là bao nhiêu lần?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận trước thuế", 20.0, code="50", row=1),
            _row("Tiền lãi vay đã trả", -10.0, row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "ratio", "ratio"), tables)
        self.assertFalse(ca.ok)

    def test_current_ratio_rejects_other_short_term_payable(self):
        q = "Hệ số thanh toán hiện hành của AAA năm 2024 là bao nhiêu lần?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Tài sản ngắn hạn", 200.0, code="100", row=1),
            _row("Phải trả ngắn hạn khác", 100.0, code="319", row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "ratio", "ratio"), tables)
        self.assertFalse(ca.ok)

    def test_net_margin_rejects_retained_earnings(self):
        q = "Biên lợi nhuận ròng của AAA năm 2024 là bao nhiêu phần trăm?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận sau thuế chưa phân phối", 30.0, code="421", row=1),
            _row("Doanh thu thuần", 100.0, code="10", row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "percent", "margin"), tables)
        self.assertFalse(ca.ok)

    def test_count_multiple_conditions_working_capital_and_cfo(self):
        q = ("Năm 2024, có bao nhiêu doanh nghiệp đồng thời ghi nhận vốn lưu động "
             "ròng âm và lưu chuyển tiền thuần từ hoạt động kinh doanh dương?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 80.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 10.0, row=3),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 120.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 10.0, row=3),
            ]),
            _table("df3", "CCC_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 80.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", -10.0, row=3),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB", "CCC"], "count", "count"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)

    def test_count_direct_condition_over_years(self):
        q = ("Trong các năm 2021, 2023 và 2024, AAA có bao nhiêu năm ghi nhận "
             "quỹ khen thưởng, phúc lợi lớn hơn 40 tỷ đồng?")
        tables = [
            _table("df1", "AAA_financial_statements_2021_consolidated",
                   [_row("Quỹ khen thưởng, phúc lợi", 30.0, col_name="2021")]),
            _table("df2", "AAA_financial_statements_2023_consolidated",
                   [_row("Quỹ khen thưởng, phúc lợi", 50.0, col_name="2023")]),
            _table("df3", "AAA_financial_statements_2024_consolidated",
                   [_row("Quỹ khen thưởng, phúc lợi", 60.0, col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "count", "count", "quy khen thuong phuc loi",
                   years=[2021, 2023, 2024]), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_count_temporal_multi_condition(self):
        q = ("Từ 2023 sang 2024, có bao nhiêu doanh nghiệp đồng thời tăng tỷ trọng "
             "hàng tồn kho trên tổng tài sản và giảm biên lợi nhuận gộp?")
        tables = []
        values = {
            "AAA": {2023: (10.0, 100.0, 30.0, 100.0),
                    2024: (20.0, 100.0, 20.0, 100.0)},
            "BBB": {2023: (10.0, 100.0, 20.0, 100.0),
                    2024: (8.0, 100.0, 25.0, 100.0)},
        }
        idx = 1
        for ticker, periods in values.items():
            for year, (inventory, assets, gross, revenue) in periods.items():
                tables.append(_table(
                    f"df{idx}", f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Hàng tồn kho", inventory, col_name=str(year), row=1),
                        _row("Tổng tài sản", assets, col_name=str(year), row=2),
                        _row("Lợi nhuận gộp", gross, col_name=str(year), row=3),
                        _row("Doanh thu thuần", revenue, col_name=str(year), row=4),
                    ]))
                idx += 1
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB"], "count", "count", years=[2023, 2024]),
            tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)

    def test_ratio_difference_between_companies(self):
        q = ("Biên lợi nhuận gộp năm 2024 của AAA chênh lệch bao nhiêu điểm phần "
             "trăm so với BBB?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 30.0, row=1),
                _row("Doanh thu thuần", 100.0, row=2),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 20.0, row=1),
                _row("Doanh thu thuần", 100.0, row=2),
            ]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB"], "percentage_point", "difference"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_average_margin_change_for_companies_with_revenue_increase(self):
        q = ("Mức thay đổi trung bình từ năm 2023 đến năm 2024 của biên lợi nhuận "
             "gộp tại các công ty có doanh thu thuần năm 2024 tăng so với năm "
             "2023 là bao nhiêu điểm phần trăm?")
        tables = []
        values = {
            "AAA": {2023: (20.0, 100.0), 2024: (36.0, 120.0)},
            "BBB": {2023: (30.0, 100.0), 2024: (20.0, 80.0)},
            "CCC": {2023: (10.0, 100.0), 2024: (30.0, 150.0)},
        }
        idx = 1
        for ticker, periods in values.items():
            for year, (gross, revenue) in periods.items():
                tables.append(_table(
                    f"df{idx}", f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Lợi nhuận gộp", gross, col_name=str(year), row=1),
                        _row("Doanh thu thuần", revenue, col_name=str(year), row=2),
                    ]))
                idx += 1
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB", "CCC"], "percentage_point", "average",
                   years=[2023, 2024]), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)  # AAA +10pp, CCC +10pp
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)


if __name__ == "__main__":
    unittest.main()
