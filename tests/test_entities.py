from pathlib import Path

import pandas as pd
import pytest

from vifinqa.router.entities import StockMap, parse_question


@pytest.fixture()
def stock_map(tmp_path: Path) -> StockMap:
    path = tmp_path / "code_stock.csv"
    pd.DataFrame(
        [
            ("FPT", "Công ty Cổ phần FPT"),
            ("FTS", "Công ty Cổ phần Chứng khoán FPT"),
            ("VNM", "Công ty Cổ phần Sữa Việt Nam"),
            ("GAS", "Tổng Công ty Khí Việt Nam - CTCP"),
            ("POW", "Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP"),
            ("DTK", "Tổng Công ty Điện lực TKV - CTCP"),
            ("OCB", "Ngân hàng TMCP Phương Đông"),
            ("EIB", "Ngân hàng TMCP Xuất Nhập khẩu Việt Nam"),
            ("MBB", "Ngân hàng TMCP Quân đội"),
            ("SAB", "Tổng Công ty cổ phần Bia - Rượu - Nước giải khát Sài Gòn"),
            ("DBC", "CTCP Tập đoàn Dabaco Việt Nam"),
            ("HAG", "Cong ty Co phan Hoang Anh Gia Lai"),
            ("HNG", "Cong ty Co phan Nong nghiep Quoc te Hoang Anh Gia Lai"),
            ("DCM", "Cong ty Co phan Phan bon Dau khi Ca Mau"),
            ("DXS", "Cong ty Co phan Dich vu Bat dong san Dat Xanh"),
            ("SAM", "Cong ty Co phan SAM Holdings"),
        ],
        columns=["Mã CK", "Tên công ty"],
    ).to_csv(path, index=False)
    return StockMap(path)


def test_finds_multiple_company_names_in_mention_order(stock_map):
    parsed = parse_question(
        "So sánh Công ty Cổ phần Sữa Việt Nam và Tổng Công ty Khí Việt Nam "
        "trong năm 2024.",
        stock_map,
    )
    assert parsed.tickers == ["VNM", "GAS"]
    assert parsed.ticker_source == "explicit_name"


def test_finds_lowercase_tickers(stock_map):
    parsed = parse_question("so sánh vnm với gas năm 2023", stock_map)
    assert parsed.tickers == ["VNM", "GAS"]
    assert parsed.ticker_source == "explicit"


def test_long_company_alias_masks_embedded_other_ticker(stock_map):
    parsed = parse_question(
        "Lợi nhuận của Công ty Cổ phần Chứng khoán FPT năm 2023 là bao nhiêu?",
        stock_map,
    )
    assert parsed.tickers == ["FTS"]


def test_parenthetical_target_wins_over_named_counterparty(stock_map):
    parsed = parse_question(
        "Cong ty Hoang Anh Gia Lai bao lanh khoan vay cho Cong ty Co phan "
        "Nong nghiep Quoc te Hoang Anh Gia Lai (HNG) nam 2023 bao nhieu?",
        stock_map,
    )
    assert parsed.tickers == ["HNG"]


def test_lowercase_vietnamese_word_is_not_misread_as_ticker(stock_map):
    parsed = parse_question(
        "Tien chi de mua sam tai san co dinh cua Cong ty Co phan Dich vu "
        "Bat dong san Dat Xanh nam 2021 la bao nhieu?",
        stock_map,
    )
    assert parsed.tickers == ["DXS"]


def test_parent_company_scope_excludes_earlier_counterparty_alias(stock_map):
    parsed = parse_question(
        "So du phai thu tu Cong ty CP Bao bi Dau khi Viet Nam cua cong ty me "
        "DCM nam 2023 la bao nhieu?",
        stock_map,
    )
    assert parsed.tickers == ["DCM"]


def test_curated_bank_brands_are_exact_multi_entity_aliases(stock_map):
    parsed = parse_question(
        "Tổng tài sản của MBBank lớn hơn Eximbank bao nhiêu triệu đồng?",
        stock_map,
    )
    assert parsed.tickers == ["MBB", "EIB"]
    assert parsed.ticker_source == "explicit_name"


def test_repeated_parent_company_scope_keeps_both_comparison_entities(stock_map):
    parsed = parse_question(
        "Lợi nhuận của công ty mẹ Tổng Công ty cổ phần Bia - Rượu - Nước "
        "giải khát Sài Gòn trừ lợi nhuận của công ty mẹ CTCP Tập đoàn Dabaco "
        "Việt Nam là bao nhiêu?",
        stock_map,
    )
    assert parsed.tickers == ["SAB", "DBC"]


@pytest.mark.parametrize(
    ("question", "years"),
    [
        ("VNM trong giai đoạn 2021-2024", [2021, 2022, 2023, 2024]),
        ("VNM từ năm 2022 đến năm 2025", [2022, 2023, 2024, 2025]),
        ("VNM các năm 2018 và 2020", [2018, 2020]),
    ],
)
def test_year_ranges_are_expanded_but_year_lists_are_not(question, years, stock_map):
    assert parse_question(question, stock_map).years == years


def test_hundred_billion_question_unit(stock_map):
    parsed = parse_question(
        "Lợi nhuận của VNM năm 2024 là bao nhiêu trăm tỷ đồng?", stock_map
    )
    assert parsed.unit_scale == 1e11
    assert parsed.unit_name == "trăm tỷ đồng"
    assert parsed.output_type == "number"


def test_parenthetical_unit_is_not_misread_as_count(stock_map):
    parsed = parse_question(
        "Giá vốn của VNM năm 2024 là bao nhiêu? (đơn vị: đồng)", stock_map
    )
    assert parsed.output_type == "number"
    assert parsed.unit_name == "đồng"


def test_requested_money_unit_beats_percent_condition(stock_map):
    parsed = parse_question(
        "Trong các năm có tỷ lệ lợi nhuận trên doanh thu lớn hơn 10%, "
        "doanh thu thấp nhất của VNM là bao nhiêu tỷ đồng?",
        stock_map,
    )
    assert parsed.output_type == "number"
    assert parsed.unit_scale == 1e9
    assert parsed.unit_name == "tỷ đồng"


def test_first_requested_unit_beats_later_condition_unit(stock_map):
    parsed = parse_question(
        "Chi phí có thể tăng tối đa bao nhiêu phần trăm trước khi hệ số của "
        "VNM giảm về 2,0 lần?",
        stock_map,
    )
    assert parsed.output_type == "percent"


def test_count_of_units_is_distinct_from_parenthetical_unit(stock_map):
    parsed = parse_question(
        "Có bao nhiêu đơn vị gồm VNM và GAS có hệ số dưới 1,5 lần?", stock_map
    )
    assert parsed.output_type == "count"


@pytest.mark.parametrize(
    ("question", "output_type", "unit_name", "is_percent"),
    [
        ("ROE của VNM năm 2024 là bao nhiêu phần trăm?", "percent", "%", True),
        (
            "Biên lợi nhuận của VNM tăng bao nhiêu điểm phần trăm năm 2024?",
            "percentage_point",
            "điểm phần trăm",
            False,
        ),
        (
            "Tỷ lệ CFO trên LNST của VNM năm 2024 là bao nhiêu lần?",
            "ratio",
            "lần",
            False,
        ),
        ("Doanh thu VNM cao nhất vào năm nào?", "year", "năm", False),
        ("Có bao nhiêu ngân hàng OCB và EIB tăng trưởng dương?", "count", "số lượng", False),
    ],
)
def test_output_type_has_precedence_over_metric_words(
    question, output_type, unit_name, is_percent, stock_map
):
    parsed = parse_question(question, stock_map)
    assert parsed.output_type == output_type
    assert parsed.unit_name == unit_name
    assert parsed.is_percent is is_percent


def test_answer_tail_turn_unit_overrides_percent_filter_metric(stock_map):
    parsed = parse_question(
        "Trong nhom co ty trong tai san dai han cao nhat, vong quay tong tai san "
        "cua VNM nam 2024 la bao nhieu vong?",
        stock_map,
    )
    assert parsed.output_type == "ratio"
    assert parsed.unit_name == "vòng"


@pytest.mark.parametrize(
    ("question", "unit_scale", "unit_name"),
    [
        ("VNM co bao nhieu trieu co phieu?", 1e6, "triệu cổ phiếu"),
        ("Khoan vay cua VNM la bao nhieu trieu USD?", 1e6, "triệu USD"),
    ],
)
def test_non_money_quantity_units_are_not_labeled_as_vnd(
    question, unit_scale, unit_name, stock_map
):
    parsed = parse_question(question, stock_map)
    assert parsed.output_type == "number"
    assert parsed.unit_scale == unit_scale
    assert parsed.unit_name == unit_name


def test_count_output_supports_in_population_wording(stock_map):
    parsed = parse_question(
        "Co bao nhieu trong so cac ngan hang OCB va EIB co tang truong duong?",
        stock_map,
    )
    assert parsed.output_type == "count"
