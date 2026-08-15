import json

import pandas as pd

from vifinqa.devset.p24_authoring_ext import P24ForensicTableLoader


def test_forensic_loader_retains_numeric_cell_mistaken_for_code(tmp_path):
    store = tmp_path / "store"
    (store / "tables").mkdir(parents=True)
    pd.DataFrame([{
        "report_id": "AAA_financial_statements_2024_separate",
        "ticker": "AAA", "year": 2024, "doc_type": "separate",
        "table_pos": 2, "unit_scale": 1_000_000.0,
        "grid_json": json.dumps([
            ["", "31/12/2024", "31/12/2023"],
            ["Internal payable", "307.293", "174.706"],
        ]),
    }]).to_parquet(store / "tables" / "AAA.parquet", index=False)
    pd.DataFrame([{
        "report_id": "AAA_financial_statements_2024_separate",
        "ticker": "AAA", "year": 2024, "doc_type": "separate", "n_tables": 1,
    }]).to_parquet(store / "reports.parquet", index=False)

    loaded = P24ForensicTableLoader(store)(
        "AAA_financial_statements_2024_separate", 2,
    )
    hit = loaded[(loaded.row == 1) & (loaded.col == 1)]
    assert len(hit) == 1
    assert hit.iloc[0].value == 307293.0
    assert hit.iloc[0].unit_scale == 1_000_000.0
