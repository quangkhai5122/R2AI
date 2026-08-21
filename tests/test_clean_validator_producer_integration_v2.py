import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from vifinqa.codegen import generate


class _Store:
    def __init__(self, *_args, **_kwargs):
        pass


class _Bundle:
    def __init__(self, rec, _store, _k, run_signature="", **_kwargs):
        self.id = rec["id"]
        self.question = rec["question"]
        self.route = rec["route"]
        self.run_signature = run_signature
        self.tables = [{"var": "df1", "report_id": f"AAA_{self.id}",
                        "table_pos": 1}]
        self.dfs = {"df1": pd.DataFrame([{
            "row": 1, "col": 1, "value": float(self.id),
            "unit_scale": 1.0, "label": "Shares", "col_name": "2024",
        }])}
        self._candidate = SimpleNamespace(
            var="df1", row=1, col=1, value=float(self.id), unit_scale=1.0,
            label="Shares", col_name="2024", score=90.0, rescue=False,
            fact_year=2024, report_year=2024, fact_slot="F1",
            fact_role="value", fact_metric="Shares", ticker="AAA",
            report_id=f"AAA_{self.id}", table_pos=1, code="",
        )
        self.shortlist_trace = {"candidate_count": 1, "rescue_mode": "none"}

    def shortlist_v2(self, _encoder=None, top_n=24):
        del top_n
        return [self._candidate]

    def atomic_slots(self):
        return [{"ticker": "AAA", "year": 2024, "metric": "Shares",
                 "role": "value", "family": "routed_fact"}]

    def select_v2_messages(self, _encoder=None):
        return [{"role": "user", "content": self.question}]

    def used_vars(self, _code):
        return self.tables


class _Client:
    name = "fake"

    def chat_batch(self, conversations, **_kwargs):
        program = {
            "schema_version": 2,
            "output_type": "number",
            "facts": {"shares": {"ref": 1, "as": "number", "role": "value"}},
            "bindings": {},
            "root": {"var": "shares"},
        }
        return [[json.dumps(program)] for _ in conversations]


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "validate_clean_codegen_v2.py"
    spec = importlib.util.spec_from_file_location("clean_validator_integration_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_selection_v2_producer_output_passes_clean_validator(tmp_path):
    retrieval = tmp_path / "retrieval.jsonl"
    output = tmp_path / "codegen.jsonl"
    rows = [
        {
            "id": i,
            "question": f"q{i}",
            "route": {
                "output_type": "number", "unit_scale": 1.0,
                "unit_name": "count", "plan": {"facts": []},
            },
        }
        for i in (1, 2, 3)
    ]
    retrieval.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    with patch.object(generate, "Store", _Store), \
            patch.object(generate, "QuestionBundle", _Bundle):
        generate.run_codegen(
            retrieval, tmp_path, output, _Client(),
            use_rule_fallback=False, debug_rounds=0, checkpoint_every=3,
            llm_mode="select_v2", llm_target="all", n_samples=2,
            temperature=0.2, run_signature="integration-signature",
        )
    validator = _load_validator()
    report = validator.validate_codegen(
        retrieval, output, expected_count=3, require_complete_llm=True,
    )
    assert report["validated_records"] == 3
    assert report["llm_completed"] == 3
    assert report["selection_outcomes"] == {"accepted": 3}
