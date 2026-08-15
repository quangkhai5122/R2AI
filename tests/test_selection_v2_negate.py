from types import SimpleNamespace

import pandas as pd

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.selection_v2 import compile_program


def test_negate_preserves_quantity_type_and_grounding():
    candidate = SimpleNamespace(
        var="df1", row=1, col=1, label="Chi phí", col_name="2024",
        value=25.0, unit_scale=1.0, score=90.0, rescue=False,
        fact_year=2024, fact_slot="F1", report_id="AAA_2024",
        table_pos=1, code="",
    )
    program = {
        "schema_version": 2, "output_type": "number",
        "facts": {"cost": {"ref": 1, "as": "money", "role": "value"}},
        "bindings": {},
        "root": {"op": "negate", "args": [{"var": "cost"}]},
    }
    compiled = compile_program(
        program, [candidate], {"output_type": "number", "unit_scale": 1}, "q",
    )
    frame = pd.DataFrame([{"row": 1, "col": 1, "value": 25.0}])
    assert run_code(compiled.query, {"df1": frame})["value"] == -25.0

