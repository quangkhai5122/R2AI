import json

from vifinqa.codegen.selection_v2_prompt import SYSTEM


def test_v2_examples_are_minified_and_use_exact_fact_names():
    examples = [
        line for line in SYSTEM.splitlines()
        if line.startswith('{"schema_version":2')
    ]
    assert len(examples) == 3
    for raw in examples:
        program = json.loads(raw)
        assert set(program["facts"]) == {
            f"F{i}" for i in range(1, len(program["facts"]) + 1)
        }
        assert '"slot":' not in raw
        assert '"role":' not in raw
        assert "```" not in raw
    assert "Minify JSON. No prose, markdown, fences or code." in SYSTEM
