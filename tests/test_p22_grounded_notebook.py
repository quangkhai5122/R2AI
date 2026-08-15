from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_p22_notebook_uses_schema8_semantic_v5_masks_and_oom_safe_flags():
    path = ROOT / "kaggle" / "vifinqa-codegen-p22.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "schema_version') == 8" in source
    assert "targets/p22b_semantic_groundable_v5.json', 2" in source
    assert "targets/p22c_semantic_groundable_v5.json', 4" in source
    assert "codegen_p22b_semantic_v5_sel14b.jsonl" in source
    assert "codegen_p22c_semantic_v5_sel14b.jsonl" in source
    assert source.count("--max-tokens 512 --max-input-tokens 6000") >= 2
    assert source.count("--batch-size 1") >= 3  # smoke + B + C
    assert "--checkpoint-every 1" in source
    assert "--llm-ids-file targets/p22b_rejected_non_year.json" not in source
    assert "APPROVE_STAGE_C = False" in source
