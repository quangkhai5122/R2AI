from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "45_make_p22_target_masks.py"
    spec = importlib.util.spec_from_file_location("p22_target_masks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_control_builds_expected_disjoint_masks():
    module = _module()
    masks = module.build_masks(
        ROOT / "artifacts" / "codegen_p21r_all_v3.jsonl",
        ROOT / "artifacts" / "retrieval.jsonl",
        ROOT / "artifacts" / "shortlist_rescue_audit.json",
    )
    b = masks["p22b_rejected_non_year.json"]
    c = masks["p22c_rescue_fact_complete.json"]
    bc = masks["p22bc_combined.json"]
    assert (b["count"], c["count"], bc["count"]) == (55, 48, 103)
    assert set(b["ids"]).isdisjoint(c["ids"])
    assert set(bc["ids"]) == set(b["ids"]) | set(c["ids"])
    assert b["inputs"]["retrieval"]["sha256"] == module.FROZEN_RETRIEVAL_SHA256


def test_mask_builder_never_references_locked_gold():
    source = (ROOT / "scripts" / "45_make_p22_target_masks.py").read_text(
        encoding="utf-8",
    ).lower()
    assert "p24_locked_gold" not in source
    assert "p24_locked_questions" not in source

