from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(ROOT / "kaggle" / "kaggle_codegen.py", "p22_kaggle_runner")


def _args(mask_hash="", mask_count=0):
    return SimpleNamespace(
        backend="hf", model="model", n=1, temperature=0.2,
        max_tokens=384, debug_rounds=0, k=0, rule_first=False,
        llm_target="empty", llm_mode="select_v2", use_dense=False,
        dense_model="", seed=13, load_4bit=True, max_input_tokens=7000,
        quantization="awq", tp=2, max_model_len=8192,
        rescue_no_candidates=True, rescue_table_k=20, rescue_min_score=28.0,
        batch_size=2, checkpoint_every=16, limit=0,
        llm_ids_sha256=mask_hash, llm_ids_count=mask_count,
    )


def test_target_mask_must_be_manifest_fingerprinted(tmp_path: Path):
    runner = _runner()
    target = tmp_path / "targets" / "p22b.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"ids": [8, 3, 5]}), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    ids, actual, path = runner.load_llm_ids(
        "targets/p22b.json", tmp_path,
        {"targets/p22b.json": digest}, require_manifest=True,
    )
    assert ids == {3, 5, 8}
    assert actual == digest
    assert path == target.resolve()

    with pytest.raises(SystemExit, match="not fingerprinted"):
        runner.load_llm_ids("targets/p22b.json", tmp_path, {}, True)


def test_target_mask_rejects_duplicates(tmp_path: Path):
    runner = _runner()
    target = tmp_path / "mask.json"
    target.write_text("[1, 1]", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate"):
        runner.load_llm_ids(str(target), tmp_path, {}, require_manifest=False)


def test_run_signature_fingerprints_target_mask_bytes_and_count():
    runner = _runner()
    scorer = dict(runner.FUZZY_SCORER)
    base = runner.run_signature(_args("a" * 64, 55), "manifest", scorer)
    assert base != runner.run_signature(_args("b" * 64, 55), "manifest", scorer)
    assert base != runner.run_signature(_args("a" * 64, 56), "manifest", scorer)


def test_payload_manifest_hashes_optional_targets(tmp_path: Path):
    builder = _load(ROOT / "scripts" / "04_make_kaggle_payload.py",
                    "p22_payload_builder")
    for rel, data in {
        "retrieval.jsonl": b"{}\n",
        "code/kaggle_codegen.py": b"runner",
        "store/reports.parquet": b"store",
        "targets/p22b.json": b'{"ids":[1]}',
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = builder._build_manifest(tmp_path)
    assert manifest["schema_version"] == 8
    assert "targets/p22b.json" in manifest["files"]

