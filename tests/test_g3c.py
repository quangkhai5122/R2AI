from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from vifinqa.g3c.cache import ScoreCache, VectorCache
from vifinqa.g3c.common import (
    GPU_RESULT_SCHEMA,
    config_fingerprint,
    load_config,
    validate_config,
    write_json,
)
from vifinqa.g3c.freeze import (
    build_dev_selection,
    freeze_selected_candidate,
    load_candidate_freeze,
)
from vifinqa.g3c.leaves import decompose_atomic_leaves
from vifinqa.g3c.payload import (
    _assert_no_forbidden_paths,
    _file_record,
    _manifest_files,
    _validate_payload_file_set,
)
from vifinqa.g3c.protocol import (
    build_protocol_freeze,
    validate_protocol_freeze,
)
from vifinqa.g3c.retrieval import (
    reciprocal_rank_fusion,
    select_with_leaf_quota,
)
from vifinqa.g3c.serialize import (
    row_passages,
    sanitize_numeric_text,
    table_passage,
)


class ExactStore:
    def find_reports(self, ticker, year, doc_type, allow_fallback=True):
        assert allow_fallback is False
        return [f"{ticker}_financial_statements_{year}_{doc_type}"]


def route(tickers, years, doc_type="consolidated", metric_norm=""):
    return {
        "tickers": tickers,
        "years": years,
        "doc_type": doc_type,
        "metric_norm": metric_norm,
        "metric_keys": [],
    }


def test_config_is_pinned_and_pre_cutoff():
    config = load_config("configs/g3c_qwen_retrieval_v1.json")
    assert config["models"]["embedding"]["revision"] == (
        "5cf2132abc99cad020ac570b19d031efec650f2b"
    )
    assert config["models"]["reranker"]["revision"] == (
        "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    )
    assert config["runtime"]["sequential_model_loading"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("models", "embedding", "revision"), "main"),
        (("runtime", "quantization"), "nf4"),
        (("runtime", "attention_implementation"), "flash_attention_2"),
        (("policy", "promotion_runs"), 2),
    ],
)
def test_config_rejects_protocol_drift(field, value):
    config = load_config("configs/g3c_qwen_retrieval_v1.json")
    changed = deepcopy(config)
    cursor = changed
    for key in field[:-1]:
        cursor = cursor[key]
    cursor[field[-1]] = value
    with pytest.raises(ValueError):
        validate_config(changed)


def test_cagr_uses_endpoints_and_roles_without_gold():
    question = (
        "CAGR cua loi nhuan sau thue VRE theo bao cao rieng cua cong ty me "
        "tu nam 2019 den nam 2021 (2 ky) la bao nhieu phan tram?"
    )
    leaves = decompose_atomic_leaves(
        question, route(["VRE"], [2019, 2020, 2021], "separate"), ExactStore()
    )
    assert [(leaf.period_year, leaf.role) for leaf in leaves] == [
        (2019, "base"), (2021, "end")
    ]
    assert [leaf.report_year for leaf in leaves] == [2019, 2021]
    assert [leaf.report_ids[0] for leaf in leaves] == [
        "VRE_financial_statements_2019_separate",
        "VRE_financial_statements_2021_separate",
    ]
    assert {leaf.metric_key for leaf in leaves} == {"net_profit"}


def test_count_creates_one_leaf_per_ticker():
    question = (
        "Trong HUT, PC1, PNJ, co bao nhieu doanh nghiep co no phai tra "
        "duong theo bao cao hop nhat nam 2015, so voi nguong 0?"
    )
    leaves = decompose_atomic_leaves(
        question, route(["HUT", "PC1", "PNJ"], [2015]), ExactStore()
    )
    assert [leaf.ticker for leaf in leaves] == ["HUT", "PC1", "PNJ"]
    assert {leaf.metric_key for leaf in leaves} == {"liabilities"}


def test_ratio_creates_numerator_and_denominator():
    question = (
        "No phai tra bang bao nhieu lan tong tai san cua DTK "
        "theo bao cao hop nhat nam 2020?"
    )
    leaves = decompose_atomic_leaves(
        question, route(["DTK"], [2020]), ExactStore()
    )
    assert [(leaf.metric_key, leaf.role) for leaf in leaves] == [
        ("liabilities", "numerator"),
        ("total_assets", "denominator"),
    ]
    assert {leaf.operation for leaf in leaves} == {"ratio"}


def test_scope_delta_has_two_exact_scopes():
    question = (
        "Doanh thu thuan hop nhat cao hon hoac thap hon so lieu rieng cua "
        "cong ty me VGC nam 2019 bao nhieu trieu dong? "
        "Tinh hop nhat tru rieng."
    )
    leaves = decompose_atomic_leaves(
        question, route(["VGC"], [2019]), ExactStore()
    )
    assert [(leaf.doc_type, leaf.role) for leaf in leaves] == [
        ("consolidated", "minuend"),
        ("separate", "subtrahend"),
    ]
    assert {leaf.metric_key for leaf in leaves} == {"net_revenue"}


def test_prior_period_binds_current_report_and_opening_period():
    question = (
        "Trong bao cao hop nhat nam 2018 cua QNS, so dau ky tuong ung "
        "cuoi nam 2017 cua loi nhuan truoc thue la bao nhieu trieu dong?"
    )
    leaves = decompose_atomic_leaves(
        question, route(["QNS"], [2017, 2018]), ExactStore()
    )
    assert len(leaves) == 1
    assert leaves[0].period_year == 2017
    assert leaves[0].report_year == 2018
    assert dict(leaves[0].qualifiers)["period"] == "opening"


def test_nested_margin_average_has_metric_by_period_cross_product():
    question = (
        "Bien loi nhuan sau thue binh quan cua SNZ theo bao cao rieng "
        "cua cong ty me trong hai nam 2021 va 2022 la bao nhieu phan tram?"
    )
    leaves = decompose_atomic_leaves(
        question, route(["SNZ"], [2021, 2022], "separate"), ExactStore()
    )
    assert len(leaves) == 4
    assert {leaf.metric_key for leaf in leaves} == {
        "net_profit", "net_revenue"
    }
    assert {leaf.period_year for leaf in leaves} == {2021, 2022}
    assert {leaf.report_year for leaf in leaves} == {2021, 2022}
    assert {leaf.report_ids[0] for leaf in leaves} == {
        "SNZ_financial_statements_2021_separate",
        "SNZ_financial_statements_2022_separate",
    }


def test_table_passage_excludes_numeric_values_but_keeps_years():
    meta = {
        "report_id": "BVH_financial_statements_2015_consolidated",
        "ticker": "BVH",
        "year": 2015,
        "doc_type": "consolidated",
        "table_pos": 0,
        "page": 2,
        "n_rows": 2,
        "unit_scale": 1_000_000.0,
        "unit_source": "explicit",
        "context": "Loi nhuan dat 1.174.931 trieu dong nam 2015.",
        "grid_json": json.dumps([
            ["Chi tieu", "Nam 2015"],
            ["Loi nhuan sau thue", "1.174.931"],
        ]),
    }
    content = table_passage(meta)["content"]
    assert "1.174.931" not in content
    assert "1174931" not in content
    assert "<num>" in content
    assert "2015" in content


def test_row_passage_never_serializes_cell_value():
    meta = {
        "report_id": "AAA_financial_statements_2020_separate",
        "ticker": "AAA",
        "year": 2020,
        "doc_type": "separate",
        "table_pos": 3,
        "page": 9,
        "unit_scale": 1_000_000.0,
        "unit_source": "header",
    }
    cells = [{
        **meta,
        "row": 5,
        "label": "Loi nhuan sau thue",
        "row_code": "60",
        "col_name": "Nam 2020",
        "value": 987654321.25,
    }]
    content = row_passages(cells, meta)[0]["content"]
    assert "987654321" not in content
    assert "2020" in content
    assert "loi nhuan sau thue" in content


def test_numeric_sanitizer_preserves_year_only():
    value = sanitize_numeric_text(
        "Nam 2024 dat 12.345,67 va tang 8,5%."
    )
    assert "2024" in value
    assert "12.345" not in value
    assert "8,5" not in value


def test_rrf_and_leaf_quota_are_deterministic():
    a = ("r1", 1)
    b = ("r2", 2)
    c = ("r3", 3)
    scores = reciprocal_rank_fusion([[a, b], [b, c]], 60)
    assert scores[b] > scores[a]
    selected = select_with_leaf_quota(
        {"leaf-b": [b, c], "leaf-a": [a, b]},
        [b, a, c], quota=1, depth=3,
    )
    assert selected == [b, a, c]


def test_vector_and_score_caches_roundtrip(tmp_path):
    vectors = VectorCache(tmp_path / "vectors.npz")
    vectors.put("abc", np.array([1.0, 2.0], dtype=np.float32))
    vectors.save()
    reloaded = VectorCache(tmp_path / "vectors.npz")
    assert np.allclose(reloaded.get("abc"), [1.0, 2.0])

    scores = ScoreCache(tmp_path / "scores.json")
    scores.put("pair", 0.75)
    scores.save()
    assert ScoreCache(tmp_path / "scores.json").get("pair") == 0.75


def test_payload_path_guard_rejects_gold_and_evaluation():
    with pytest.raises(ValueError):
        _assert_no_forbidden_paths([
            {"path": "data/g3b_gold.jsonl"},
            {"path": "reports/dev_evaluation.json"},
        ])


def test_payload_file_set_allows_kaggle_to_strip_upload_metadata(tmp_path):
    core = tmp_path / "code" / "runner.py"
    core.parent.mkdir(parents=True)
    core.write_text("pass\n", encoding="utf-8")
    metadata = tmp_path / "dataset-metadata.json"
    metadata.write_text('{"id":"owner/dataset"}\n', encoding="utf-8")
    manifest_path = tmp_path / "g3c_gpu_payload_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "paths": {"dataset_metadata": "dataset-metadata.json"},
        "files": [_file_record(tmp_path, "code/runner.py")],
        "upload_sidecars": [
            _file_record(tmp_path, "dataset-metadata.json")
        ],
    }

    _validate_payload_file_set(tmp_path, manifest)
    metadata.unlink()
    _validate_payload_file_set(tmp_path, manifest)


def test_payload_file_set_still_rejects_core_drift_and_extra_files(tmp_path):
    core = tmp_path / "core.txt"
    core.write_text("core\n", encoding="utf-8")
    metadata = tmp_path / "dataset-metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "g3c_gpu_payload_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "paths": {"dataset_metadata": "dataset-metadata.json"},
        "files": [_file_record(tmp_path, "core.txt")],
        "upload_sidecars": [
            _file_record(tmp_path, "dataset-metadata.json")
        ],
    }

    core.unlink()
    with pytest.raises(ValueError, match="missing=.*core.txt"):
        _validate_payload_file_set(tmp_path, manifest)
    core.write_text("core\n", encoding="utf-8")
    (tmp_path / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra=.*unexpected.txt"):
        _validate_payload_file_set(tmp_path, manifest)


def test_manifest_files_excludes_upload_only_metadata(tmp_path):
    (tmp_path / "dataset-metadata.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "core.txt").write_text("core\n", encoding="utf-8")
    paths = {
        row["path"] for row in _manifest_files(
            tmp_path, excluded_paths={"dataset-metadata.json"},
        )
    }
    assert paths == {"core.txt"}


def test_protocol_freeze_roundtrip(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "protocol.json"
    built = build_protocol_freeze(
        repo_root=repo_root,
        config_path=(
            repo_root / "configs/g3c_qwen_retrieval_v1.json"
        ),
        output_path=output,
    )
    validated = validate_protocol_freeze(
        repo_root=repo_root,
        config_path=(
            repo_root / "configs/g3c_qwen_retrieval_v1.json"
        ),
        freeze_path=output,
        verify_worktree=True,
    )
    assert validated["protocol_fingerprint"] == built["protocol_fingerprint"]


def test_dev_gate_selects_and_freezes_only_passing_candidate(tmp_path):
    config_path = "configs/g3c_qwen_retrieval_v1.json"
    config = load_config(config_path)
    gpu = {
        "schema_version": GPU_RESULT_SCHEMA,
        "mode": "dev",
        "backend": "qwen",
        "scientific_evidence_valid": True,
        "config_sha256": config_fingerprint(config),
        "stages_written": ["R0", "R1"],
        "stage_artifacts": {
            "R0": {
                "hard_constraint_violation_count": 0,
                "sha256": "0" * 64,
            },
            "R1": {
                "hard_constraint_violation_count": 0,
                "sha256": "1" * 64,
            },
        },
        "runtime": {"timings": {"total_seconds": 10.0}},
        "model_revisions": {
            name: {
                "model_id": value["model_id"],
                "revision": value["revision"],
                "tokenizer_revision": value["tokenizer_revision"],
            }
            for name, value in config["models"].items()
        },
        "instructions_sha256": "2" * 64,
        "protocol_fingerprint": "5" * 64,
        "payload_fingerprint": "3" * 64,
        "run_signature": "4" * 64,
    }
    gpu_path = tmp_path / "gpu.json"
    write_json(gpu_path, gpu)

    def evaluation(leaf, full, docs, tables):
        return {
            "policy_mode": "dev",
            "evidence_mode": "end_to_end",
            "integrity": {"passed": True},
            "metrics": {
                "leaf_recall_at_k": leaf,
                "full_plan_coverage_rate": full,
                "docs_f2_macro": docs,
                "tables_f2_macro": tables,
                "answer_accuracy": 0.1,
                "execution_accuracy": 0.1,
            },
        }

    r0_path = tmp_path / "r0.json"
    r1_path = tmp_path / "r1.json"
    write_json(r0_path, evaluation(0.80, 0.68, 0.90, 0.61))
    write_json(r1_path, evaluation(0.83, 0.72, 0.895, 0.605))
    selection_path = tmp_path / "selection.json"
    selection = build_dev_selection(
        config_path=config_path,
        gpu_result_manifest_path=gpu_path,
        evaluations={"R0": r0_path, "R1": r1_path},
        output_path=selection_path,
    )
    assert selection["selected_stage"] == "R1"
    freeze_path = tmp_path / "freeze.json"
    freeze_selected_candidate(
        config_path=config_path,
        gpu_result_manifest_path=gpu_path,
        selection_path=selection_path,
        output_path=freeze_path,
    )
    assert load_candidate_freeze(freeze_path)["selected_stage"] == "R1"

