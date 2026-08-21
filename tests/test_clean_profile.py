import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vifinqa.clean.profile import (
    PAYLOAD_SCHEMA_VERSION,
    advertised_parameter_billions,
    contract_fingerprints,
    validate_clean_runtime,
    validate_model_limit,
)


def test_clean_payload_schema_and_registry_fingerprints_are_stable_shape():
    assert PAYLOAD_SCHEMA_VERSION == 9
    fingerprints = contract_fingerprints()
    assert set(fingerprints) == {
        "metric_registry_sha256", "operator_registry_sha256"
    }
    assert all(len(value) == 64 for value in fingerprints.values())


def test_clean_runtime_rejects_raw_code_and_id_masks():
    with pytest.raises(ValueError, match="select_v2"):
        validate_clean_runtime(llm_mode="code")
    with pytest.raises(ValueError, match="llm_ids_file"):
        validate_clean_runtime(llm_mode="select_v2", llm_ids_file="ids.json")


def test_organizer_model_limit_accepts_14b_but_rejects_above_15b():
    assert advertised_parameter_billions("Qwen2.5-Coder-14B-Instruct") == 14.0
    validate_model_limit("Qwen2.5-Coder-14B-Instruct")
    with pytest.raises(ValueError, match="above"):
        validate_model_limit("Example-15.1B")

