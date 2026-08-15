from __future__ import annotations

import hashlib

from vifinqa.codegen.llm_client import GenerationSample, _annotate_hf
from vifinqa.codegen.selection_v2 import evaluate_samples
from vifinqa.codegen.selection_v2_replay import _saved_samples


def test_max_token_stop_is_traced_as_generation_truncated_not_parse_error():
    sample = GenerationSample(
        '{"schema_version":2,"facts":{',
        finish_reason="length", token_count=512, max_tokens=512,
    )
    decision, trace = evaluate_samples(
        [sample], [object()], {"output_type": "number"}, "q", lambda _q: {},
    )
    assert decision is None
    attempt = trace["attempts"][0]
    assert attempt["stage"] == "generation"
    assert attempt["reason_code"] == "generation_truncated"
    assert attempt["generation_finish_reason"] == "length"
    assert attempt["generation_tokens"] == 512
    assert attempt["generation_max_tokens"] == 512
    assert trace["rejection_counts"] == {"generation_truncated": 1}


def test_hf_annotation_distinguishes_length_from_eos_stop():
    samples = _annotate_hf(
        ["partial", "done"], [[10, 11, 12], [10, 2, 0]],
        max_tokens=3, eos_token_id=2, pad_token_id=0,
    )
    assert samples[0].finish_reason == "length"
    assert samples[0].hit_max_tokens
    assert samples[0].token_count == 3
    assert samples[1].finish_reason == "stop"
    assert not samples[1].hit_max_tokens
    assert samples[1].token_count == 2


def test_checkpoint_replay_preserves_generation_metadata():
    raw = '{"schema_version":2'
    row = {"selection_trace": {
        "samples_received": 1,
        "attempts": [{
            "index": 1,
            "raw_response": raw,
            "raw_truncated": False,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "generation_finish_reason": "length",
            "generation_tokens": 512,
            "generation_max_tokens": 512,
        }],
    }}
    sample = _saved_samples(row, 1)[0]
    assert isinstance(sample, GenerationSample)
    assert sample.finish_reason == "length"
    assert sample.token_count == 512
    assert sample.max_tokens == 512
    assert sample.hit_max_tokens
