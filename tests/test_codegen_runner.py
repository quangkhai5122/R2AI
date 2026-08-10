from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vifinqa.codegen import generate
from vifinqa.codegen.llm_client import HfBatchClient
from vifinqa.utils.viet_text import fuzzy_scorer_provenance


ROOT = Path(__file__).resolve().parents[1]


def _load_kaggle_runner():
    path = ROOT / "kaggle" / "kaggle_codegen.py"
    spec = importlib.util.spec_from_file_location("kaggle_codegen_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_payload_builder():
    path = ROOT / "scripts" / "04_make_kaggle_payload.py"
    spec = importlib.util.spec_from_file_location("payload_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeStore:
    def __init__(self, *_args, **_kwargs):
        pass


class _FakeBundle:
    def __init__(self, rec, _store, _k, run_signature=""):
        self.id = rec["id"]
        self.question = rec["question"]
        self.run_signature = run_signature
        self.tables = [{"var": "df1", "report_id": "AAA_2024_consolidated",
                        "table_pos": 0}]
        self.dfs = {"df1": object()}
        self.shortlist_trace = {"candidate_count": 1}

    def prompt_messages(self):
        return [{"role": "user", "content": self.question}]

    def select_messages(self, _encoder=None):
        return self.prompt_messages()

    def used_vars(self, _code):
        return [self.tables[0]]


class _Client:
    name = "fake"

    def __init__(self, fail_on_call=0):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def chat_batch(self, conversations, **_kwargs):
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise RuntimeError("simulated backend failure")
        return [["float(df1)"] for _ in conversations]


def _exec_ok(code, _dfs):
    return {"status": "ok", "value": 1.0, "error": None}


def _exec_fail(_code, _dfs):
    return {"status": "error", "value": None, "error": "invalid code"}


class CodegenCheckpointTests(unittest.TestCase):
    def _recs(self, path: Path, n=5):
        with path.open("w", encoding="utf-8") as f:
            for i in range(1, n + 1):
                f.write(json.dumps({"id": i, "question": f"q{i}"}) + "\n")

    def _run(self, retrieval, output, client, signature="sig-a", **kwargs):
        exec_fn = kwargs.pop("exec_fn", _exec_ok)
        with patch.object(generate, "Store", _FakeStore), \
                patch.object(generate, "QuestionBundle", _FakeBundle), \
                patch.object(generate, "run_code", exec_fn):
            generate.run_codegen(
                retrieval, Path("unused"), output, client,
                checkpoint_every=2, debug_rounds=0,
                use_rule_fallback=False, run_signature=signature, **kwargs,
            )

    def test_resume_requires_matching_signature(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            first = _Client()
            self._run(retrieval, output, first)
            self.assertEqual(first.calls, 3)

            same = _Client(fail_on_call=1)
            self._run(retrieval, output, same)
            self.assertEqual(same.calls, 0)

            changed = _Client()
            self._run(retrieval, output, changed, signature="sig-b")
            self.assertEqual(changed.calls, 3)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({r["run_signature"] for r in rows}, {"sig-b"})
            self.assertTrue(all(
                r["llm_attempt_status"] == "completed" for r in rows
            ))

    def test_completed_code_failures_are_not_requeued(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval, n=2)
            first = _Client()
            self._run(retrieval, output, first, exec_fn=_exec_fail)
            self.assertEqual(first.calls, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(all(r["source"] == "none" for r in rows))
            self.assertTrue(all(
                r["llm_attempt_status"] == "completed" for r in rows
            ))

            same = _Client(fail_on_call=1)
            self._run(retrieval, output, same, exec_fn=_exec_fail)
            self.assertEqual(same.calls, 0)

    def test_selection_rule_kept_rejected_and_no_candidates_resume(self):
        def fake_rule(bundle, _encoder=None):
            row = generate._final(bundle, 1.0, "float(df1)", "rule")
            row["detail_conf"] = 99.0
            return row

        def fake_selection(bundle, _samples, _encoder=None, return_trace=False):
            outcome = {1: "accepted", 2: "rejected", 3: "no_candidates"}[
                bundle.id
            ]
            trace = {"schema_version": 1, "outcome": outcome,
                     "rejection_counts": {}}
            record = None
            if outcome == "accepted":
                record = generate._final(
                    bundle, 2.0, "float(df1) + 1", "llm_select",
                )
            return (record, trace) if return_trace else record

        def run_selection(retrieval, output, client, signature):
            with patch.object(generate, "Store", _FakeStore), \
                    patch.object(generate, "QuestionBundle", _FakeBundle), \
                    patch.object(generate, "_rule_result", fake_rule), \
                    patch.object(generate, "_selection_result", fake_selection):
                generate.run_codegen(
                    retrieval, Path("unused"), output, client,
                    checkpoint_every=3, debug_rounds=0,
                    use_rule_fallback=True, run_signature=signature,
                    llm_mode="select",
                )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval, n=3)
            first = _Client()
            run_selection(retrieval, output, first, "select-sig-a")
            self.assertEqual(first.calls, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["source"] for r in rows], ["rule", "rule", "rule"])
            self.assertEqual(
                [r["selection_trace"]["outcome"] for r in rows],
                ["accepted", "rejected", "no_candidates"],
            )
            self.assertTrue(all(
                r["llm_attempt_status"] == "completed" for r in rows
            ))

            same = _Client(fail_on_call=1)
            run_selection(retrieval, output, same, "select-sig-a")
            self.assertEqual(same.calls, 0)

            changed = _Client()
            run_selection(retrieval, output, changed, "select-sig-b")
            self.assertEqual(changed.calls, 1)

    def test_matching_legacy_selection_trace_is_completed(self):
        row = {
            "source": "rule", "run_signature": "legacy-sig",
            "selection_trace": {"schema_version": 1, "outcome": "rejected"},
        }
        self.assertTrue(generate._has_completed_llm_attempt(row))

    def test_checkpoint_survives_later_chunk_failure(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self._run(retrieval, output, _Client(fail_on_call=2))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(sum(r["source"] == "llm" for r in rows), 2)
            self.assertEqual(sum(r["source"] == "none" for r in rows), 3)
            self.assertEqual(sum(
                r.get("llm_attempt_status") == "completed" for r in rows
            ), 2)

    def test_expired_budget_keeps_complete_rule_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            client = _Client(fail_on_call=1)
            self._run(retrieval, output, client, time_budget_s=1e-12)
            self.assertEqual(client.calls, 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["source"] == "none" for r in rows))
            self.assertTrue(all("llm_attempt_status" not in r for r in rows))


class _SelectionTraceBundle:
    def __init__(self, candidates=None):
        self.id = 7
        self.question = "test selection trace"
        self.route = {"unit_scale": 1.0, "output_type": "number"}
        self.run_signature = "trace-sig"
        self._candidates = candidates if candidates is not None else [
            SimpleNamespace(
                var="df1", row=1, label="Revenue", col=1, col_name="2024",
                value=100.0, unit_scale=1.0, score=85.0,
                report_id="AAA_2024", table_pos=0, code="10",
            ),
            SimpleNamespace(
                var="df1", row=2, label="Zero base", col=2, col_name="2023",
                value=0.0, unit_scale=1.0, score=80.0,
                report_id="AAA_2024", table_pos=0, code="20",
            ),
        ]

    def shortlist(self, _encoder=None, top_n=12):
        return self._candidates[:top_n]

    def used_vars(self, _code):
        return [{"var": "df1", "report_id": "AAA_2024", "table_pos": 0}]


class SelectionTraceTests(unittest.TestCase):
    def test_attempt_taxonomy_and_first_valid_decision_are_recorded(self):
        bundle = _SelectionTraceBundle()
        samples = [
            "not json",
            '{"op":"difference","operands":[1]}',
            '{"op":"ratio","operands":[1,2]}',
            '{"op":"lookup","operands":[1]}',
            '{"op":"lookup","operands":[2]}',
        ]
        replay = {"status": "ok", "value": 100.0, "error": None,
                  "semantic": {"ok": True}}
        with patch.object(generate, "_run_validated", return_value=replay):
            rec, trace = generate._selection_result(
                bundle, samples, None, return_trace=True,
            )

        self.assertEqual(rec["answer"], 100.0)
        self.assertEqual(trace["schema_version"], 1)
        self.assertEqual(trace["outcome"], "accepted")
        self.assertEqual(trace["accepted_attempt"], 4)
        self.assertEqual(trace["attempts_evaluated"], 4)
        self.assertEqual(
            [a["reason_code"] for a in trace["attempts"]],
            ["parse_error", "invalid_selection", "synthesis_error", "",
             "not_evaluated_after_acceptance"],
        )
        self.assertEqual(trace["attempts"][3]["stage"], "accepted")
        self.assertIn("raw_sha256", trace["attempts"][0])

    def test_execution_semantic_and_replay_rejections_are_distinct(self):
        bundle = _SelectionTraceBundle()
        sample = ['{"op":"lookup","operands":[1]}']
        cases = [
            ({"status": "error", "value": None, "error": "missing cell"},
             "execution", "execution_failed"),
            ({"status": "semantic_error", "value": 100.0,
              "error": "answer is not derived from any dataframe",
              "semantic": {"ok": False, "errors": ["ungrounded"]}},
             "semantic", "semantic_validation_failed"),
            ({"status": "ok", "value": 101.0, "error": None,
              "semantic": {"ok": True}},
             "replay", "answer_mismatch"),
        ]
        for replay, stage, reason_code in cases:
            with self.subTest(stage=stage), \
                    patch.object(generate, "_run_validated", return_value=replay):
                rec, trace = generate._selection_result(
                    bundle, sample, None, return_trace=True,
                )
                self.assertIsNone(rec)
                self.assertEqual(trace["outcome"], "rejected")
                self.assertEqual(trace["attempts"][0]["stage"], stage)
                self.assertEqual(trace["attempts"][0]["reason_code"], reason_code)

    def test_raw_response_is_sanitized_bounded_and_hashed(self):
        raw = "\x00" + "x" * (generate._SELECTION_RAW_MAX_CHARS + 25)
        attempt = generate._attempt_record(1, raw)
        self.assertNotIn("\x00", attempt["raw_response"])
        self.assertEqual(len(attempt["raw_response"]),
                         generate._SELECTION_RAW_MAX_CHARS)
        self.assertTrue(attempt["raw_truncated"])
        self.assertEqual(attempt["raw_chars"], len(raw))
        self.assertEqual(len(attempt["raw_sha256"]), 64)

    def test_explicit_none_and_empty_shortlist_are_auditable(self):
        rec, trace = generate._selection_result(
            _SelectionTraceBundle(), ['{"op":"none","operands":[]}'], None,
            return_trace=True,
        )
        self.assertIsNone(rec)
        self.assertEqual(trace["attempts"][0]["reason_code"], "model_none")

        rec, trace = generate._selection_result(
            _SelectionTraceBundle(candidates=[]), ["anything"], None,
            return_trace=True,
        )
        self.assertIsNone(rec)
        self.assertEqual(trace["outcome"], "no_candidates")
        self.assertEqual(trace["attempts"][0]["reason_code"], "no_candidates")

    def test_trace_survives_flush_and_resume_loader(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "trace.jsonl"
            trace = {"schema_version": 1, "outcome": "rejected",
                     "attempts": [{"reason_code": "parse_error"}]}
            result = {"id": 7, "source": "none", "selection_trace": trace}
            generate._flush(out, [{"id": 7}], {7: result})
            self.assertEqual(generate._load_previous(out)[7]["selection_trace"],
                             trace)


class HfOomRetryTests(unittest.TestCase):
    def test_cuda_oom_halves_batch_and_retries_without_skipping(self):
        class FakeTensor:
            def __init__(self, batch):
                self.batch = batch
                self.shape = (batch, 3)

        class FakeEncoding(dict):
            def __init__(self, batch):
                super().__init__(input_ids=FakeTensor(batch))

            def to(self, _device):
                return self

        class FakeOutput:
            def __init__(self, size):
                self.size = size

            def __getitem__(self, _key):
                return [None] * self.size

        class FakeTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, conversation, **_kwargs):
                return conversation[0]["content"]

            def __call__(self, texts, **_kwargs):
                return FakeEncoding(len(texts))

            def batch_decode(self, seqs, **_kwargs):
                return [f"code-{i}" for i in range(len(seqs))]

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                batch = kwargs["input_ids"].batch
                self.calls.append(batch)
                if batch > 2:
                    raise RuntimeError("CUDA out of memory")
                return FakeOutput(batch * kwargs["num_return_sequences"])

        class FakeCuda:
            def __init__(self):
                self.clears = 0

            def is_available(self):
                return True

            def empty_cache(self):
                self.clears += 1

        @contextmanager
        def inference_mode():
            yield

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda()
        fake_torch.inference_mode = inference_mode
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda **_kwargs: SimpleNamespace(
            update=lambda _n: None, close=lambda: None
        )

        client = HfBatchClient.__new__(HfBatchClient)
        client.batch_size = 4
        client.max_input = 100
        client.device = "cuda:0"
        client.tok = FakeTokenizer()
        client.model = FakeModel()
        conversations = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
        with patch.dict(sys.modules, {"torch": fake_torch, "tqdm": fake_tqdm}):
            results = client.chat_batch(conversations, n=1, temperature=0.7,
                                        max_tokens=32)
        self.assertEqual(client.model.calls, [4, 2, 2])
        self.assertEqual(fake_torch.cuda.clears, 1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(len(group) == 1 for group in results))


class PayloadManifestTests(unittest.TestCase):
    def test_manifest_verification_detects_tampering(self):
        runner = _load_kaggle_runner()
        with tempfile.TemporaryDirectory() as td:
            payload = Path(td)
            rels = [
                "retrieval.jsonl", "store/reports.parquet",
                "code/kaggle_codegen.py", "code/vifinqa/codegen/generate.py",
                "code/vifinqa/codegen/llm_client.py",
                "code/vifinqa/codegen/prompts.py",
                "code/vifinqa/codegen/executor.py",
                "code/vifinqa/codegen/selection.py",
                "code/vifinqa/retrieval/shortlist.py",
                "code/vifinqa/utils/viet_text.py",
            ]
            hashes = {}
            for rel in rels:
                path = payload / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(rel.encode())
                hashes[rel] = hashlib.sha256(rel.encode()).hexdigest()
            manifest = {"schema_version": runner.PAYLOAD_SCHEMA_VERSION,
                        "fuzzy_scorer": fuzzy_scorer_provenance(),
                        "files": hashes}
            (payload / runner.MANIFEST_NAME).write_text(json.dumps(manifest))
            checked, digest = runner.verify_payload(payload, payload / "code")
            self.assertEqual(checked["schema_version"], runner.PAYLOAD_SCHEMA_VERSION)
            self.assertEqual(len(digest), 64)

            (payload / "code/vifinqa/codegen/generate.py").write_text("tampered")
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                runner.verify_payload(payload, payload / "code")

    def test_manifest_rejects_fuzzy_scorer_drift(self):
        runner = _load_kaggle_runner()
        with tempfile.TemporaryDirectory() as td:
            payload = Path(td)
            files = {"placeholder": hashlib.sha256(b"x").hexdigest()}
            (payload / "placeholder").write_bytes(b"x")
            manifest = {
                "schema_version": runner.PAYLOAD_SCHEMA_VERSION,
                "fuzzy_scorer": {"backend": "rapidfuzz", "version": "3.14.5"},
                "files": files,
            }
            (payload / runner.MANIFEST_NAME).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(SystemExit, "fuzzy scorer mismatch"):
                runner.verify_payload(payload)

    def test_run_signature_fingerprints_fuzzy_scorer(self):
        runner = _load_kaggle_runner()
        args = SimpleNamespace(
            backend="hf", model="model", n=1, temperature=0.7,
            max_tokens=96, debug_rounds=1, k=0, rule_first=False,
            llm_target="all", llm_mode="select", use_dense=False,
            dense_model="", seed=13, load_4bit=True,
            max_input_tokens=5000, quantization="awq", tp=2,
            max_model_len=6144, rescue_no_candidates=False,
            rescue_table_k=20, rescue_min_score=28.0,
            batch_size=4, checkpoint_every=32, limit=0,
        )
        scorer = fuzzy_scorer_provenance()
        self.assertEqual(runner.FUZZY_SCORER, scorer)
        sig = runner.run_signature(args, "manifest-digest", scorer)
        changed = runner.run_signature(
            args, "manifest-digest", {**scorer, "version": "2"},
        )
        self.assertEqual(len(sig), 64)
        self.assertNotEqual(sig, changed)

        for field, value in {
            "batch_size": 2,
            "checkpoint_every": 16,
            "limit": 12,
        }.items():
            with self.subTest(field=field):
                changed_args = SimpleNamespace(**{
                    **vars(args), field: value,
                })
                self.assertNotEqual(
                    sig,
                    runner.run_signature(
                        changed_args, "manifest-digest", scorer,
                    ),
                )

    def test_notebook_is_valid_json_and_has_no_hot_patch(self):
        notebook_path = ROOT / "kaggle" / "vifinqa-codegen.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertEqual(notebook["nbformat"], 4)
        self.assertNotIn("%%writefile", source)
        self.assertIn("payload-manifest.json", source)
        self.assertIn("difflib.SequenceMatcher", source)

    def test_payload_builder_hashes_small_payload_and_preserves_id(self):
        builder = _load_payload_builder()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for rel, content in {
                "retrieval.jsonl": b"{}\n",
                "code/kaggle_codegen.py": b"runner",
                "store/reports.parquet": b"parquet",
            }.items():
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            manifest = builder._build_manifest(out)
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["fuzzy_scorer"], fuzzy_scorer_provenance())
            self.assertEqual(set(manifest["files"]), {
                "retrieval.jsonl", "code/kaggle_codegen.py",
                "store/reports.parquet",
            })
            self.assertEqual(
                builder._dataset_id("", {"id": "owner/vifinqa-payload"}, "ignored"),
                "owner/vifinqa-payload",
            )

    def test_payload_dry_run_validates_without_replacing_output(self):
        builder = _load_payload_builder()
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            store = temp / "store"
            store.mkdir()
            (store / "reports.parquet").write_bytes(b"small")
            retrieval = temp / "retrieval.jsonl"
            retrieval.write_text("{}\n")
            out = ROOT / "artifacts" / f"dry-run-{temp.name}"
            self.assertFalse(out.exists())
            argv = [
                "04_make_kaggle_payload.py", "--store-dir", str(store),
                "--retrieval", str(retrieval), "--out", str(out),
                "--dataset-id", "owner/vifinqa-payload", "--dry-run",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()) as stdout:
                builder.main()
            self.assertIn("dry-run OK", stdout.getvalue())
            self.assertFalse(out.exists())

    def test_payload_output_is_confined_to_artifacts(self):
        builder = _load_payload_builder()
        safe = ROOT / "artifacts" / "some-payload"
        self.assertEqual(builder._safe_output_path(ROOT, safe), safe.resolve())
        with self.assertRaisesRegex(SystemExit, "outside"):
            builder._safe_output_path(ROOT, ROOT / "kaggle")
        with self.assertRaisesRegex(SystemExit, "root itself"):
            builder._safe_output_path(ROOT, ROOT / "artifacts")


if __name__ == "__main__":
    unittest.main()
