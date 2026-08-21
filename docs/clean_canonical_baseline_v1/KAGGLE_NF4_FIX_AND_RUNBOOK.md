# Kaggle NF4 fix and runbook

## Outcome

The schema-9 clean payload now uses the unquantized
`Qwen/Qwen2.5-Coder-7B-Instruct` checkpoint and quantizes it at load time with
bitsandbytes NF4. The frozen configuration remains Selection v2, `k=0`,
`n=2`, temperature `0.2`, 512 output tokens, seed 13, and `llm_target=all`.
The model has 7.61B parameters, below the organizer-confirmed 15B ceiling.

Do not fix the original failure by installing `gptqmodel`. That would change
the runtime to a different pre-quantized backend and add a compatibility-heavy
dependency outside this baseline. The clean launcher rejects AWQ/GPTQ model
names so this failure cannot silently return.

## Root cause and additional audit findings

The failed notebook selected
`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` but did not pass `--load-4bit`. The current
Transformers loader selected a quantized-model backend that required
`gptqmodel`, which was absent in the Kaggle image.

The audit also found and fixed these independent risks:

1. The notebook had no dependency, CUDA, GPU, or model-access preflight.
2. It launched all 1,012 questions without an exact-path smoke test.
3. A time-budget checkpoint was structurally submittable even when many LLM
   attempts were still pending. The final validator now refuses submission
   until every expected attempt is complete.
4. The payload environment fingerprint describes the local build host, not the
   Kaggle GPU runtime. The notebook now writes separate runtime reports with
   package, CUDA, GPU, model, quantization, and payload-manifest provenance.
5. The old notebook did not verify unique IDs, question alignment, finite
   answers, a single run signature, Selection traces, archive layout, or the
   final ZIP hash.
6. An initial packaging approach replaced the canonical verifier under the
   same module name and could self-import. The final payload keeps the verifier
   and NF4 launcher side by side; a regression test imports and verifies the
   launcher from the built payload itself.

## Final files

- Payload builder: `scripts/59_make_clean_payload_v4.py`
- Kaggle NF4 launcher: `kaggle/kaggle_clean_codegen_nf4.py`
- Checkpoint validator: `kaggle/validate_clean_codegen.py`
- Frozen run config: `configs/clean_canonical_baseline_v1/b1_select_v2_7b_nf4.json`
- Notebook to import: `kaggle/vifinqa-clean-canonical-v1-nf4-v2.ipynb`
- Built payload: `artifacts/clean_v1/kaggle_payload`

The earlier `59_make_clean_payload_v3.py` and notebook without the `-v2`
suffix are superseded and must not be used.

## Rebuild and upload a new Kaggle dataset version

Run from the repository root:

```powershell
python scripts\59_make_clean_payload_v4.py --dataset-id lequangkhai5122005/vifinqa-clean-canonical-v1
kaggle datasets version -p artifacts\clean_v1\kaggle_payload -m "schema9: HF Qwen 7B + bitsandbytes NF4 + fail-closed QA" --dir-mode zip
```

Use `kaggle datasets version`, not `kaggle datasets create`, because this
dataset slug already exists. Do not append a trailing `.` to either command.

After the version finishes processing on Kaggle, import
`kaggle/vifinqa-clean-canonical-v1-nf4-v2.ipynb` and attach the latest version
of `lequangkhai5122005/vifinqa-clean-canonical-v1`.

## Kaggle settings and Run all

1. Enable a GPU accelerator. The HF/SDPA path is intended for Kaggle T4.
2. Enable Internet so the base Qwen checkpoint can be downloaded. If Internet
   must remain disabled, attach a local model dataset and set `VIFINQA_MODEL`
   to that model directory in the first code cell.
3. Run all cells. The dependency cell installs only missing or too-old NF4
   packages; it does not install `gptqmodel` or `autoawq`.
4. The smoke cell loads the real model and processes three questions through
   the exact Selection-v2 path. The full cell then runs all 1,012 questions.
5. If the full audit reports an incomplete checkpoint, rerun only the full-run
   cell and the cells below it. Exact-signature resume is enabled.
6. Download `submission_clean_nf4/submission.zip` only after the audit and ZIP
   validation cells pass.

Keep these files together with the submission:

- `runtime_preflight_nf4.json`
- `runtime_full_nf4.json`
- `codegen_audit_nf4.json`
- `submission_manifest_nf4.json`

## Verified locally

- Payload schema: 9
- Runtime profile: `hf-bitsandbytes-nf4-v1`
- Retrieval records: 1,012 unique records
- Valid canonical misses: 220, each with a lexical fallback
- Payload files: 282, including manifest-hashed NF4 launcher and validator
- Payload size: approximately 103.1 MB
- Public ID masks: absent/forbidden
- Official-derived gold artifacts: absent/forbidden
- Focused NF4/clean tests: 13 passed
- Full repository suite: 338 passed
- Final notebook: nbformat-valid, all six code cells compile

The actual Qwen download, 4-bit CUDA load, throughput, and 1,012-question
completion remain Kaggle-only checks because the local environment has no
Torch/CUDA stack. The smoke cell is the remote verification gate.
