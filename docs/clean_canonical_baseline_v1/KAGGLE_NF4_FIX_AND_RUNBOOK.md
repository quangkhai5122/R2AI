# Kaggle runtime NF4 - canonical status

## Current outcome

The clean schema-9 run completed with the base
Qwen/Qwen2.5-Coder-14B-Instruct checkpoint quantized at load time by
bitsandbytes NF4. AWQ/GPTQ was not used.

Frozen semantic settings are Selection v2, k=0, n=2, temperature=0.2,
max_tokens=512, max_input_tokens=5000, batch_size=4, seed=13, and
llm_target=all.

## Historical AWQ incident

An earlier notebook selected a pre-quantized 7B AWQ checkpoint. Transformers
then required gptqmodel, which was absent. Installing gptqmodel was rejected
because it would create a different backend and a compatibility-heavy runtime.

The canonical fix is to use the base checkpoint and pass load_4bit so
bitsandbytes performs NF4 quantization at runtime. The clean NF4 launcher rejects
AWQ/GPTQ model names.

This section is incident history. It does not mean the completed run used 7B or
AWQ.

## Canonical files

- payload builder: scripts/59_make_clean_payload_v5.py;
- NF4 launcher: kaggle/kaggle_clean_codegen_nf4.py;
- validator source: kaggle/validate_clean_codegen_v2.py;
- run config:
  configs/clean_canonical_baseline_v1/b1_select_v2_14b_nf4.json;
- notebook: kaggle/vifinqa-clean-canonical-b1-14b-nf4.ipynb;
- completed run: artifacts/clean_v1/b1_nf4;
- run record:
  experiments/clean_canonical_baseline_v1/runs/b1_14b_nf4_2026-08-21.json.

Earlier v3/v4 payload builders and old clean-canonical notebooks are historical
and must not be used.

## Rebuild and upload

    python scripts/59_make_clean_payload_v5.py --dataset-id lequangkhai5122005/vifinqa-clean-canonical-v1
    kaggle datasets version -p artifacts/clean_v1/kaggle_payload -m "schema9: Qwen 14B runtime NF4 canonical" --dir-mode zip

Import the canonical 14B notebook and attach only the newest dataset version.

## Actual completed runtime

- Python 3.12.13;
- torch 2.10.0+cu128;
- transformers 5.0.0;
- accelerate 1.13.0;
- bitsandbytes 0.50.1;
- CUDA 12.8;
- two Tesla T4 GPUs;
- model revision aedcc2d42b622764e023cf882b6652e646b95671.

The old notebook text said transformers<5 but only enforced a minimum, so
transformers 5.0.0 was the actual successful runtime. The canonical notebook
checks the full version specification and serializes run_config_nf4.json.

## Completed artifact checks

- 1,012 unique aligned records;
- 1,012 completed LLM attempts;
- one run signature;
- all answers finite;
- codegen hash matches both audits and submission handoff;
- folder and ZIP results replay codegen answers exactly;
- archive layout is safe;
- submission hash matches the handoff manifest.

The audit contains no answer labels, so these checks establish integrity, not
accuracy.

## Required future handoff

Keep the following together:

- runtime_preflight_nf4.json;
- runtime_smoke_nf4.json;
- runtime_full_nf4.json;
- run_config_nf4.json;
- codegen_results_nf4.jsonl;
- codegen_audit_nf4.json;
- submission_manifest_nf4.json;
- submission.zip;
- executed notebook or full log.

The completed 2026-08-21 run predates run_config serialization and did not
export elapsed time. Its immutable run record explicitly documents those gaps.
