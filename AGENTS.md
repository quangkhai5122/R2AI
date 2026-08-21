# R2AI clean-canonical branch handoff

These instructions apply to the entire repository.

## Current source of truth

This checkout is the clean-canonical-baseline-v1 workstream. Root README.md,
RUNBOOK.md, and CLAUDE.md contain substantial legacy schema-8/P2.x history and
must not override the branch-specific files below:

- docs/clean_canonical_baseline_v1/README.md
- docs/clean_canonical_baseline_v1/RUNBOOK.md
- docs/clean_canonical_baseline_v1/B1_14B_NF4_RUN_ANALYSIS.md
- experiments/clean_canonical_baseline_v1/registry.json
- experiments/clean_canonical_baseline_v1/runs/b1_14b_nf4_2026-08-21.json

## Completed clean run

B1 completed on Kaggle with Qwen/Qwen2.5-Coder-14B-Instruct, Hugging Face, and
bitsandbytes runtime NF4. It is not a 7B or AWQ run. Preserve
artifacts/clean_v1/b1_nf4 as immutable run evidence.

Do not infer answer accuracy from status=ok, LLM acceptance, or the integrity
audit. No answer labels were used. Keep verified findings, interpretations, and
pending scientific claims separate.

## Active files

- config: configs/clean_canonical_baseline_v1/b1_select_v2_14b_nf4.json
- notebook: kaggle/vifinqa-clean-canonical-b1-14b-nf4.ipynb
- NF4 launcher: kaggle/kaggle_clean_codegen_nf4.py
- payload builder: scripts/59_make_clean_payload_v5.py
- output audit: scripts/63_audit_b1_nf4_run.py

Files under history directories are provenance only and are not runnable
instructions.
The schema-8 P2.x runner kaggle/kaggle_codegen.py and notebooks named
kaggle/vifinqa-codegen*.ipynb remain legacy paths and may mention AWQ. They are
not clean runtime entrypoints.


## Public-bias guard

Do not tune thresholds, add ID lists, or implement per-question fixes from the
1,012 official records. The next scientific milestone is G3: freeze a
source-derived OOD tune/locked benchmark split by ticker/report/year. Only then
test B2 guarded evidence/shortlist rescue.

The five private submissions must represent distinct preregistered hypotheses,
not five small score-driven edits.

## Engineering conventions

Keep datasets, retrieval, generation, validation, submission, configs, and
artifacts separate. Preserve deterministic seeds and hashes. Any new candidate
must use a new output directory and run signature; never overwrite B0 or B1.

Use pytest with -p no:cacheprovider and an artifacts-local --basetemp because the
repository-root .pytest_cache may have host ACL issues.
