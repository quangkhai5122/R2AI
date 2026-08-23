# Clean canonical baseline v1

> Branch-specific source of truth as of 2026-08-21. If older root-level or
> incident documentation conflicts with this page, use this page and
> RUNBOOK.md in this directory.

## Current state

G0-G2 are implemented. The first full clean B1 run has completed on Kaggle with
Qwen/Qwen2.5-Coder-14B-Instruct and runtime NF4. It is not a 7B or AWQ run.

- B0 deterministic: 1,012 records; 743 ok and 269 failed.
- B1 14B NF4: 1,012 records; 763 ok and 249 failed.
- B1 LLM attempts: 1,012 completed.
- B1 final LLM contribution: 53 records.
- B1 codegen SHA-256:
  a8c2b93279daa7099ce0fdcead123cf9df134687fefeb48b484dd66267f9371c
- B1 submission SHA-256:
  c98f1859e41a924458abfc7f5b2f2673e028136e7a73b2cb04c6cb84467cb75c

All integrity/replay checks pass. No answer labels were used in the run audit, so
these coverage counts do not establish accuracy or generalization.

## Canonical entrypoints

- Run analysis: B1_14B_NF4_RUN_ANALYSIS.md
- Operations: RUNBOOK.md
- Current run config:
  ../../configs/clean_canonical_baseline_v1/b1_select_v2_14b_nf4.json
- Immutable run record:
  ../../experiments/clean_canonical_baseline_v1/runs/b1_14b_nf4_2026-08-21.json
- Kaggle notebook:
  ../../kaggle/vifinqa-clean-canonical-b1-14b-nf4.ipynb
- NF4 launcher: ../../kaggle/kaggle_clean_codegen_nf4.py
- Payload builder: ../../scripts/59_make_clean_payload_v5.py
- Output audit: ../../scripts/63_audit_b1_nf4_run.py
- G3 evaluation runbook:
  ../g3a_evaluation_gate/G3B_RUNBOOK.md
- Frozen G3 session analysis:
  ../g3a_evaluation_gate/G3A_G3B_SESSION_RESULTS_2026-08-23.md

## Runtime contract

The canonical runtime loads the base Qwen 14B checkpoint through Hugging Face
and quantizes it at load time with bitsandbytes NF4. AWQ/GPTQ checkpoints,
gptqmodel, and autoawq are outside this path.

The completed run observed:

- model revision aedcc2d42b622764e023cf882b6652e646b95671;
- two Tesla T4 GPUs;
- Python 3.12.13 and CUDA 12.8;
- torch 2.10.0+cu128, transformers 5.0.0, accelerate 1.13.0, and
  bitsandbytes 0.50.1;
- payload schema 9 and validation profile clean-codegen-select-v2-v2.

The uploaded payload still declared 7B as its default, but the effective model
was explicitly overridden to 14B and is bound into the run signature. The run
record preserves both values. Future builds default to 14B.

## Anti-public-bias contract

The clean payload forbids target masks, ID allowlists, official-derived gold,
raw-code mode, and skipping manifest verification. Aggregate traces from the
1,012 official records may identify general failure classes, but they must not
select thresholds, per-ID patches, or submission variants.

G3A/G3B is now complete and frozen. G3A v1 remained byte-identical; the
separate extension and 109-question G3B corpus add typed/compositional,
scope/period, and OOD diagnostic coverage. The next change is G3C retrieval,
selected on tune and frozen before locked/hard evaluation.

## Candidate interpretation

B1 increases execution coverage by 20 records but changes 33 already-ok B0
answers. Without labels, neither set of changes can be called correct. Preserve
B0 and B1 as independent anchors.

The main trace bottleneck is evidence/shortlist coverage:

- 546/1,012 records have no Selection-v2 candidates;
- 684/1,012 have incomplete semantic fact slots;
- grounding_error is 255/423 evaluated rejections;
- only 53/1,012 final answers use the LLM.

Do not spend another private slot on a near-duplicate 7B/14B scale comparison
unless a locked OOD experiment justifies it.

## Historical material

Older AWQ failure notes, validator hotfixes, superseded notebooks, and payload
builder revisions are retained only under history locations. They are evidence
about incidents, not runnable instructions.
