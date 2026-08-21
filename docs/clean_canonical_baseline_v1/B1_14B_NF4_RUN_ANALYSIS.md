# B1 Qwen 14B NF4 run analysis - 2026-08-21

## Executive conclusion

The run completed and is sound enough to freeze as a candidate. All 1,012 IDs are valid, all LLM attempts completed, questions align with retrieval, codegen replays exactly into the submission, and every provenance/hash/archive check passed. The effective runtime was Qwen/Qwen2.5-Coder-14B-Instruct at revision aedcc2d42b622764e023cf882b6652e646b95671, loaded through Hugging Face and bitsandbytes NF4 on two Tesla T4 GPUs.

The run does not prove that B1 is more accurate than B0. This audit uses no answer labels. B1 increases status=ok coverage from 743 to 763 records (+20, +1.976 percentage points), but it also changes 33 answers that were already status=ok in B0. Without a frozen labeled OOD set, the 53 final LLM changes cannot be classified as improvements or regressions.

The observed bottleneck is not primarily JSON parsing or model scale. Selection v2 produces no shortlist for 546/1,012 records (53.95%), while 684/1,012 records have incomplete semantic fact slots. Of 423 evaluated attempts that were rejected, 255 (60.3%) fail grounding. The next scientific step should therefore be a locked OOD gate followed by a guarded evidence/shortlist-rescue ablation, not another near-duplicate model run selected from public behavior.

## Scope and sources

- Run artifacts: [artifacts/clean_v1/b1_nf4](../../artifacts/clean_v1/b1_nf4)
- Canonical run record: [b1_14b_nf4_2026-08-21.json](../../experiments/clean_canonical_baseline_v1/runs/b1_14b_nf4_2026-08-21.json)
- Reproducible audit: [scripts/63_audit_b1_nf4_run.py](../../scripts/63_audit_b1_nf4_run.py)
- Generated audit output: artifacts/clean_v1/b1_nf4/analysis_recomputed.json
- B0 control: artifacts/clean_v1/b0_results.jsonl
- Retrieval snapshot: artifacts/clean_v1/retrieval.jsonl

The audit uses provenance, traces, execution status, B0 outputs, and submission replay. It does not use answer labels, leaderboard scores, ID allowlists, or official-derived gold. All findings below concern integrity, coverage, and failure modes; they are not estimates of competition accuracy.

## Overall assessment

**Share with caveats.**

The artifact is technically ready to preserve and can be submitted as a distinct candidate. It is not yet scientifically justified as more accurate than B0 because independent OOD accuracy evidence is absent.

## Observed provenance

| Field | Observed value |
|---|---|
| Model | Qwen/Qwen2.5-Coder-14B-Instruct |
| Model revision | aedcc2d42b622764e023cf882b6652e646b95671 |
| Runtime profile | hf-bitsandbytes-nf4-v1 |
| Quantization | NF4 4-bit, double quant, fp16 compute |
| GPU | 2 x Tesla T4; 15,636,037,632 bytes each; SM 7.5 |
| Python / CUDA | 3.12.13 / 12.8 |
| torch | 2.10.0+cu128 |
| transformers | 5.0.0 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.50.1 |
| Payload stable verified hash | 146898f6eb689c24ec4c59d310aaac522320df457ad8a7f24a606dd4dba666a8 |
| Run signature | 4b2ea9cca04e03a67f5087a9f919725657a4a1e7d91f696395a0a25bb1c2c569 |
| Codegen SHA-256 | a8c2b93279daa7099ce0fdcead123cf9df134687fefeb48b484dd66267f9371c |
| Submission SHA-256 | c98f1859e41a924458abfc7f5b2f2673e028136e7a73b2cb04c6cb84467cb75c |

Before any source edits in this session, nine NF4/provenance tests passed and verified that packaged run code matched checkout a5a540b0ddeee6041b10abfc872e3d04617a0120. This is post-hoc evidence; the remote artifact did not embed the Git commit directly.

## Integrity and submission replay

Every checked gate passed:

- codegen, retrieval, B0, and submission contain the same 1,012 unique IDs;
- codegen questions align with retrieval questions;
- all 1,012 LLM attempts are marked completed;
- every answer is finite;
- the codegen contains exactly one non-empty run signature;
- the codegen hash matches both the Kaggle audit and submission handoff;
- local_audit.json equals codegen_audit_nf4.json;
- folder and ZIP results.json answers replay exactly from codegen;
- the ZIP has 1,522 members and no absolute or path-traversal member;
- the submission ZIP hash matches its manifest.

A finite answer is not equivalent to a solved question. The 249 failed records are serialized with answer=0.0. They are schema-valid placeholders and an accuracy risk.

## B1 versus B0

| Metric | B0 | B1 14B NF4 | Delta |
|---|---:|---:|---:|
| status=ok | 743 | 763 | +20 |
| status=failed | 269 | 249 | -20 |
| status=ok coverage | 73.42% | 75.40% | +1.98 pp |
| final source=LLM | 0 | 53 | +53 |

Exact status transitions:

- 249 records: failed -> failed;
- 20 records: failed -> ok;
- 743 records: ok -> ok;
- 0 records: ok -> failed;
- among the 743 records that remain ok, 33 answers change.

The 53 final LLM outputs consist of exactly 20 cases where the rule produced nothing and 33 cases where a weak rule disagreed with the LLM. This makes execution coverage monotone, but it does not make accuracy monotone.

## Selection v2 and arbitration

| Outcome | Records | Share |
|---|---:|---:|
| accepted | 263 | 25.99% |
| rejected | 203 | 20.06% |
| no_candidates | 546 | 53.95% |

Of 263 accepted LLM programs:

- 53 become the final answer;
- 169 agree with the rule, so arbitration keeps the rule;
- 41 disagree with a sufficiently confident rule, so arbitration keeps the rule.

Accepted LLM programs must not be reported as final LLM contribution. Final LLM contribution is 53/1,012 (5.24%).

Of 546 no-candidate records, 345 are still solved by rule/rule_composite and 201 fail. Of 203 rejected records, 155 are still solved by rule/rule_composite and 48 fail. The 249 final failures therefore decompose exactly into 201 no-candidate failures and 48 rejected-program failures.

## Generation and validator behavior

The notebook received all 2,024 samples, exactly n=2 for 1,012 records. Only 686 attempts were evaluated; the rest were skipped because the shortlist was empty or an earlier attempt had already been accepted.

There were 423 evaluated rejections:

| Rejection | Attempts | Share of rejections |
|---|---:|---:|
| grounding_error | 255 | 60.3% |
| model_none | 73 | 17.3% |
| generation_truncated | 34 | 8.0% |
| binding_error | 27 | 6.4% |
| schema_error | 12 | 2.8% |
| output_type_error | 9 | 2.1% |
| all other reasons | 13 | 3.1% |

Only two attempts fail parsing and 12 fail schema validation. Improving JSON formatting alone is unlikely to address the dominant failure. Thirty-five of 2,024 generations end by length (1.73%), and 34 are rejected as truncated. Raising max_tokens can affect a small tail but not the 546 empty shortlists.

A clear efficiency issue exists: the model generates 1,092 samples for the 546 no-candidate records, yet none of those attempts is evaluated. A runtime-only candidate can skip generation after an empty shortlist is known while preserving final-answer semantics. It needs a new run profile/signature and an exact-output-equivalence test.

## Structural diagnostic slices

These aggregates are hypothesis generators only. They must not be used for per-ID fixes or threshold selection on the 1,012 official questions.

- Routes with a canonical metric key: 655/792 ok (82.70%).
- Canonical misses: 108/220 ok (49.09%).
- lookup plan: 391/468 ok (83.55%); 202 accepted.
- ranking plan: 129/214 ok (60.28%); 155 no-candidate (72.43%).
- average plan: 48/70 ok (68.57%); 51 no-candidate (72.86%).
- count output: 8/23 ok (34.78%).
- year output: 7/53 ok (13.21%); all seven final answers come from the LLM.
- number output: 521/646 ok (80.65%).

Canonical misses have much lower status coverage, but their no-candidate rate is only modestly higher (56.36% versus 53.28%). This suggests two separable issues: ontology/route coverage weakens the rule path, while shortlist guards discard many candidate rows even when a canonical metric is available.

Counters such as metric_mismatch, route_not_grounded, and year_ambiguous count rejected candidates, not questions. They must not be divided by 1,012 and reported as question-level failure rates.

## Reproducibility and metadata drift

### Uploaded default was 7B; effective runtime was 14B

The uploaded payload declares Qwen 7B as default_model, while the notebook runtime report, Hugging Face repository record, smoke report, and full report all identify Qwen 14B. The run signature includes the effective model, so 14B checkpoints cannot be mixed with 7B checkpoints. This is metadata drift, not a mixed-model run.

Canonical source/config/notebook defaults after this session use 14B NF4. The immutable run record keeps both the declared payload default and effective model so history is not rewritten.

### AWQ is historical only

AWQ/GPTQ was not used in this run. The active runtime loads the base checkpoint and quantizes it at load time with bitsandbytes NF4. gptqmodel and autoawq are not canonical dependencies.

### The transformers bound was not enforced

The old notebook declared transformers>=4.44,<5 but only checked the minimum. An installed transformers 5.0.0 therefore passed and the run succeeded. The canonical notebook now checks the full version specification and serializes a run_config artifact. This run must be documented as transformers 5.0.0, not 4.x.

### Smoke and full differ on one overlap record

Two of the three overlapping smoke records match answer/source/outcome; one differs. Smoke and full deliberately have different signatures because limit, checkpoint grouping, and RNG grouping differ, and temperature is 0.2. This is not checkpoint contamination. It is evidence that a three-record smoke does not measure stochastic full-run stability.

### Runtime duration is unavailable

Runtime JSON files are written before generation and contain no finished_at or elapsed duration. The executed notebook and complete log were not exported. Throughput and total runtime therefore cannot be recovered reliably from file timestamps.

## Recommended next program

### G3: freeze evaluation before changing the pipeline

The immediate priority is a source-derived OOD benchmark independent of the 1,012 official questions:

1. generate or label questions from report cells with direct provenance;
2. split by ticker/report/year so related documents cannot cross tune and locked sets;
3. freeze and hash the locked set before inspecting candidates;
4. report shortlist coverage, semantic-fact completeness, executable-program rate, answer accuracy, zero/failure rate, rule-versus-LLM disagreement, and stability across seeds;
5. use the 1,012 official records only as a crash/invariant regression after the candidate and thresholds are frozen.

The current trace identifies where the pipeline does not execute; it cannot identify which answer is correct.

### B2: guarded evidence/shortlist rescue

Only after G3 has a baseline, keep the 14B model, compiler, arbitration, and runtime fixed, and change evidence acquisition only when the strict shortlist is empty:

- retain hard ticker/entity, report-type, and period/year guards;
- widen lexical/canonical-component evidence in a separate rescue path;
- evaluate row-aware scoring as its own ablation;
- record strict/rescue provenance and reason codes;
- do not use an ID list or a threshold selected from official records.

Gate B2 on improved OOD fact recall and answer accuracy without exceeding grounding, unit, and year-error guardrails. It is the highest-priority hypothesis because 53.95% of this run has no shortlist.

### B3: typed structural specialists

If B2 improves evidence coverage but ranking/count/year/average remain weak on the locked OOD set, add operator-specific deterministic or typed planners. Do not add a second raw-Pandas executor and do not add per-ID exceptions. B3 must differ from B2 at the answer-architecture level, not by a few thresholds.

### Runtime-only efficiency candidate

Skipping LLM generation for a known-empty shortlist could avoid 1,092/2,024 samples in this run. Treat this as a cost/runtime optimization with exact-output-equivalence tests, not as a private-submission candidate because final semantics are unchanged.

## Five-slot private portfolio

Reserve the five slots for distinct hypotheses:

1. B0 deterministic clean control;
2. current B1 14B NF4 Selection v2;
3. B2 guarded evidence/shortlist rescue after the OOD gate;
4. B3 typed structural specialist after the OOD gate;
5. a low-error-correlation candidate: another <=15B model family or a preregistered conservative ensemble.

Do not reserve a submission slot for 7B merely because an old config named it. A 7B run is useful only as an offline cost/stability measurement; this branch has no completed clean 7B remote artifact.

## Current decision

- Freeze B1 14B NF4 without mutating its artifacts.
- Do not claim B1 accuracy superiority over B0.
- Do not tune further on official IDs.
- Implement G3/OOD evaluation next.
- After G3, prioritize one guarded B2 rescue over another model-scale variant.
- Keep B0 and B1 as independent portfolio anchors.

## Re-running the audit

From the repository root:

    python scripts/63_audit_b1_nf4_run.py --out artifacts/clean_v1/b1_nf4/analysis_recomputed.json

Every integrity_checks entry must be true. The generated output is not a gold evaluation and must not be used to infer leaderboard accuracy.
