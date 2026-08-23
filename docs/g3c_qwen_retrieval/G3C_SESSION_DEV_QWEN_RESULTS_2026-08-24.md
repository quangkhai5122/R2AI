# G3C real-Qwen dev evaluation and R4 candidate freeze - 2026-08-24

## Outcome

The full 54-question `primary_tune` run is valid scientific dev evidence under
the frozen G3C protocol. Strict import validation passed for all six stages,
the pre-registered gate selected R4, and the selected candidate is now frozen.

R4 is verified to improve retrieval and submission-shaped metrics over R0 on
the dev set. It is not yet verified to generalize: `primary_locked + hard` has
not been opened. Answer and execution accuracy did not improve, so this result
does not establish an end-to-end answer gain.

The promotion payload was built only after the passing freeze, validated, and
uploaded as the private Kaggle dataset:

    lequangkhai5122005/vifinqa-g3c-qwen-retrieval-promotion-v1

The one-shot promotion GPU run and locked local evaluator remain pending.

Subsequent status: promotion completed later on 2026-08-24. See
`G3C_SESSION_PROMOTION_RESULTS_2026-08-24.md`. The sentence above records the
state at this dev-freeze handoff and is not the current operational status.

## Frozen provenance

- protocol fingerprint:
  `af86c8ffc276cc0a92ceeb3cc0ddc3a7eeaa7b6a1e4430dc2cade8ac4c9621c5`;
- dev payload fingerprint:
  `5584010ab665510b592662cfc42327da19e1148c87fdef8b5cfe4155cdcbd9a4`;
- dev GPU run signature:
  `d6ef2ed02cbb8b4439d7d41b5b9d7cabde1f0c3738b6fe6ccd39c8ea1ae969e5`;
- GPU result manifest SHA-256:
  `eef2cec52129460e258186c122ea7270087ff2cf4a97763ffca0c730963dec0a`;
- selected stage: `R4`;
- candidate fingerprint:
  `1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a`;
- selected retrieval SHA-256:
  `4e956cfd1dcc1ab4aa4ab3696db06eee62d8aee2e025cc74e7bd8cb3cb4dfa19`;
- selected evaluation SHA-256:
  `23fdc30e8051e3dbdf56e1914165ce75b54f7ace229107ee31ccf77e7de248cb`;
- promotion payload fingerprint:
  `05a98f22ba6b69eccd91f854be1ddc78b0192c050ab7bdfdfdb639af735f97be`.

The payloads record Git HEAD
`47cef389c135d54cd068b3cb220861950a1d31ef` and `source_git_dirty=true` because
the G3C workstream was still uncommitted. This is a provenance limitation, but
runtime reproducibility does not rely on that commit alone: the protocol
behavior tree, complete payload file inventory, candidate freeze, models,
instructions, retrieval, and evaluations are independently hash-bound. Do not
rebuild the payload merely to obtain a clean Git flag.

The run used the pinned Qwen3-Embedding-4B and Qwen3-Reranker-4B revisions,
two Tesla T4 GPUs, Transformers 4.53.3, and full unquantized FP16 sequential
model loading. Every stage contains exactly 54 records and 20 candidates per
record. All hard ticker/report/year/scope violation counts are zero.

Measured runtime was 3,996.283 seconds (66.60 minutes), including 1,751.642
seconds for embeddings and 2,237.682 seconds for reranking. Peak GPU memory was
8,898,365,952 bytes (8.29 GiB).

## Commands completed

From the repository root, the following RUNBOOK operations completed:

    python scripts/77_g3c_validate_gpu_results.py --payload artifacts/g3c_v1/dev_payload_v2 --results artifacts/g3c_v1/dev_qwen_results --out artifacts/g3c_v1/dev_qwen_import_validation.json
    python scripts/78_g3c_evaluate_stages.py --mode dev --payload artifacts/g3c_v1/dev_payload_v2 --gpu-results artifacts/g3c_v1/dev_qwen_results --out-dir artifacts/g3c_v1/dev_local_eval
    python scripts/79_g3c_select_freeze.py --gpu-result-manifest artifacts/g3c_v1/dev_qwen_results/g3c_gpu_result_manifest.json --evaluation-index artifacts/g3c_v1/dev_local_eval/evaluation_index.json --selection-out artifacts/g3c_v1/dev_local_eval/g3c_dev_selection.json --freeze-out artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json
    python scripts/76_g3c_build_payload.py build --mode promotion --candidate-freeze artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json --out-dir artifacts/g3c_v1/promotion_payload
    python scripts/76_g3c_build_payload.py validate --payload artifacts/g3c_v1/promotion_payload

Kaggle CLI 2.2.4 failed before dataset creation when `-p` was a multi-component
relative Windows path: its temporary manifest name retained path separators.
No dataset existed after that failed attempt. Running the same immutable payload
from its own directory with `-p .` succeeded:

    cd artifacts/g3c_v1/promotion_payload
    python -m kaggle datasets create -p . --dir-mode zip

The remote dataset subsequently listed successfully and exposed the expected
payload files.

Post-handoff verification also passed:

- active protocol freeze validation;
- dev and promotion payload validation;
- R0 retrieval, codegen, and evaluation are byte-identical to the frozen G3B
  B0 dev control (SHA-256 values `635249...`, `587e72...`, and `45984b...`);
- targeted G3C suite: 22 passed in 2.55 seconds;
- full repository suite: 378 passed in 46.90 seconds.

## Dev metrics

All metrics below come from the frozen end-to-end evaluator. P/R/F2 are macro
metrics, TABLES MRR is MRR@5, Leaf is Leaf Recall@5, Full is
FullPlanCoverage, and Answer is exact/tolerance-aware answer accuracy.

| Stage | DOCS P | DOCS R | DOCS F2 | TABLES P | TABLES R | TABLES F2 | TABLES MRR | Leaf | Full | Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.882716 | 0.922840 | 0.903025 | 0.331041 | 0.817901 | 0.612058 | 0.789198 | 0.808642 | 0.685185 | 0.166667 |
| R0L | 0.978395 | 0.932099 | 0.935078 | 0.354938 | 0.864198 | 0.651224 | 0.812963 | 0.854938 | 0.740741 | 0.166667 |
| R1 | 0.984568 | 0.947531 | 0.948787 | 0.340123 | 0.830247 | 0.624410 | 0.751543 | 0.820988 | 0.703704 | 0.166667 |
| R2 | 0.978395 | 0.981481 | 0.980640 | 0.353086 | 0.882716 | 0.659753 | 0.778395 | 0.873457 | 0.777778 | 0.166667 |
| R3 | 0.978395 | 0.981481 | 0.980640 | 0.353086 | 0.882716 | 0.659753 | 0.779938 | 0.873457 | 0.777778 | 0.166667 |
| R4 | 0.981481 | 0.990741 | 0.988215 | 0.377160 | 0.929012 | 0.698987 | 0.810185 | 0.929012 | 0.888889 | 0.166667 |

R4 versus R0 therefore gives:

- DOCS F2: `+0.085190`;
- TABLES F2: `+0.086929`;
- Leaf Recall@5: `+0.120370`;
- FullPlanCoverage: `+0.203704`, from 37/54 to 48/54;
- answer and execution accuracy: `+0.000000`, both remaining 9/54.

## Gate decision and ablation interpretation

R0L, R2, R3, and R4 pass every pre-registered gate. R1 fails the minimum Leaf
and FullPlan deltas. R4 wins mechanically because FullPlan delta is the first
tie-break criterion and R4 has the largest value.

The controlled ladder supports the following implementation-level reading:

1. R0 to R0L: label-blind atomic-leaf decomposition is already useful without
   a neural model (`+0.046296` Leaf and `+0.055556` Full).
2. R0L to R1: adding dense RRF increases DOCS F2 but reduces Leaf, Full, TABLES
   F2, and TABLES MRR. Dense union alone is not a valid promotion candidate.
3. R1 to R2: the table reranker recovers and exceeds the lost coverage,
   reaching 42/54 full plans.
4. R2 to R3: the quota changes top-five membership for seven questions and
   order for eight, but those gains/losses cancel in aggregate and codegen is
   byte-identical. Coverage and F2 are identical; TABLES MRR changes only from
   0.778395 to 0.779938.
5. R3 to R4: row-aware reranking supplies the largest incremental gain:
   `+0.055555` Leaf, `+0.111111` Full, and `+0.039234` TABLES F2.

This is evidence for the complete R4 treatment, not proof that row reranking
alone will generalize independently of the preceding leaf/table stages.

## Paired and slice diagnostics

Against R0, R4 gains 12 full-plan questions and loses one; it gains Leaf recall
on 13 questions and loses it on four. It removes 127 R0 false-positive tables
and introduces 114 new false positives. The net direction is favorable, but
the large two-way churn shows that R4 is a material reranking treatment rather
than a small reorder.

The strongest family improvements are:

| Family | Leaf delta | Full delta | TABLES F2 delta |
| --- | ---: | ---: | ---: |
| scope_delta | +0.700 | +1.000 | +0.538 |
| cagr | +0.400 | +0.400 | +0.252 |
| debt_assets_ratio | +0.200 | +0.400 | +0.154 |
| ranking_argmin | +0.200 | +0.400 | +0.176 |

The unresolved or regressing slices are important:

- `count_positive`: Leaf falls from 0.667 to 0.533 and TABLES F2 from 0.588
  to 0.471; three of its five questions still lack full plans.
- `ranking_argmax`: Leaf falls from 0.867 to 0.800 and one previously complete
  plan is lost; two questions remain incomplete.
- `nested_margin_average`: coverage is unchanged, while DOCS F2 falls from
  0.933 to 0.873 and TABLES F2 changes from 0.692 to 0.687; one question remains
  incomplete.
- The six R4 incomplete questions are three `count_positive`, two
  `ranking_argmax`, and one `nested_margin_average` question.

The scope/period stress view improves materially: Leaf `0.800 -> 0.975`, Full
`0.700 -> 0.950`, DOCS F2 `0.836 -> 0.968`, and TABLES F2 `0.562 -> 0.695`.
Every LOYO block improves, although `from_2022` has the smallest Full gain
(`0.750 -> 0.833`). The five-question composition view does not improve
coverage and shows small TABLES plus larger DOCS regression. These views
overlap and are diagnostics, not independent replications.

An exploratory post-hoc paired bootstrap with seed 3303 and 100,000 resamples
gave positive per-question 95% percentile intervals for all four retrieval
deltas. A more conservative family-cluster bootstrap kept FullPlan positive
but allowed the Leaf and TABLES F2 intervals to cross slightly below zero.
This was not part of the registered gate and must not replace the one-shot
promotion test. It indicates that the aggregate gain is partly concentrated by
question family.

## Why answer accuracy did not move

The answer stack was intentionally frozen, and the result isolates its current
bottleneck:

- every stage has exactly 29 `status=ok` and 25 `status=failed` codegen rows;
- every stage has 19 `rule_composite`, 10 `rule`, and 25 `none` sources;
- typed-plan coverage remains zero;
- the same nine questions are correct at every stage: all five `note_lookup`
  and all four `simple_average` records;
- R4 changes nine numeric answers relative to R0, and six move closer to gold,
  but none enters tolerance.

FullPlanCoverage measures presence of every gold table, not correct row/cell
selection, operand binding, unit handling, or formula compilation. For example,
R4 recovers the complete table plan for several CAGR and scope-delta questions,
yet the deterministic selector still chooses wrong cells or scales. Retrieval
is therefore no longer the only dominant error source on this dev set.

This finding supports the planned separation: first validate R4 retrieval on
locked+hard; only after that should a new G3D typed planner/cell-grounder be
tested as a separate frozen hypothesis. R4 must not be patched using the six
residual dev questions.

## Next decision

1. Attach the newly uploaded promotion dataset to
   `kaggle/vifinqa-g3c-promotion-qwen-retrieval.ipynb` and run it once without
   editing the notebook, thresholds, prompts, models, or stage.
2. Download and strictly validate the output before opening the evaluator.
3. Run the promotion evaluator exactly once; preserve its open marker even if
   evaluation fails after opening.
4. If R4 survives locked+hard without material retrieval regression, retain it
   as the P-B retrieval candidate and start a separately registered G3D
   answer/planner experiment. If it fails, retain R0 and report the failed
   generalization test without dev-specific repair.

No official 1,012-question score or private submission should be used to
choose between R0 and R4 before this one-shot promotion decision is complete.
