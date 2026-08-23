# G3C implementation session - 2026-08-23

## Outcome

G3C is implemented and locally verified as a guarded, label-blind
local-to-Kaggle-to-local retrieval experiment. The dev protocol and stripped
Kaggle payload are frozen and ready to run.

No real Qwen GPU run occurred in this session. Therefore:

- no G3C retrieval or submission metric has been measured;
- no R0L/R1/R2/R3/R4 stage has passed the dev gate;
- no candidate has been selected;
- no promotion payload has been created;
- no generalization or leaderboard claim is supported.

The next action is the dev Kaggle notebook, not a promotion run.

Operational update on 2026-08-24: the original payload v1 was rejected in
Kaggle cell 2 because dataset-metadata.json is an upload-only file and is not
mounted. No Qwen inference ran. Payload v1 remains immutable provenance; the
active runtime inputs are dev_protocol_freeze_v2.json and dev_payload_v2. See
G3C_SESSION_KAGGLE_PAYLOAD_V2_FIX_2026-08-24.md.

## Request implemented

The requested plan was reviewed before implementation. The implementation:

- preserves the clean R0 control and frozen G3 evaluator;
- uses Qwen3 Embedding 4B and Qwen3 Reranker 4B only on Kaggle GPU;
- computes all dev ablations and both retrieval/submission metrics;
- keeps gold and evaluators local;
- has a standalone G3C runbook;
- records source, model, config, payload, cache, result and candidate hashes;
- prevents promotion before a passing dev candidate freeze.

## Review of the proposed plan

The high-level plan was scientifically appropriate, but it needed the following
controlled amendments.

### 1. Add R0L before the neural stages

The proposed R0 to R1 jump mixed two effects: better per-leaf query formation
and Qwen dense retrieval. R0L adds the same new label-blind leaf decomposition
and hard guards with lexical retrieval only. This lets dev evidence distinguish
a neural gain from a deterministic query-formation gain.

### 2. Do not reuse production router leaves as if they were complete

The existing route is useful for ticker, year, scope and metric hints, but it
does not expose all evidence leaves required by CAGR, count, ratios, nested
margin averages, scope deltas and prior-period lookups.

A separate G3C decomposer was added. Its API accepts only:

- question text;
- the existing clean route;
- the canonical metric registry;
- store report metadata.

It cannot accept family labels, G3B gold, review records or question-ID lists.
Production router semantics were not mutated.

### 3. Correct multi-year report binding

The first implementation audit exposed a correctness issue that a simple
"report exists" invariant did not catch. A generic match of "report ... year
X" could bind every CAGR/average endpoint to the first report year.

The corrected rule is:

- ordinary multi-year leaves bind each period to its own exact-year report;
- prior-period questions bind the opening period to the explicitly later
  container report.

Regression assertions now verify report year and report ID for CAGR, nested
multi-year margin and prior-period cases.

### 4. Use the official reranker formulation

Qwen3-Reranker-4B is loaded with AutoModelForCausalLM. Relevance is the softmax
probability of the final-token yes logit against no. It is not implemented as
AutoModelForSequenceClassification.

The reranker uses the official prompt shape, an immutable model/tokenizer
revision, FP16, SDPA and use_cache=false.

### 5. Make R4 and the quota affect actual ranking

Two implementation hazards were corrected:

- the row lexical prefilter now selects 24 rows, neural-scores all 24, then
  keeps the best 12 row results; the previous draft effectively scored only 12
  and left the 24-row cap dead;
- quota-mandatory tables are ordered by global relevance, not by sorted
  leaf_id, so coverage does not create an arbitrary top-rank order.

R4 reorders tables and is not an unused metadata side channel.

### 6. Freeze numeric gates before Qwen evidence

The qualitative "material positive" gate was made executable:

- Leaf Recall@5 delta at least +0.02;
- FullPlanCoverage delta at least +0.03;
- DOCS F2 regression no worse than -0.01;
- TABLES F2 regression no worse than -0.01;
- zero hard-constraint violations and passing integrity.

The tie break is also deterministic and uses cumulative runtime for the
individual stage.

### 7. Separate dev and promotion payloads

The dev payload contains all six stages and primary_tune questions only. A
promotion payload cannot be built without a passing candidate freeze and
contains R0 plus exactly one selected stage.

The promotion evaluator writes a sentinel before opening locked results. It
cannot be reopened by an ordinary rerun.

### 8. Strengthen provenance and resume safety

The protocol freeze hashes every behavior-bearing G3C module, relevant clean
B0 dependencies, evaluation entrypoints, requirements and both notebooks.

Payload, GPU result, dev selection and candidate freeze all carry the same
protocol fingerprint. Cache keys include config and protocol fingerprints.
A completed output directory is reusable only with the exact same run
signature and exact file set.

## Implemented code and entrypoints

Configuration and protocol:

- configs/g3c_qwen_retrieval_v1.json
- vifinqa/g3c/common.py
- vifinqa/g3c/protocol.py
- scripts/80_g3c_freeze_protocol.py

Label-blind retrieval:

- vifinqa/g3c/leaves.py
- vifinqa/g3c/serialize.py
- vifinqa/g3c/retrieval.py
- vifinqa/g3c/modeling.py
- vifinqa/g3c/cache.py
- vifinqa/g3c/pipeline.py

Payload, import, metrics and freeze:

- vifinqa/g3c/payload.py
- vifinqa/g3c/validate.py
- vifinqa/g3c/paired.py
- vifinqa/g3c/freeze.py
- scripts/75_g3c_audit_leaves.py
- scripts/76_g3c_build_payload.py
- scripts/77_g3c_validate_gpu_results.py
- scripts/78_g3c_evaluate_stages.py
- scripts/79_g3c_select_freeze.py

Kaggle:

- kaggle/kaggle_g3c_qwen_retrieval.py
- kaggle/requirements-g3c.txt
- kaggle/vifinqa-g3c-dev-qwen-retrieval.ipynb
- kaggle/vifinqa-g3c-promotion-qwen-retrieval.ipynb

Tests:

- tests/test_g3c.py

## Original v1 frozen identifiers, superseded operationally

G3 evaluation fingerprint:

    242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877

G3C config SHA-256:

    a691497b9581136de2dc20adedca06868a0e791aee5c8fb9b5c3f657f5b9ded1

G3C behavior tree:

    578a55861e55faae8cf5b84404c6d805f351b7abef78d7efe7ce79f27e0fa2dd

G3C protocol fingerprint:

    116f26a5ed69c166087a68668e7b9c71bbe59a15749d440e78d1029db48c2b7e

Dev payload fingerprint:

    fb0f72ffe441e025d9d815b6f81264c15687448bf4075783068ccb311f0bbd13

Fake smoke run signature:

    83957ce127479da51c1359ab64862557731b673a4a7c2c6dbc9747c0426a28a7

Pinned model revisions:

- embedding:
  5cf2132abc99cad020ac570b19d031efec650f2b;
- reranker:
  22e683669bc0f0bd69640a1354a6d0aebcfeede5.

## Measured local evidence

### Frozen evaluation inputs

- G3B validate: 109 questions, 0 pending reviews, valid.
- G3 evaluation freeze validate: valid at the expected fingerprint.

### Leaf audits

Dev primary_tune:

- 54 questions;
- leaf-count distribution: 1 leaf = 10, 2 = 19, 3 = 15, 4 = 10;
- missing exact-report leaves: 0;
- invariant errors: 0.

Promotion structural-only audit:

- 55 questions;
- leaf-count distribution: 1 leaf = 10, 2 = 20, 3 = 15, 4 = 10;
- missing exact-report leaves: 0;
- invariant errors: 0.

The promotion audit used no gold or G3C performance outcomes and did not enter
candidate selection. It is structural engineering evidence only.

### Original payload v1, now superseded operationally

- mode: dev;
- 54 stripped question rows;
- exact fields: id and question;
- 248 files;
- 95,829,516 bytes, or 91.39 MiB;
- gold included: false;
- G3B corpus included: false;
- evaluator reports included: false;
- official public questions included: false.

### Packaged smoke

The packaged fake backend ran three questions through R0, R0L, R1, R2, R3 and
R4. All six stage artifacts validated with:

- three records per stage;
- zero duplicate candidates;
- zero hard-constraint violations;
- exact result/cache hashes;
- scientific_evidence_valid=false.

The same signature resumed idempotently. Reusing the completed directory with
limit 2 instead of 3 was rejected before overwrite.

B0 handoff on fake R4 produced three codegen rows:

- rule_composite: 2;
- none: 1.

The offline submission builder produced three entries and three CSV files,
passed expression compilation, and displayed its DO NOT UPLOAD guard.

These are compatibility checks, not Qwen or accuracy evidence.

### Tests

- targeted G3C tests: 19 passed;
- full repository suite: 375 passed in 13.91 seconds;
- Python compileall: passed;
- both notebook JSON documents parsed;
- every notebook code cell compiled.

## Baselines retained for later comparison

Frozen B0 dev:

- DOCS F2 0.903025;
- TABLES F2 0.612058;
- Leaf Recall@5 0.808642;
- FullPlanCoverage 0.685185;
- answer/execution accuracy 0.166667.

Frozen B0 promotion:

- DOCS F2 0.898216;
- TABLES F2 0.613054;
- Leaf Recall@5 0.786364;
- FullPlanCoverage 0.654545;
- answer/execution accuracy 0.181818.

The promotion baseline is context, not a tuning target.

## Interpretation

The local evidence establishes that the experiment is executable, label
boundaries are enforced, stages are separated, and local/Kaggle artifacts can
be compared reproducibly.

It does not establish that Qwen retrieval improves recall, precision, answer
accuracy, execution accuracy or OOD generalization. Fake scoring and invariant
audits cannot answer those questions.

## Next action

Follow G3C_RUNBOOK.md:

1. upload a new dataset version from artifacts/g3c_v1/dev_payload_v2;
2. import kaggle/vifinqa-g3c-dev-qwen-retrieval.ipynb;
3. enable a 16 GB GPU and Internet;
4. Run all;
5. download g3c_dev_results.zip;
6. run strict local import, all-stage evaluation and the frozen dev gate.

Do not create or run the promotion payload until step 6 produces a passing,
hash-bound candidate freeze.
