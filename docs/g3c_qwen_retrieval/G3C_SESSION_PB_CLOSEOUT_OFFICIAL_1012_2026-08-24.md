# G3C P-B closeout and official 1,012 implementation session

Date: 2026-08-24

## Outcome

Steps 1 and 2 of the post-Promotion plan were implemented locally:

1. P-B/R4 was formally closed, registered and bound to its one-shot evidence.
2. A semantically exact two-T4 execution path for the official 1,012 questions
   was designed, frozen, packaged and locally validated.

The remote Qwen phases remain pending. Therefore no official R4 retrieval file
or official engineering-audit result exists yet.

## Step 1: P-B closeout

The closeout records the bounded scientific conclusion:

- R4 improved all registered retrieval metrics on dev and on the one permitted
  same-corpus/different-question Promotion evaluation;
- answer and execution accuracy did not improve;
- the Promotion evaluator is consumed and may not be reopened;
- official questions have no selection role;
- G3D typed planning plus row/cell grounding is the next scientific
  hypothesis, with R4 fixed underneath it.

Created evidence:

    experiments/g3c_qwen_retrieval_v1/pb_r4_closeout.json
    experiments/g3c_qwen_retrieval_v1/registry.json

Fingerprints:

    closeout: 1c76675392369ab1d3133e54aeed38813492426f89ea1f627e0c21e482f0759d
    registry: 4c1026ba69ffbd2f8fd6a73d807ac20221cb2e29e562bb9c25eaea42345321de

## Step 2 review: why the Promotion runner could not simply be scaled

The 55-question Promotion run took 3,407.707 seconds on a 2 x T4 instance, but
the frozen implementation placed both models on generic `cuda`, so only GPU 0
performed inference. Measured phase times were 1,447.024 seconds for embedding
and 1,954.467 seconds for reranking.

A direct one-GPU extrapolation to 1,012 questions is not viable. A naive split
of individual samples is also invalid: left padding changes tensor shapes and
can change stored FP16 embeddings, which can change near-tied dense ranks.

This was measured rather than assumed. The dev and Promotion caches shared
1,132 embedding keys; 789 overlapping FP16 vectors were not bitwise equal.
Maximum absolute difference was 0.00061035 and minimum cosine similarity was
approximately 0.999969. They are numerically close but fail the requested exact
equivalence contract. No old vector or score cache is used for official work.

## Exact parallelization adopted

The implementation parallelizes only at boundaries that do not alter a frozen
model call:

- reconstruct the original globally sorted passage/query embedding schedule;
- assign complete batches of four to a T4, never individual samples;
- assign complete questions to a reranker shard, never individual pairs from a
  question call;
- retain the original per-call pair order and internal batch size two;
- recombine outputs by original official question order;
- run exact Promotion-derived numeric canaries on both devices.

This yields three Kaggle phases: embedding, reranker shards 0+1, and reranker
shards 2+3. The conservative projections are 2.535 hours for embedding and
4.872 hours for each reranker phase. The projected 12.280-hour monolithic run
was rejected as unsafe.

## Official input audit

The frozen preparation logic was run label-blind over all 1,012 official
questions. Results:

| Item | Count |
| --- | ---: |
| Questions | 1,012 |
| Exact-R4 supported | 998 |
| Missing-exact-report precondition | 14 |
| Atomic leaves | 4,889 |
| R4 serialized queries | 4,836 |
| Table passages | 85,026 |
| Total embeddings | 89,862 |
| Embedding batches | 22,466 |
| Table reranker pairs | 116,011 |
| Row reranker upper bound | 116,064 |

Unsupported IDs are:

    8, 62, 104, 336, 344, 411, 464,
    517, 525, 583, 689, 786, 809, 880

The original runner has no R4 output for these inputs because it raises before
inference. A conservative, preregistered totalization keeps exact R0 order and
adds explicit fallback provenance. It does not search for a nearby report or
modify any question-specific rule. The exact-equivalence claim applies to the
998 supported questions; the other 14 are explicitly R0 passthrough.

## Numeric equivalence guard

A canary was reconstructed from immutable Promotion Qwen caches:

    artifacts/g3c_v1/official_preflight/exact_numeric_canary.json
    artifacts/g3c_v1/official_preflight/exact_numeric_canary_vectors.npz

It contains six complete embedding batches and seven complete reranker batches
covering table and row scoring. Cached semantic replay preserved candidate
keys/order, non-dense diagnostics, and submission projection exactly. A tiny
local `dense_score_max` diagnostic difference caused by the local NumPy BLAS
was bounded at 1e-6 and does not affect rank/output. On Kaggle, raw Qwen canary
vectors and scores must match the original stored GPU values exactly.

Canary fingerprint:

    a9f4acdf8a3437e5115be39e1ef79b86c0f8db67f553d77856bdb6f64cb5c130

## Frozen implementation

Core additions:

    configs/g3c_official_1012_v1.json
    vifinqa/g3c_official/
    kaggle/kaggle_g3c_official.py
    kaggle/vifinqa-g3c-official-embedding.ipynb
    kaggle/vifinqa-g3c-official-rerank-a.ipynb
    kaggle/vifinqa-g3c-official-rerank-b.ipynb
    scripts/81_g3c_closeout_pb.py
    scripts/82_g3c_build_exact_canary.py
    scripts/83_g3c_freeze_official_protocol.py
    scripts/84_g3c_build_official_payload.py
    scripts/85_g3c_finalize_official.py
    tests/test_g3c_official.py

Frozen protocol:

    experiments/g3c_qwen_retrieval_v1/official_protocol_freeze.json

    protocol fingerprint:
    958ba9d2dd4c979d54635f251415211b75316bc927b0e145a39e7c4270faf51f

    behavior tree SHA-256:
    cb5634e212ee067ae4fec6851bc4669b12497b0402b62eac8179f085a10720e2

Built payload:

    artifacts/g3c_v1/official_payload

    payload fingerprint:
    4428637d718fddbd99db7d034337af66743a7dca23026b2da3a04af9dc55f227

    workload fingerprint:
    7e23632782b72d0cb5bed2137d50f0e4dcbf077223e426e9fd44254fc8fb9d5b

## Verification completed locally

- P-B closeout and registry hash validation: passed.
- Promotion-derived numeric canary reconstruction and cached replay: passed.
- Official workload build and invariants: passed.
- Official protocol behavior-tree validation: passed.
- Payload exact file-set, sidecar, label-boundary and SHA-256 validation:
  passed.
- Full 1,012-question fake-backend orchestration: passed with 89,862 vectors,
  shard sizes 249/249/250/264, 998 exact-R4 records, 14 declared R0
  passthroughs, zero hard violations and zero non-finite values. Local phase
  times were 367.726 seconds for embedding, 217.673 seconds for shards 0+1,
  and 222.630 seconds for shards 2+3. These timings are not Qwen projections.
- Targeted G3C plus official tests: 28 passed.
- Full repository suite: 384 passed.
- Notebook JSON and every code cell compile: passed before documentation
  closeout.

The full fake-backend output is under
`artifacts/g3c_v1/official_fake_e2e`. It is engineering-only and cannot support
a model, runtime, or scientific claim. The finalizer deliberately requires
real-Qwen manifests for the official artifact.

## Pending remote evidence

The following are not yet completed:

- exact two-T4 official embeddings;
- reranker shards 0+1;
- reranker shards 2+3;
- strict local merge and the final 1,012-record engineering audit;
- immutable official R4 artifact freeze.

Use `G3C_OFFICIAL_1012_RUNBOOK.md`. Do not use the historical dev or Promotion
notebooks for this workload.
