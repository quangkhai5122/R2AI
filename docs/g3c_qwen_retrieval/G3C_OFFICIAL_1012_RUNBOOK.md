# G3C frozen R4 official 1,012 runbook

Status date: 2026-08-24

Current status: P-B/R4 is closed and immutable. The official 1,012-question
protocol and payload were built and validated locally. All three real-Qwen
Kaggle phases and strict local finalization passed on 2026-08-24. Preserve the
completed outputs and do not rerun or overwrite them.

This run is a post-freeze engineering audit only. It may establish that the
frozen retrieval candidate runs without crashes and produces aligned, finite,
schema-valid output on all official questions. It cannot select a model,
threshold, fallback, or question-specific change, and it is not an answer or
private-distribution evaluation.

## 1. Frozen identity

The immutable identities are:

    selected candidate: g3c-qwen-retrieval-v1-r4
    candidate fingerprint:
    1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a

    P-B closeout fingerprint:
    1c76675392369ab1d3133e54aeed38813492426f89ea1f627e0c21e482f0759d

    official protocol fingerprint:
    958ba9d2dd4c979d54635f251415211b75316bc927b0e145a39e7c4270faf51f

    official behavior tree SHA-256:
    cb5634e212ee067ae4fec6851bc4669b12497b0402b62eac8179f085a10720e2

    payload fingerprint:
    4428637d718fddbd99db7d034337af66743a7dca23026b2da3a04af9dc55f227

    workload fingerprint:
    7e23632782b72d0cb5bed2137d50f0e4dcbf077223e426e9fd44254fc8fb9d5b

The P-B registry records that the promotion evaluator consumed its only run.
Never rebuild or reopen promotion evidence.

## 2. Exact-equivalence boundary

For every question on which frozen G3C's exact-report precondition holds, the
official runner preserves:

- the exact Qwen model and tokenizer revisions;
- FP16, SDPA, full 2,560-dimensional embeddings, and no quantization;
- embedding batch size 4 and the original sorted passage-then-query batch
  boundaries;
- reranker batch size 2, prompt/instruction text, and maximum lengths 768/384;
- each complete reranker call for one question on one GPU;
- dense, RRF, quota, row-fusion, candidate ordering, and top-5 projection
  semantics.

Parallelism changes only which identical T4 executes a complete frozen batch
or complete question. It never splits a batch or question. An exact numeric
canary reconstructed from the immutable Promotion caches runs independently on
both T4s before official work. Embedding arrays must be bitwise equal and
reranker score floats must be exactly equal to the frozen canary.

No dev or Promotion Qwen cache is seeded into the official run. This is
intentional. Of 1,132 overlapping dev/Promotion embedding keys, 789 stored
FP16 vectors differed because their original padding/batch contexts differed.
Reusing them would violate the exact-batch claim even though cosine similarity
was very high.

### Undefined frozen cases

Frozen G3C raises before inference when an atomic leaf has no exact report. The
official audit found 14 such questions:

    8, 62, 104, 336, 344, 411, 464,
    517, 525, 583, 689, 786, 809, 880

There is no frozen R4 ranking to preserve for those inputs. The preregistered,
label-blind totalization is therefore an exact R0 passthrough with explicit
`g3c.execution=r0_passthrough_unsupported` provenance. It performs no fuzzy or
nearby-report search and preserves the R0 candidate order byte-for-structure.
The remaining 998 questions use exact frozen R4.

This distinction must remain visible in every report. Do not claim that 1,012
questions received neural R4 reranking.

## 3. Measured workload and runtime decision

The frozen workload contains:

| Item | Count |
| --- | ---: |
| Official questions | 1,012 |
| Exact-R4 questions | 998 |
| R0 passthrough questions | 14 |
| Atomic leaves | 4,889 |
| Unique serialized queries used by R4 | 4,836 |
| Unique table passages | 85,026 |
| Embedding vectors | 89,862 |
| Frozen embedding batches | 22,466 |
| Table reranker pairs | 116,011 |
| Row reranker pairs, conservative upper bound | 116,064 |

The Promotion run took 56.80 minutes but used only one of the two visible T4s.
Scaling its measured phase timings to this workload gives:

| Phase | Conservative projection |
| --- | ---: |
| Embedding GPU work | 5.071 GPU-hours |
| Embedding on 2 T4s | 2.535 wall-hours |
| Reranking | 19.489 GPU-hours upper bound |
| One of four rerank shards | 4.872 wall-hours upper bound |
| All work in one 2-T4 session | 12.280 wall-hours upper bound |

The one-session estimate is too close to or beyond a normal Kaggle session
limit. The frozen execution is split into three bounded Kaggle versions:

1. two-T4 embedding;
2. reranker shards 0 and 1, one whole-question shard per T4;
3. reranker shards 2 and 3, one whole-question shard per T4.

Each embedding worker owns 11,233 complete batches (44,932 and 44,930
vectors). Reranker table-pair counts are 29,004, 29,004, 29,002, and 29,001;
row-pair upper bounds are 29,016 for every shard. The partitioner balances
estimated token work, not question count.

## 4. Local preflight

Run from the repository root in the project environment:

    python scripts/81_g3c_closeout_pb.py validate
    python scripts/82_g3c_build_exact_canary.py validate
    python scripts/83_g3c_freeze_official_protocol.py validate
    python scripts/84_g3c_build_official_payload.py validate
    python -m pytest tests/test_g3c.py tests/test_g3c_official.py -q -p no:cacheprovider --basetemp artifacts/pytest_g3c_official

All validators must print the fingerprints in sections 1 and 3. Stop on any
drift. Do not rebuild the protocol or edit the payload after a failure.

Active payload:

    artifacts/g3c_v1/official_payload

Kaggle dataset ID:

    lequangkhai5122005/vifinqa-g3c-r4-official-1012-v1

The payload contains only official `id`/`question`, frozen R0 retrieval, the
financial store, runtime code and hash-bound contracts. It contains no gold,
G3B corpus rows, manual review records, evaluator, or public scores.

## 5. Upload the immutable payload

Kaggle CLI on Windows can mishandle a multi-component `-p` path. Change into
the payload directory and use `-p .`:

    cd artifacts/g3c_v1/official_payload
    python -m kaggle datasets create -p . --dir-mode zip
    cd ../../..

If the slug already exists, use a dataset version instead:

    cd artifacts/g3c_v1/official_payload
    python -m kaggle datasets version -p . -m "Frozen G3C R4 official 1012 payload 4428637d" --dir-mode zip
    cd ../../..

Keep the dataset private. Kaggle consumes `dataset-metadata.json` as an upload
sidecar and may omit it from `/kaggle/input`; the validator explicitly permits
that one absence while keeping every core file mandatory.

## 6. Kaggle phase E: exact two-T4 embeddings

Import and run:

    kaggle/vifinqa-g3c-official-embedding.ipynb

Attach only the newest frozen official payload. Enable **2 x T4 GPU** and
Internet, restart the session, and Run all. Cell 2 must show two identical T4
descriptors and Transformers 4.53.3. Cell 4 must complete its exact-canary
assertion and report:

    vectors: 89862

Expected outputs:

    /kaggle/working/g3c_official_embedding/
    /kaggle/working/g3c_official_embedding_results.zip

Save a successful notebook version and make its output available as a private
input to both reranker notebooks. Do not unzip/recompress individual chunk
files on Kaggle; their manifests bind exact sizes and SHA-256 hashes.

After downloading and extracting the zip locally so the manifest is directly
inside the directory below, validate it (enter as one line in PowerShell):

    python scripts/85_g3c_finalize_official.py validate-embedding --payload artifacts/g3c_v1/official_payload --results artifacts/g3c_v1/official_embedding_results

## 7. Kaggle phase A: reranker shards 0 + 1

Import and run:

    kaggle/vifinqa-g3c-official-rerank-a.ipynb

Attach exactly two inputs: the frozen official payload and the successful
embedding notebook output. Enable 2 x T4 and Internet. The notebook validates
all embedding chunks before loading the reranker.

Expected outputs:

    /kaggle/working/g3c_official_rerank_a/
    /kaggle/working/g3c_official_rerank_a_results.zip

Extract locally and validate:

    python scripts/85_g3c_finalize_official.py validate-pair --payload artifacts/g3c_v1/official_payload --embedding-results artifacts/g3c_v1/official_embedding_results --results artifacts/g3c_v1/official_rerank_a_results --shards 0 1

## 8. Kaggle phase B: reranker shards 2 + 3

Import and run:

    kaggle/vifinqa-g3c-official-rerank-b.ipynb

Use the same frozen payload and embedding output. Do not use phase A as a
cache source: score-cache keys are question-local, and the shard assignments
are disjoint.

Expected outputs:

    /kaggle/working/g3c_official_rerank_b/
    /kaggle/working/g3c_official_rerank_b_results.zip

Extract locally and validate:

    python scripts/85_g3c_finalize_official.py validate-pair --payload artifacts/g3c_v1/official_payload --embedding-results artifacts/g3c_v1/official_embedding_results --results artifacts/g3c_v1/official_rerank_b_results --shards 2 3

## 9. Merge, audit and freeze locally

Only after all three strict validations pass:

    python scripts/85_g3c_finalize_official.py finalize --payload artifacts/g3c_v1/official_payload --embedding-results artifacts/g3c_v1/official_embedding_results --pair-a artifacts/g3c_v1/official_rerank_a_results --pair-b artifacts/g3c_v1/official_rerank_b_results --out-dir artifacts/g3c_v1/official_qwen_results

The finalizer refuses to overwrite a non-empty output directory. It requires:

- exactly 1,012 ordered, unique IDs and exact question-text alignment;
- exactly four non-overlapping, hash-bound shards;
- candidate depth no greater than 20 and no duplicate table key;
- zero hard ticker/report/year/scope violations;
- no NaN or infinity anywhere in retrieval records;
- exact R0 candidate order for the 14 declared passthrough questions;
- both numeric canaries on every GPU phase;
- no labels or public score dependency.

Expected final files are:

    artifacts/g3c_v1/official_qwen_results/r0_retrieval.jsonl
    artifacts/g3c_v1/official_qwen_results/r4_retrieval.jsonl
    artifacts/g3c_v1/official_qwen_results/g3c_official_engineering_audit.json
    artifacts/g3c_v1/official_qwen_results/g3c_official_result_manifest.json
    artifacts/g3c_v1/official_qwen_results/g3c_official_artifact_freeze.json

## 10. Resume and failure rules

- Within a live Kaggle session, rerunning the failed execution cell reuses only
  hash-validated embedding chunks or the question-local score checkpoint.
- Embeddings checkpoint every 256 complete frozen batches per worker.
- Reranker scores checkpoint every four complete questions.
- A completed phase is reusable only when its run signature matches the exact
  payload, workload, shard assignment and upstream embedding signature.
- Never seed official work from dev/Promotion vectors or scores.
- Never lower batch size, quantize, change attention, or split a frozen batch
  after OOM and call it the same run.
- Preserve all partial output and logs after a timeout or canary mismatch.
  Diagnose first; a changed behavior contract requires a new protocol version,
  while the failed version remains provenance.
- Do not inspect leaderboard scores to decide a code or threshold change.

## 11. Claim boundary after completion

A passing final artifact supports only this statement:

> The frozen P-B/R4 retrieval candidate completed a label-blind official
> 1,012-question engineering audit with 998 exact-R4 records, 14 declared R0
> passthrough records, aligned finite output, and zero hard-constraint
> violations.

It does not show answer accuracy, retrieval recall, or private performance on
the official questions. Those labels are not available to this pipeline.

## 12. Completed artifact and public diagnostic

The final official engineering artifact is:

    artifacts/g3c_v1/official_qwen_results/r4_retrieval.jsonl

SHA-256:

    1281af9a737fd235e61b275c4ffebe34624d4790be6c043c81ca88d8552132d5

The three phase wall times were 8,614.993 seconds for embeddings, 17,945.451
seconds for rerank A and 17,649.451 seconds for rerank B, or 12.281 hours in
sequence. All strict validators and numeric canaries passed.

For a one-time public aggregate retrieval measurement, the B1-fixed diagnostic
is separate from canonical private P-B:

    python scripts/86_g3c_build_public_retrieval_diagnostic.py validate

Upload only:

    artifacts/g3c_v1/official_submission_b1fixed_r4_v1/submission.zip

Its SHA-256 is
`a20257967fa0dbdf1659978ef901489eb0bd7ae6d1d03fbc1ccf55667164c319`.
The detailed audit, baseline metrics and no-retuning policy are in
`G3C_SESSION_OFFICIAL_RESULTS_PUBLIC_DIAGNOSTIC_2026-08-24.md`.
