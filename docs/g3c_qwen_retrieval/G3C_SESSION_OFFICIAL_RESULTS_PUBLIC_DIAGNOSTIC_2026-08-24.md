# G3C official results and public retrieval diagnostic - 2026-08-24

## Outcome

All three frozen official Qwen phases completed and passed independent local
validation. The merged output contains 1,012 aligned questions: exact frozen
R4 for 998 and the 14 preregistered R0 passthroughs. The engineering audit has
zero hard-constraint violations, duplicate candidates, empty candidate lists
or non-finite values.

A separate B1-fixed public retrieval diagnostic was then built. It retains the
already-scored B1 answer, evidence, data CSV and pandas query for every question
and changes only `relevant_docs` and `relevant_tables`. It is ready for one
aggregate Dashboard observation. It is not the canonical private P-B artifact
and must not select or retune R4 or G3D.

## Frozen identities

- candidate fingerprint:
  `1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a`;
- official protocol fingerprint:
  `958ba9d2dd4c979d54635f251415211b75316bc927b0e145a39e7c4270faf51f`;
- payload fingerprint:
  `4428637d718fddbd99db7d034337af66743a7dca23026b2da3a04af9dc55f227`;
- workload fingerprint:
  `7e23632782b72d0cb5bed2137d50f0e4dcbf077223e426e9fd44254fc8fb9d5b`;
- official R0 SHA-256:
  `76bedf827712a7f78a1db6b34a1da89f58b842621c01e42f89329592478f93d0`;
- official R4 SHA-256:
  `1281af9a737fd235e61b275c4ffebe34624d4790be6c043c81ca88d8552132d5`;
- result fingerprint:
  `1da1747e78031d23e41c41886f831b234eb7df927ab1a4507bfaa20da22f2afa`;
- artifact fingerprint:
  `e059954e6a7edf7bdd3fe04dce9a67709b458064bc0169dfe156512cbfce48bb`.

No official question labels or public score entered the R4 selection or
official runner.

## Phase validation and runtime

| Phase | Coverage | Wall time | Validation |
| --- | ---: | ---: | --- |
| Embedding | 89,862 vectors | 8,614.993 s | pass; 6-case canary on both T4s |
| Rerank A | shards 0+1, 498 questions | 17,945.451 s | pass; 7-case canary per GPU |
| Rerank B | shards 2+3, 514 questions | 17,649.451 s | pass; 7-case canary per GPU |
| Sequential total | 1,012 questions | 44,209.894 s / 12.281 h | pass |

The preregistered projection was 12.279 hours; measured time differed by only
0.092 minutes. Scaling Promotion directly would have projected 17.417 hours,
so exact batch/question parallelism saved 5.137 hours, or 29.5%, without
changing frozen model calls.

The child manifests list GPU index zero because each worker has its own
`CUDA_VISIBLE_DEVICES` mapping. The two embedding workers and paired reranker
shards ran on separate T4s. Peak memory was approximately 8 GiB per worker.

## Official engineering audit

- ordered and unique IDs: 1,012/1,012;
- exact question alignment: 1,012/1,012;
- supported exact R4: 998;
- declared R0 passthrough: 14;
- candidate depth: min 9, mean 19.971344, max 20;
- duplicate/empty/non-finite/hard-violation counts: all zero;
- public score read: false;
- gold read: false.

This proves execution integrity only. It does not provide official retrieval
recall, F2, answer accuracy or private-distribution evidence.

## R4 versus R0 structural effect

R4 is a material retrieval treatment rather than a small reorder:

| Diagnostic | Count |
| --- | ---: |
| top-1 table changed | 739/1,012 |
| top-5 set changed | 978/1,012 |
| zero top-5 overlap | 125/1,012 |
| at least one top-5 table outside the full R0 top-20 | 562/1,012 |
| R4 top-1 outside the full R0 top-20 | 161/1,012 |

Mean R0/R4 top-5 overlap is 2.21 tables. Of 5,060 R4 top-5 positions,
1,083 are new outside the R0 depth-20 pool. This is label-blind structural
evidence, not a correctness metric.

## Public diagnostic construction

Frozen B1 comparator:

- codegen SHA-256:
  `a8c2b93279daa7099ce0fdcead123cf9df134687fefeb48b484dd66267f9371c`;
- prior submission SHA-256:
  `c98f1859e41a924458abfc7f5b2f2673e028136e7a73b2cb04c6cb84467cb75c`.

Commands:

    python scripts/86_g3c_build_public_retrieval_diagnostic.py build
    python scripts/86_g3c_build_public_retrieval_diagnostic.py validate

Validated output:

    artifacts/g3c_v1/official_submission_b1fixed_r4_v1/submission.zip

- submission SHA-256:
  `a20257967fa0dbdf1659978ef901489eb0bd7ae6d1d03fbc1ccf55667164c319`;
- results SHA-256:
  `01d63f76e7f51b9b91d87c8dfb8347a4f899c160b95381ed3d7ec0a02603a6ef`;
- manifest fingerprint:
  `c557ad5cd9d42b85ec8a426e990e9806d8bdb7f9feb841f79a03d7eac75fadba`;
- audit fingerprint:
  `36dd9a35907d5e816812ac38885d467e1bde8c0f723c7fbe275b39304d3401c2`.

Submission integrity:

- 1,012 expressions are eval-compilable and replay exactly;
- 1,522 archive members and safe relative paths;
- all 1,521 data CSV members are byte-identical to B1;
- only archive member `results.json` differs;
- only row fields `relevant_docs` and `relevant_tables` differ;
- no answer, evidence, CSV or pandas-query mutation.

Visible retrieval deltas versus the scored B1 file:

| Change | Count |
| --- | ---: |
| relevant-table order | 998 |
| relevant-table set | 974 |
| relevant-document order | 479 |
| relevant-document set | 196 |

Mean relevant-table count changes from 5.340909 to 5.823123 because frozen B1
execution evidence remains declared when it falls outside the new R4 top five.
Mean relevant-document count changes from 2.436759 to 2.408103.

## Cross-platform CSV incident

The first two local builds replayed successfully but were rejected by the new
byte-delta guard. `pandas.to_csv` produced CRLF locally and Windows text-mode
writing translated it again in the first build. After disabling write-time
translation, CRLF still differed from the frozen Kaggle/Linux LF bytes.

The final fix explicitly uses `lineterminator="\n"` in `tidy_csv_text` and
writes the encoded UTF-8 bytes without platform translation. Regression tests
cover both behaviors. The rejected builds are retained under directories ending
in `failed_windows_newline` and `failed_windows_crlf`; neither is uploadable.

## Preregistered Dashboard interpretation

Baseline metrics supplied before this diagnostic was built:

| Metric | B1 |
| --- | ---: |
| TABLES F2 / Precision / Recall / MRR@5 | 0.4518 / 0.3037 / 0.6313 / 0.6152 |
| DOCS F2 / Precision / Recall / MRR@5 | 0.9125 / 0.9510 / 0.9109 / 0.9723 |
| Answer / Execution Accuracy | 0.1897 / 0.1897 |

Answer and Execution Accuracy must remain exactly 0.1897. A difference means
the comparison is invalid because the supposedly fixed layer or grader changed.
The primary descriptive readout is TABLES F2 plus DOCS F2 non-regression;
precision, recall and MRR explain the aggregate direction.

Read aggregate metrics once. Do not inspect question-level public residuals,
change R4, tune top-k/thresholds, select a private candidate, or redirect G3D
from this result. G3D remains the next scientific hypothesis with R4 fixed and
a new unopened source-derived holdout.

## Verification completed

- three phase validators: pass;
- official final engineering audit/freeze: pass;
- public diagnostic build/replay/archive audit: pass;
- independent public diagnostic validator: pass;
- targeted official/submission/diagnostic suite: 18 passed;
- full repository suite: 389 passed in 59.27 seconds;
- Markdown/diff whitespace check: pass.
