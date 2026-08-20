# 05 — Kế hoạch hành động và tái lập

## Mục tiêu

Tạo một production/research line duy nhất từ `main`, port phần canonical tốt nhất của `improve_baseline_kien`, thay phép chọn bằng public score bằng grouped-OOD protocol, rồi đóng băng năm private candidate có thể replay.

## Trình tự ưu tiên

### G0 — Khóa provenance hiện tại

Hoàn thành trước khi sửa thuật toán:

1. lưu export/screenshot Dashboard cho mọi checkpoint quan trọng;
2. xác nhận public/private denominator và quy tắc năm lượt;
3. lấy lại hoặc regenerate artifact `.2806/.2866` có exact hash;
4. lưu model revision, Kaggle notebook output, wheel versions và hardware log;
5. đánh dấu mọi leaderboard claim là `reported`, `artifact-verified` hoặc `dashboard-verified`;
6. dừng tạo exact public-ID overlay mới.

Deliverable: một immutable `experiment_registry.jsonl` và một reproduction bundle cho best checkpoint. Nếu artifact hash đã mất vĩnh viễn, ghi rõ `unreproducible historical checkpoint`; không thay bằng file gần giống.

### G1 — Engineering baseline sạch

Thực hiện trong commit riêng, không trộn algorithm change:

- thêm CI chạy 301+ tests, compile và format/diff checks;
- pin exact dependency bằng lock file/container image;
- lưu Python/Pandas/NumPy/PyArrow/Transformers/Torch versions;
- xóa 20 tracked `.orig` sau khi xác nhận nội dung đã có trong Git history;
- phân loại script `production`, `experiment`, `retired`;
- thêm license/NOTICE và external-data attribution;
- giữ test basetemp ngoài `.pytest_cache` lỗi ACL hoặc sửa ACL bằng thao tác quản trị có kiểm soát.

Gate: clean clone phải chạy full tests và build một rule-only smoke mà không cần file ngoài manifest.

### G2 — Hợp nhất canonical schema

Port có chọn lọc từ nhánh cải tiến:

1. `f08b927` — canonical metric dictionary v2;
2. phần retrieval/component expansion từ `3dfdf38`/`f597979`;
3. operator logic có kiểm chứng từ `783a773`;
4. audit policy từ `ef77d55`, nhưng bỏ exact-ID selection.

Không cherry-pick nguyên commit nếu nó kéo theo file cũ/xóa P2.2. Tạo module đích rõ ràng, ví dụ:

```text
vifinqa/finance/
  metrics.py          canonical metrics and qualifiers
  operators.py        typed signatures and unit algebra
  matching.py         metric/qualifier evidence
  schemas.py          immutable dataclasses / serialization
```

Sau đó sửa router, shortlist, deterministic solver, P2.4 authoring và Selection v2 cùng đọc registry này. Không giữ alias/formula copy riêng.

Gate:

- serialization deterministic và fingerprinted;
- registry test cho alias collision, qualifier conflict, sector/report type;
- current 301 tests không regression;
- synthetic property tests cho mọi operator;
- không hardcode question ID.

### G3 — Xây public-free development corpus

Tạo generator độc lập từ report cells/formulas. Dataset record tối thiểu:

```text
question
canonical_program
answer
entity/report/period/metric roles
evidence cells
unit provenance
generator/template/paraphraser revisions
source license
split groups
```

Tách group theo ticker/year/metric family/composition. Giữ một locked bundle không mở cho đến khi retrieval/reasoning thresholds freeze.

Gate data quality:

- 100% program replay;
- no evidence overlap giữa grouped train/locked theo view đang đánh giá;
- no official question ID/text seed;
- duplicate/paraphrase leakage scan;
- distribution report theo operation, depth, output type, unit, sector và OCR difficulty;
- manual audit phân tầng trước khi dùng metric.

### G4 — Per-leaf hybrid retrieval

Implementation order:

1. question graph → leaf slot expansion;
2. lexical/canonical control;
3. BGE-M3 candidate union;
4. hard-negative dataset;
5. learned reranker;
6. per-leaf quota and coverage certificate;
7. trace feature contributions và rejection reasons.

Hard negatives cần bao gồm:

- same metric, wrong year;
- same year, wrong entity;
- consolidated vs separate conflict;
- current vs prior/beginning balance;
- right row, metadata/note column;
- similar metric name, wrong qualifier;
- component vs derived metric;
- same numeric value by coincidence.

Gate: treatment phải tăng exact-cell/full-plan recall trên OOD locked folds; không promote bằng table retrieval score trên public leaderboard.

### G5 — Unified typed reasoning

Giữ Selection v2 compiler làm execution boundary. Chuyển `formula_solver` logic thành operator implementations trong registry, rồi cho hai planner dùng chung:

- deterministic planner;
- LLM typed-AST planner.

Thêm:

- multiple plan hypotheses từ router;
- all-valid AST canonicalization;
- program equivalence tests;
- counterfactual replay trên tables có perturbation;
- plan-level uncertainty và abstention;
- property-based tests cho unit/period/order.

Gate: không raw pandas, mọi accepted output có exact leaf coverage, compile và replay. Báo `planner wrong`, `missing leaf`, `ambiguous evidence`, `type/unit failure` riêng.

### G6 — Fine-tuning và calibration

Chỉ bắt đầu sau khi G3–G5 ổn định; nếu schema đổi trong lúc train, model supervision sẽ drift.

- reranker contrastive training bằng hard negatives;
- 7B QLoRA AST planner bằng external/synthetic programs;
- nested cross-validation cho arbitration gate;
- estimate pairwise error correlation giữa P-A…P-D;
- promote P-D chỉ nếu conditional correctness trên disagreements có ích.

Model registry bắt buộc ghi exact total/non-embedding parameters, release timestamp và license. Xin BTC xác nhận Qwen 14B; không chờ đến ngày nộp.

### G7 — Build và khóa private portfolio

Cho P-A…P-E:

1. chạy full inference từ cùng frozen corpus/store;
2. validate 1.012/private universe, unique IDs và finite answers;
3. compile/replay toàn bộ query;
4. validate evidence/table/doc references;
5. build ZIP và verify exact layout;
6. tạo candidate manifest;
7. tạo portfolio manifest chứa năm candidate hashes;
8. commit/tag/timestamp trước private score đầu tiên;
9. upload đúng hash đã khóa.

Không rebuild artifact giữa các lượt vì khác compression/order cũng làm hash khác và làm provenance mơ hồ.

## Acceptance gates

| Gate | Điều kiện bắt buộc | Nếu fail |
|---|---|---|
| Source | clean commit, tests/compile pass | không build artifact |
| Data | fingerprint, replay, split leakage clean | rebuild dataset version mới |
| Retrieval | OOD leaf/full-plan gain, worst-fold guard | giữ lexical control |
| Reasoning | typed compile + exact grounding + no rule regression | fail closed/fallback |
| Calibration | out-of-fold, stable by stratum | dùng conservative deterministic policy |
| Eligibility | open model, date/params/license verified | thay bằng eligible 7B |
| Artifact | complete manifest, all queries replay | không nộp |
| Portfolio | five hashes locked before feedback | dừng, không gọi là pre-registered |

## Reproduction bundle đề xuất

```text
releases/private_2026/
  SOURCE_COMMIT
  environment.lock
  data_manifest.json
  model_registry.json
  split_manifest.json
  registry/
    metrics.json
    operators.json
  candidates/
    P-A/
      config.json
      run_manifest.json
      results.json
      submission.zip
      audit.json
    P-B/
    P-C/
    P-D/
    P-E/
  portfolio.lock.json
  reports/
    ood_metrics.json
    disagreement_matrix.csv
    integrity_report.json
```

Artifact lớn có thể ở GitHub Release/object store thay vì Git, nhưng repo phải chứa content-addressed manifest và cách tải/verify. Không chỉ ghi hash của một file đã mất.

## Experiment registry

Mỗi row nên có:

- experiment ID, parent/control;
- source commit/dirty status;
- hypothesis và single treatment;
- data/split/model/config hashes;
- local/remote runtime;
- test/smoke/full status;
- metrics theo evidence tier;
- artifact hashes;
- decision `promote/reject/diagnostic` và lý do;
- whether public-derived information was used;
- timestamp before/after any leaderboard observation.

Điều này ngăn việc RUNBOOK trở thành nguồn sự thật duy nhất và cho phép so các session không bị lẫn config.

## Kế hoạch test bổ sung

### Registry

- alias collision và forbidden qualifier;
- metric components không tạo cycle;
- statement/sector constraints;
- deterministic serialization/hash.

### Retrieval

- per-leaf quota;
- hard negative same-row/wrong-column;
- wrong period/entity/report scope;
- dense model unavailable fallback;
- no candidate score leakage từ gold.

### Reasoning

- operator arity/type/unit properties;
- argument order for subtract/divide/growth;
- nested `rank(growth(ratio))`;
- multiple equivalent ASTs;
- counterfactual tables;
- no raw ref outside facts, no public literal/ID switches.

### Private mode

- reject any ID allowlist/`expect-selected-ids`;
- reject official-derived train/eval manifest;
- reject raw-code mode;
- reject mutable model revision;
- enforce five candidate hashes against portfolio lock.

## Risk register

| Rủi ro | Khả năng / tác động | Mitigation |
|---|---|---|
| Canonical registry overfits known terms | cao / cao | LOMO, sector/report qualifier tests, alias provenance |
| Synthetic dev quá dễ | cao / cao | compositional/OCR hard sets, external programs, manual audit |
| Dense retrieval làm mất exact labels | vừa / cao | union với BM25, not replacement |
| Fine-tune học template generator | cao / cao | generator-family holdout, paraphrase diversity |
| 14B không hợp rule | vừa / rất cao | BTC confirmation + 7B fallback |
| Best checkpoint artifact mất | hiện hữu / cao | regenerate or label unreproducible, release manifests |
| Branch merge làm mất P2.2/P2.4 | cao / cao | selective port, no wholesale merge |
| Private sequential overfit | cao / rất cao | five manifests locked before first score |
| Calibration overfit | vừa / cao | nested CV, worst-fold gate |
| Raw executor capability | vừa / cao | typed compiler only in production |

## Việc nên làm ngay tiếp theo

1. Xác nhận rule model parameter và denominator với BTC.
2. Phục hồi/replay checkpoint `.2866` hoặc hạ mức bằng chứng của nó.
3. Tạo nhánh tích hợp từ `main`; port metric registry v2 với tests, không port exact-ID blend.
4. Định nghĩa unified `Metric`/`Operator` schema trước khi chỉnh retriever.
5. Sinh public-free grouped dev v1 và khóa split.
6. Chạy lexical-vs-canonical-vs-hybrid ablation trên dev đó.
7. Chỉ sau evidence retrieval mới tích hợp typed planner/fine-tune.

Đây là thứ tự giảm rủi ro cao nhất: giải quyết provenance và measurement trước, rồi mới tăng model complexity.
