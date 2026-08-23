# G3A–G3B Session Plan

## Mục tiêu

Hoàn thiện evaluation contract trước khi bắt đầu G3C retrieval.

**Boundary của session này:**
- Được phép sửa/bổ sung evaluation data, gold schema, split, evaluator, review/provenance.
- **Không** thay đổi production retrieval, planner, compiler, arbitration, model hoặc threshold của B0/B1.
- `data/g3a_v1/` phải giữ nguyên byte-for-byte như regression layer đã freeze.

---

## G3A — Evaluation Gate Hardening

### 1. Giữ G3A v1 bất biến
- Không regenerate hoặc sửa `data/g3a_v1/`.
- Giữ nguyên fingerprint hiện tại.
- Mọi mở rộng tạo artifact/version mới.

### 2. Mở rộng coverage trước G3C
Bổ sung các family còn thiếu:
- ranking / argmax / argmin;
- count + filter/comparison;
- CAGR;
- percentage-point change;
- nested/compositional arithmetic;
- note-table facts;
- ambiguous scope: separate vs consolidated;
- current/prior-period ambiguity;
- non-money outputs: count, year, ratio, percent.

### 3. Tách evaluator mode
- `dev`: chỉ expose `primary_tune`.
- `promotion`: dùng locked/hard sau khi candidate/config đã freeze.
- Không dùng `primary_locked` để tune lặp lại.

### 4. Gold assurance
- Hard/compositional/ambiguous examples phải được review lại độc lập.
- Review hash phải bind vào: question + evidence + AST/program + answer + relevant docs/tables.
- Random audit một phần primary set.

### 5. Giữ nguyên competition-shaped metrics
- DOCS: P/R/F2/MRR@5.
- TABLES: P/R/F2/MRR@5.
- Answer Accuracy.
- Execution Accuracy.

---

## G3B — Typed, Compositional & OOD Evaluation Corpus

### 1. Typed gold dùng cùng semantic contract với Selection v2
Không tạo IR thứ ba.

Mỗi record nên có:
- canonical typed AST;
- atomic facts / leaf specs;
- operand roles;
- metric family;
- operator family;
- output type;
- period semantics;
- unit semantics;
- scope;
- fact IDs / exact evidence;
- answer.

### 2. OOD views
Tạo manifest/split riêng cho:
- **LOTO**: leave-one-ticker-out / ticker-group holdout;
- **LOYO**: year/year-block holdout;
- **LORO**: report/source-context holdout;
- **LOMO**: metric-family holdout;
- **Composition**: primitive ops seen, program-tree shape unseen;
- scope/period stress view.

Mỗi view phải có explicit leakage assertions và frozen hash.

### 3. Grounding/reasoning diagnostics
Bổ sung:
- Leaf Recall@K;
- FullPlanCoverage;
- operator accuracy;
- operand-role accuracy;
- output-type accuracy;
- canonical AST/program match;
- program execution accuracy.

### 4. Hai evaluation modes
**Oracle evidence**
- đưa gold facts/evidence vào planner;
- đo reasoning độc lập với retrieval.

**End-to-end**
- dùng retrieval/grounding thực tế;
- đo toàn pipeline.

Mục tiêu là phân biệt:
- oracle tốt, E2E kém → retrieval/grounding bottleneck;
- evidence tốt, oracle kém → reasoning bottleneck.

### 5. Final freeze
Sau khi G3B hoàn tất, tạo một G3 evaluation manifest hash gồm:
- G3A v1 fingerprint;
- G3A extension fingerprint;
- G3B corpus/gold hashes;
- OOD view hashes;
- evaluator/config hashes;
- review ledger hashes.

**Chỉ sau freeze này mới bắt đầu G3C.**

---

## Acceptance Criteria

1. `data/g3a_v1/` không đổi.
2. Không dùng official/public question text để sinh template/train data; chỉ dùng cho exclusion/leakage checks.
3. G3B AST tương thích semantic surface của Selection v2.
4. Có ranking/count/CAGR/nested + scope/period stress.
5. Có LOTO/LOYO/LORO/LOMO/composition views với leakage guards.
6. Có dev-vs-promotion evaluator policy.
7. Có oracle-evidence và end-to-end evaluation.
8. Có Leaf Recall@K + FullPlanCoverage + typed reasoning metrics.
9. B0/B1 production code/config không bị thay đổi trong session này.
10. Evaluation contract được hash/freeze trước G3C.

---

## Sau session này

Review G3A/G3B artifacts và baseline diagnostics.

Nếu đạt acceptance criteria, triển khai **G3C Retrieval** theo ablation:

`current canonical/BM25 → + Qwen3-Embedding-4B → + Qwen3-Reranker-4B → + per-leaf quota → + row/cell reranking`

Giữ planner/compiler/arbitration frozen trong toàn bộ retrieval ablation.
---

## Agent execution outcome - 2026-08-23

Status: complete.

The plan was implemented with four review corrections: leave-one-* views are
overlapping diagnostics, typed gold reuses Selection-v2 rather than a new IR,
promotion requires a hash-bound candidate manifest, and oracle retrieval
metrics are explicitly non-interpretable.

All ten acceptance criteria passed. G3A v1 remained byte-for-byte unchanged;
109 G3B records and 72 required reviews were frozen; B0 dev and pre-frozen
promotion runs completed; 356 repository tests passed.

Final evaluation-freeze fingerprint:

`242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877`

Detailed results: `G3A_G3B_SESSION_RESULTS_2026-08-23.md`.

Executed commands and handoff: `G3B_RUNBOOK.md`.

Next authorized milestone: G3C retrieval ablation with
planner/compiler/arbitration frozen.
