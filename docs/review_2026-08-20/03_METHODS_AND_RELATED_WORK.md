# 03 — Phương pháp hiện tại và đối chiếu các bài toán tương tự

## Bài toán nên được phân rã như thế nào

Một dự đoán ViFinQA đúng cần đồng thời thỏa:

1. nhận đúng entity/report scope;
2. nhận đúng period;
3. tìm đủ mọi bảng và atomic fact;
4. chọn đúng row/column/unit;
5. dựng đúng operator tree và thứ tự toán hạng;
6. compile/execution thành công;
7. khai báo đúng evidence/relevant tables/docs.

Vì vậy Answer Accuracy có thể hình dung như tích của nhiều xác suất có điều kiện. Tối ưu một score retrieval chung hoặc tăng số LLM samples không sửa được một leaf bị thiếu. Kiến trúc phải quan sát và đo từng lớp lỗi.

## Phương pháp hiện có trong repo

### Structured retrieval

Pipeline hiện khóa report bằng entity/year, BM25 table retrieval, rồi fact-aware row/cell shortlist. Đây là một thiết kế hợp lý vì nhãn tài chính có cả lexical anchor mạnh (`lợi nhuận sau thuế`, `MS 60`) lẫn paraphrase tiếng Việt.

Điểm còn thiếu là ranking theo từng leaf. Với câu `rank(growth(ratio(A,B)))`, hệ thống cần ít nhất bốn leaf cho mỗi entity-period combination. Một top-k flat hoặc quota theo report có thể có recall cao tổng thể nhưng vẫn thiếu đúng denominator của một entity.

### Neuro-symbolic reasoning

Repo đã tiến từ raw Python sang Selection v1 và typed Selection v2. Hướng hiện tại đúng: LLM làm semantic parsing/selection; deterministic code giữ type, unit, evidence và execution. P2.4 chứng minh raw executability không đủ: 51/100 câu chạy được nhưng sai.

### Canonical financial ontology

Nhánh cải tiến tạo metric registry v2 gồm aliases, codes, qualifiers, components và statement hints. Đây là abstraction cần thiết cho cả retrieval và reasoning. Vấn đề hiện tại là ontology tồn tại song song với `atomic_slots`, `formulas`, router patterns và P2.4 authoring.

### Consensus/repair

`semantic_repair`, `single_cell_consensus` và lookup rescue yêu cầu period/value-column/unit support rồi fail closed. Chúng hữu ích như verifier và tạo silver data. Khi khóa bằng exact public ID, chúng không còn là transferable policy. Private version phải áp cùng rule cho mọi câu và được calibrate trước trên OOD folds.

## Nguồn sơ cấp và bài học có thể chuyển giao

| Công trình / cuộc thi | Ý tưởng chính | Bài học cho R2AI |
|---|---|---|
| [FinQA, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.300/) | expert QA + gold reasoning programs trên financial reports | đo program/evidence, không chỉ answer; dùng DSL có thể replay |
| [FinQA official challenge repo](https://github.com/czyssrs/FinQA) | final dựa trên private test, không có intermediate result/gold reference | development phải độc lập với public feedback; end-to-end retriever + generator |
| [TAT-QA / TAGOP, ACL 2021](https://aclanthology.org/2021.acl-long.254/) | tag evidence spans/cells rồi áp symbolic aggregation operators | tách evidence extraction khỏi operator prediction; scale là output riêng |
| [FinMath, LREC 2022](https://aclanthology.org/2022.lrec-1.661/) | evidence first, top-down tree expression cho multi-step reasoning | nested tree/typed IR đáng ưu tiên hơn flat operation |
| [FinQA Challenge system, 2022](https://arxiv.org/abs/2206.08506) | row + cell retrievers, nhiều generators và ensemble | retrieval granularity và model diversity cho error profile khác nhau |
| [APOLLO, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.122/) | number-aware negative sampling; target-program augmentation; consistency objective | hard negatives phải phân biệt số/period tương tự; nhiều program tương đương không nên bị phạt |
| [TabDSR, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.169/) | decompose, sanitize, program-of-thought reason | decomposition và table sanitation là stage riêng; tránh đưa bảng nhiễu thẳng vào reasoner |
| [BGE-M3 official model card](https://huggingface.co/BAAI/bge-m3) | multilingual dense + sparse + multi-vector; khuyến nghị hybrid retrieval + reranking | phù hợp tiếng Việt, nhưng phải đo với hard-negative/OOD set của ViFinQA |
| [FinAgent-RAG preprint, 2026](https://arxiv.org/abs/2605.05409) | contrastive financial retriever, Program-of-Thought, adaptive router | hard-negative retriever và complexity routing đáng thử; raw Python/self-verification claims cần tái lập độc lập |

FinQA challenge system báo cell retriever và row retriever có thế mạnh khác nhau; ensemble row/cell tăng execution 2,44 điểm so với single model. Bài học không phải “ensemble luôn tốt”, mà là portfolio phải có **nguồn lỗi khác nhau** và gate được học ngoài public set.

TAT-QA và FinMath đều tách evidence khỏi symbolic/tree reasoning. Điều này khớp trực tiếp với kết quả P2.4: `llm_select` thất bại nhiều do operand/typing, chứ không phải chỉ thiếu khả năng viết pandas.

APOLLO chỉ ra hai chi tiết đặc biệt phù hợp:

- hard negative phải chứa số nhưng sai role/period/entity, không chỉ random negative;
- program tương đương về execution cần được coi là đúng dù AST khác gold.

Với R2AI, evaluator nên canonicalize AST hoặc so execution trên nhiều counterfactual tables, thay vì so chuỗi program.

## Nhận xét chuyên sâu về từng hướng

### 1. BM25 so với dense/hybrid retrieval

Không nên thay BM25 bằng dense retrieval. Nhãn báo cáo tài chính có mã, năm và từ khóa exact; BM25 là anchor mạnh. Dense embedding hữu ích cho paraphrase, OCR variation và metric alias. Cấu hình có cơ sở là:

1. candidate generation bằng union BM25 + BGE-M3 sparse/dense/multi-vector;
2. feature hard guards cho entity, report type, year, column role;
3. cross-encoder hoặc lightweight learned reranker;
4. per-leaf top-k và global evidence budget sau khi mỗi leaf đạt minimum coverage.

Không chọn weight bằng public leaderboard. Học weight/reranker trên corpus-generated and open-data examples, đánh giá group-held-out.

### 2. Canonical metric registry

Registry nên là nguồn duy nhất cho:

- canonical ID và aliases;
- statement/sector constraints;
- qualifiers: parent/consolidated, gross/net, short/long term, current/prior period;
- unit/type;
- component metric graph;
- formula signatures;
- required/forbidden phrases và VAS code.

Mỗi match phải tạo `MetricMatch` có canonical ID, qualifier match/mismatch, lexical/dense evidence và reason. Không dùng một fuzzy score vô nghĩa xuyên nhiều metric family.

### 3. Fact-level planning

Question parser nên xuất một graph, ví dụ:

```text
root = argmax_project(
  entities=[HPG,HSG,MSR,NKG],
  score=growth_percent(
    ratio(metric_A, metric_B)@2024,
    ratio(metric_A, metric_B)@2023),
  project=entity)
```

Sau đó expand thành leaf slots có `entity`, `period`, `metric`, `role`, `expected_type`. Retrieval chạy cho từng slot và chỉ compiler biết chúng được kết hợp ra sao. Thiết kế này tránh việc một evidence mạnh che lấp leaf yếu.

### 4. Typed formula/operator registry

`formula_solver.py` của nhánh cải tiến và Selection v2 compiler nên được hợp nhất, không gọi tuần tự như hai black box. Mỗi operator cần:

- arity và ordered roles;
- input/output scalar types;
- unit algebra;
- zero/negative guards;
- allowed period relations;
- canonicalization/equivalence rule;
- deterministic compile method;
- synthetic property tests.

Ví dụ `growth_percent(end, base)` khác `percentage_point(end, base)` và khác `difference(end, base)`. Type/role phải loại nhầm trước execution.

### 5. LLM làm parser, không làm calculator

Qwen nên nhận candidate/fact graph đã ground và chỉ xuất compact AST. Không cho raw literal ngoài số xuất hiện trong question hoặc safe constants; không cho raw pandas; không cho LLM tự quyết unit scale.

Sampling nhiều lần chỉ có ích nếu:

- mỗi sample qua compiler/verifier;
- consensus so trên canonical program/answer;
- bất đồng được dùng như uncertainty;
- gate calibrated ngoài public set.

Current implementation chọn first valid sample; đây là fail-safe đơn giản nhưng bỏ qua thông tin agreement. Có thể chuyển sang `all valid → canonicalize → vote`, nhưng chỉ sau OOD ablation vì self-consistency có thể lặp lại cùng một grounded error.

### 6. Learned reranker/fine-tuning

Một finetune đáng làm hơn tiếp tục public repair là dual-task:

- query-leaf ↔ correct row/cell contrastive ranking;
- grounded context → typed AST generation.

Nguồn dữ liệu:

- open FinQA/TAT-QA/ConvFinQA programs, chuyển về unified IR;
- corpus ViFinQA tự sinh câu hỏi từ table cells/formulas;
- OCR corruptions, alias paraphrases và unit perturbations;
- hard negatives cùng report/cùng row nhưng sai year/column/qualifier.

Không đưa official 1.012 question text hoặc human P2.4 labels vào model selection mới nếu muốn giữ protocol no-public-bias. Chúng có thể được giữ trong một sealed regression report chạy sau khi candidate đã khóa.

## Kiến trúc đích đề xuất

### Stage A — canonical document layer

- versioned store;
- normalized table/cell identity;
- explicit statement/report/entity/period/unit fields;
- immutable source/context hashes.

### Stage B — hypothesis router

- tạo nhiều typed plan hypotheses;
- không chốt regex operation duy nhất;
- complexity score quyết định budget, không quyết định answer.

### Stage C — per-leaf hybrid retrieval

- BM25 + BGE-M3 union;
- canonical/qualifier features;
- hard-negative reranker;
- coverage certificate cho từng leaf.

### Stage D — planner portfolio

- deterministic formula planner;
- Qwen typed-AST planner;
- optional fine-tuned planner;
- canonicalize program và đo agreement.

### Stage E — compiler/verifier

- giữ Selection v2 compiler;
- một operator registry;
- exact evidence, type, unit, period, entity, stable-cell and replay checks;
- counterfactual replay để phát hiện constant/accidental-correct program.

### Stage F — calibrated gate

- chọn deterministic, LLM hoặc abstain/fallback bằng out-of-fold probability;
- threshold theo operation family nhưng freeze trước private;
- lưu reason code và candidate provenance.

## Ma trận thí nghiệm tối thiểu

| Trục | Control | Treatment | Metric chính |
|---|---|---|---|
| Table retrieval | main BM25 | BM25 + canonical v2 | leaf recall@k, table F2 proxy |
| Row retrieval | lexical shortlist | hybrid + reranker | exact-cell recall@k, MRR |
| Quota | report quota | per-leaf minimum quota | full-plan fact coverage |
| Reasoner | deterministic only | deterministic + typed LLM | answer/execution by op |
| Ontology | fragmented registries | unified metric/operator graph | unresolved/ambiguous rate |
| Training | pretrained only | external/synthetic QLoRA | grouped OOD accuracy |
| Arbitration | heuristic confidence | calibrated OOF gate | Brier/ECE, worst-fold accuracy |

Mỗi treatment phải được đánh giá trên random-dev và ít nhất LOTO/LOYO/LOMO/compositional holdout. Một treatment không được promote chỉ vì mean tăng nếu worst-group giảm mạnh hoặc chỉ tăng trên official-derived regression set.

## Novelty có thể tuyên bố ở mức nào

Các thành phần riêng lẻ — hybrid retrieval, typed programs, symbolic execution, hard negatives — đều có tiền lệ. Đóng góp có thể bảo vệ hơn là sự kết hợp dành cho Vietnamese OCR financial QA:

- canonical financial metric + qualifier graph liên kết trực tiếp tới per-leaf retrieval;
- typed unit/period/evidence provenance xuyên suốt compiler và submission;
- no-public-bias group-OOD protocol trong bối cảnh không có train/dev;
- pre-registered five-submission portfolio với error-correlation audit.

Chỉ nên gọi là đóng góp khoa học sau khi có ablation OOD và reproduction package; leaderboard gain một vài ID không đủ làm novelty claim.
