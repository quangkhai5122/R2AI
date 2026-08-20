# 04 — Chiến lược vòng private: không tối ưu theo public IDs

## Ràng buộc được dùng

Từ `instructions/overview.md`:

- được dùng external data nếu trích nguồn đầy đủ;
- chỉ dùng pretrained/open models có model/data công khai; không dùng closed LLM như GPT-4o/Gemini;
- model phải phát hành trước 01/06/2026 giờ Việt Nam;
- mỗi model `<= ~14B`; tổng pipeline không bị giới hạn.

Từ `instructions/data.md`: BTC không cung cấp train/dev và giữ kín gold.

Từ người dùng: private round có năm lượt nộp. Bản `submission_instructions.md` đang track chỉ nói giới hạn theo ngày và chưa ghi con số năm; do đó xem “năm private submissions” là operational rule do người dùng cung cấp, đồng thời nên lưu screenshot/export Dashboard để hoàn thiện provenance.

## Nguyên tắc nền

Năm lượt không phải năm bước Bayesian optimization trên private score. Chúng là năm candidate đã pre-register, đại diện năm giả thuyết có error profile khác nhau. Nếu Dashboard hiện score sau từng lượt, vẫn upload đúng năm hash đã khóa và không đổi candidate sau.

## Quarantine public từ thời điểm này

Không thể xóa ảnh hưởng lịch sử vì router, ontology và rules đã được phát triển sau khi đọc public questions. Có thể ngăn bias tăng thêm bằng một boundary rõ:

### Chỉ dùng để lịch sử/regression

- 1.012 official public questions và mọi exact ID list;
- P2.4 human gold 100 câu;
- leaderboard scores và post-hoc transition;
- v5.2/v5.3 single-ID/allowlist artifacts;
- `.2806/.2866` audited ID sets.

Các nguồn này có thể được chạy **sau khi candidate đã khóa** để tạo regression report, nhưng kết quả không được thay đổi threshold, prompt, metric aliases, formula coverage hoặc candidate composition.

### Được dùng để phát triển private-generalizing pipeline

- corpus reports/tables do BTC cấp, miễn question/label được sinh độc lập;
- open FinQA/TAT-QA/ConvFinQA và nguồn hợp pháp có attribution;
- synthetic questions sinh từ canonical table facts/formulas;
- lỗi OCR/alias/unit được tạo bằng transformation có kiểm chứng;
- rule clarification và submission schema chính thức.

Không seed generator bằng official question text, không tìm nearest public template và không lọc synthetic set theo public score.

## Dev protocol không dựa vào public

### 1. Tạo canonical fact/program dataset

Từ report corpus:

1. chọn exact cells/rows có value-column hợp lệ;
2. gắn canonical metric, entity, report scope, period, type và unit;
3. sinh question từ program trước, không từ public question;
4. replay program để có answer;
5. tạo paraphrases tiếng Việt và OCR perturbations;
6. giữ provenance tới report/table/row/column và generator version.

Các family tối thiểu:

- single lookup;
- difference/growth/percentage-point;
- ratio/margin và component formulas;
- sum/average/count/filter;
- argmax/argmin projection;
- `rank(growth(ratio))` và composition 3+ tầng;
- consolidated/separate qualifier;
- current/prior/beginning/ending period;
- negative/zero/unit edge cases.

### 2. Grouped splits

Không dùng random row split đơn thuần. Tạo nhiều evaluation views từ cùng dataset:

- **LOTO:** leave-one-ticker-out hoặc ticker group holdout;
- **LOYO:** year groups holdout;
- **LOMO:** metric-family holdout;
- **LORO:** report/statement type holdout;
- **Compositional:** giữ leaf metric quen thuộc nhưng operator tree mới;
- **OCR stress:** diacritics, punctuation, split rows, sticky unit và metadata columns.

Train/tune/locked phải disjoint ở group tương ứng. Locked chỉ mở sau khi code/config/candidate policy đã freeze.

### 3. Metrics

Đo riêng:

- report/table recall, precision, F2;
- exact-cell recall@k và MRR theo leaf;
- full-plan fact coverage: mọi leaf đều có evidence;
- AST validity, compiler acceptance và execution rate;
- Answer/Execution Accuracy theo operation/output type;
- rule regression;
- worst-fold và macro mean;
- calibration: Brier score/ECE và precision theo confidence bin;
- error correlation/disagreement giữa candidate.

Không promote candidate chỉ vì coverage tăng. Gate cuối cần answer gain trên target OOD stratum, không regression đáng kể ở deterministic high-precision strata và không suy giảm worst fold vượt tolerance đã pre-register.

## Shared frozen core cho cả năm candidate

Để portfolio đo đúng giả thuyết thay vì packaging noise, năm candidate dùng chung:

- versioned extraction/store;
- canonical entity/period/unit schema;
- unified metric/operator registry;
- typed compiler và semantic/replay guards;
- submission builder, `sub-k` policy và ZIP verifier;
- dependency image và model loader;
- no-public-ID guard.

Chỉ component ghi rõ trong bảng candidate được phép khác.

## Năm candidate đề xuất

### P-A — Deterministic Anchor

**Giả thuyết:** precision và auditability thắng khi private distribution lệch public.

- canonical BM25 retrieval;
- deterministic fact resolver/formula planner;
- strict typed compiler;
- không LLM override khi plan/evidence không chứng minh được;
- conservative relevant-table budget.

Vai trò: lower-variance anchor, tương quan lỗi thấp với model-generative paths và luôn có artifact dễ replay.

### P-B — Per-leaf Hybrid Retriever

**Giả thuyết:** lỗi chính của private là alias/OCR/leaf recall, không phải planner.

- union BM25 + BGE-M3 dense/sparse/multi-vector;
- canonical qualifier features;
- learned reranker bằng hard negatives;
- per-leaf minimum quota;
- reasoner/arbitration giữ giống P-A.

P-B cô lập giá trị retrieval. Nếu vừa đổi retriever vừa thêm LLM, không thể biết gain đến từ đâu.

### P-C — Typed Planner

**Giả thuyết:** private có nhiều nested/compositional questions mà deterministic coverage chưa đủ.

- retrieval tốt nhất đã chọn ngoài public;
- open model chỉ xuất compact typed AST;
- compile/verify/replay fail-closed;
- all-valid-sample canonical consensus;
- fallback về P-A khi bất đồng/không ground.

Model mặc định nên là một model 7B chắc chắn dưới limit. Qwen2.5-Coder-14B chỉ được dùng nếu BTC xác nhận cách đếm tham số; [model card chính thức](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) ghi 14.7B total và 13.1B non-embedding.

### P-D — External/Synthetic Fine-tuned Specialist

**Giả thuyết:** một planner/reranker học program supervision có thể generalize tốt hơn prompt-only.

- model open 7B, exact revision/license/release date được lưu;
- QLoRA hoặc adapter học từ FinQA/TAT-QA programs đã map sang unified IR và synthetic Vietnamese corpus;
- training không chứa official 1.012 question text/labels;
- hard-negative cell ranking và AST generation là hai task tách hoặc multi-task;
- same compiler/verifier, không raw code.

P-D phải cho thấy disagreement có ích với P-C trên OOD locked set; nếu hai model chỉ lặp cùng lỗi, không đủ lý do chiếm một submission slot.

### P-E — OOD-Calibrated Portfolio

**Giả thuyết:** không candidate đơn nào tối ưu cho mọi operation/distribution.

- lấy output đã khóa của P-A đến P-D;
- gate chỉ dùng features có sẵn trước answer: route type, leaf coverage, retrieval margin, qualifier conflicts, AST agreement, verifier reason và calibrated confidence;
- policy được train/calibrate out-of-fold;
- không dùng public ID hoặc private score;
- fallback P-A khi gate uncertainty cao.

P-E không phải “chọn candidate có public score cao nhất theo từng nhóm”. Nó là một deterministic policy có hash và prediction file hoàn chỉnh trước private round.

## Thứ tự nộp

Khóa đủ năm trước, rồi nộp theo thứ tự:

1. P-A — anchor;
2. P-B — retrieval hypothesis;
3. P-C — compositional planner;
4. P-D — trained/diverse specialist;
5. P-E — pre-calibrated ensemble.

Thứ tự này giúp đọc kết quả sau cuộc thi, nhưng không được dùng để sửa lượt sau. Nếu Dashboard cho phép chọn một submission cuối cùng theo private score, tuân rule đó; nếu private score là final hidden evaluation thì năm artifact vẫn giữ nguyên.

## Pre-registration manifest

Mỗi candidate cần một lock manifest tối thiểu:

```json
{
  "candidate_id": "P-A",
  "source_commit": "<sha256/commit>",
  "store_sha256": "<sha256>",
  "retrieval_config_sha256": "<sha256>",
  "metric_registry_sha256": "<sha256>",
  "operator_registry_sha256": "<sha256>",
  "model": {
    "id": "<open-model-id-or-none>",
    "revision": "<immutable-revision>",
    "release_date": "<verified-date>",
    "total_parameters": "<verified-count>",
    "license": "<license>"
  },
  "dev_split_fingerprint": "<sha256>",
  "dependency_lock_sha256": "<sha256>",
  "public_question_id_allowlist": [],
  "results_json_sha256": "<sha256>",
  "submission_zip_sha256": "<sha256>"
}
```

Tạo thêm một portfolio manifest chứa năm manifest hashes và timestamp. Có thể ký bằng Git tag/commit hoặc lưu ở một nơi timestamped trước private score đầu tiên.

## Guard kỹ thuật chống leakage

Private mode nên fail nếu:

- config có `expect-selected-ids`, allowlist hoặc exact official ID set;
- code diff chứa question-specific switch;
- training/eval manifest tham chiếu `p24_tune_gold` hoặc official question file;
- candidate được tạo sau timestamp khóa portfolio;
- model revision, dependency wheel hoặc store hash không khớp;
- relevant table/evidence/query replay không đạt 100%;
- raw LLM code mode được bật.

Một static scan không thể chứng minh không leakage, nhưng nó tạo bằng chứng vận hành và bắt phần lớn lỗi vô ý.

## Promotion gates trước khi chiếm một submission slot

Mỗi P-B/P-C/P-D phải có:

1. một giả thuyết và treatment duy nhất rõ ràng;
2. OOD locked improvement hoặc error diversity có ích so với P-A;
3. exact disagreement audit: trong các câu khác P-A, treatment phải đúng có điều kiện tốt hơn random;
4. không rule regression ngoài tolerance;
5. 100% compile/replay/integrity;
6. model/data license và eligibility record;
7. frozen config và artifact hash.

P-E phải được đánh giá bằng nested cross-validation để gate không học lại lỗi của candidate trên cùng fold.

## Những việc không nên làm

- Không nộp v5.3c chỉ để kiểm phép cộng năm public IDs, trừ khi cần một control lịch sử và không lấy mất private resource.
- Không mở threshold cho thêm IDs vì một public repair thành công.
- Không tune per-operation gate trên P2.4 official-derived rồi gọi đó là private-generalizing.
- Không dùng private score đầu tiên để chọn model/weight cho bốn lượt sau.
- Không giả định model tên “14B” chắc chắn hợp rule.
- Không dùng GPT/Gemini làm labeler/planner nếu rule cấm closed LLM; nếu dùng external label generation, model cũng phải nằm trong danh mục hợp lệ và có provenance.

## Tiêu chí thành công

Một chiến lược tốt không nhất thiết có public score cuối cao hơn `.2866`. Nó phải tạo được năm artifact:

- hợp rule;
- tái lập được;
- không chứa exact public-ID behavior;
- có OOD evidence và error diversity;
- cùng một verifier contract;
- đã khóa trước private feedback.

Đó là cách biến năm lượt nộp thành một portfolio nghiên cứu, thay vì năm lần dò test.
