# 02 — Tiến độ session và các kết quả chính

## Cách dựng timeline

Timeline kết hợp 16 commit trên ba nhánh, hai biên bản session, P2.2 implementation log, P2.4 reports, RUNBOOK và hai `CHECKPOINTS.md` ở nhánh cải tiến. GitHub hiện không có PR/issue, nên commit/document/artifact là nguồn tiến độ chính.

Các điểm leaderboard dưới đây là **repo-reported**: chúng được ghi trong repo và commit GitHub nhưng dashboard cuộc thi không được truy cập trong lượt rà soát. Test, file count, hash và replay local được ghi riêng, không dùng để thay thế bằng chứng leaderboard.

## Timeline 2026-08-03 → 2026-08-20

### 03/08 — baseline và P0 correctness

Commit `3ed5ad7` tạo baseline; session P0 xác định lỗi Kaggle `LLM round 1: 1012 prompts` là do runner/payload cũ. Pipeline được sửa thành baseline-first, chunk/checkpoint/resume, OOM backoff và payload SHA verification.

Kết quả đáng giữ:

- submission #5, fixed `k=5`, rule-only: TABLES_F2 `0.4092`, DOCS_F2 `0.8093`, Answer/Execution `0.085`;
- 65 test pass ở cuối session;
- strict submission replay 1.012 record;
- parser/entity/unit/submission guards được đưa vào production.

Ranh giới: Kaggle runner mới khi đó mới được test/smoke local, chưa phải remote-confirmed.

### 04/08 — rule + Qwen và Selection structure

Commit `56c85c3` và `a20c257` chuyển từ LLM viết code sang LLM chọn cấu trúc/cell, rồi synthesizer tạo expression. Đây là thay đổi kiến trúc có giá trị lớn hơn việc tăng model size.

Các mốc công khai ghi trong RUNBOOK:

- raw Qwen code có Answer `0.1047` nhưng Execution `0.0613`, do 233 query nhiều dòng không phù hợp grader;
- Selection 7B đạt `0.1838`;
- Selection 14B đạt `0.1957`.

### 09/08 — P2.1 fact-aware và dynamic evidence

Hai commit `dd11f7e` và `714dd06` đưa fact-aware shortlist, dynamic evidence budget và Qwen 14B Selection vào pipeline.

Submission #17 đạt TABLES_F2 `0.4406`, DOCS_F2 `0.8937`, Answer/Execution `0.2115`. Đây là checkpoint nền cho replay và P2.2 sau đó.

### 10–11/08 — replay bảo thủ và P2.4 gold

Commit `8892fc9` cùng session P2.1r/P2.4 bổ sung exact-cell selection, type/unit guards, strict-empty rescue, devset sampler/seal/evaluator và provenance fingerprint.

Kết quả:

- #18 year-only replay: Answer/Execution `0.2253`;
- #19 all-types replay: `0.2292`;
- full suite tại session: 192 test pass;
- human P2.4 tune gold hoàn thành 100/100, 578 evidence refs, 21/21 complex semantic audit;
- locked human set không mở.

P2.4 baseline là phát hiện khoa học quan trọng: 83/100 query chạy được nhưng chỉ 32 đúng; 51 query executable-wrong. Vì vậy coverage không còn là nút thắt duy nhất.

### 11–16/08 — Structured Selection v2

P2.2 triển khai atomic facts, typed nested IR, deterministic compiler, semantic grounding và mask-based Kaggle execution.

Các vòng schema 5–8 ghi lại một chuỗi học hợp lý:

1. schema ban đầu gọi 15/55 Stage-B câu là atomic-complete;
2. checkpoint raw có 10 accepted nhưng grounded replay chỉ giữ ID 855, loại 9 false positive;
3. semantic-complete gate giảm B còn `[855,966]`, C còn `[102,183,355,591]`;
4. terminal-bare-VND provenance sửa ID 966;
5. v5.1 chỉ thay sáu ID và đạt Answer/Execution `0.2312`;
6. v5.2a sửa column-role/period/unit có silver support và đạt `0.2451`.

Ý nghĩa: compiler fail-closed hoạt động đúng; thất bại chủ yếu nằm ở fact planner/shortlist. Số accepted giảm khi guard đúng hơn là dấu hiệu tăng precision, không phải thất bại của kiến trúc.

### 13/08 — nhánh canonical finance độc lập

Trên `improve_baseline_kien`, các commit `3dfdf38`, `783a773`, `f597979`, `5f4d2b5` thêm retrieval cải tiến, formula solver và canonical financial metric registry.

Checkpoint `.2806` ghi:

- TABLES_F2 `0.4761`;
- DOCS_F2 `0.8918`;
- Answer/Execution `0.2806`;
- 173 test pass;
- 28 public IDs bị override so với core `0.2648`.

Artifact bị ignore; chỉ hash và metadata còn trong Git.

### 15–20/08 — main v5.2/v5.3 và branch canonical v2

Trên `main`, commit `5f74157` ghi P2.2 B+C v5.1; commit `3a16292` sửa structural-none và cập nhật chuỗi v5.2/v5.3.

- v5.2a: Answer/Execution `0.2451`;
- v5.2b: không tăng so với v5.2a;
- v5.3a: `0.2490`, thay `[245,329,730]`;
- v5.3b: `0.2470`, thay `[158,213]`;
- union v5.3c `0.2510` chỉ là dự báo, chưa có score xác nhận.

Trên `improve_baseline_kien`, commit `f08b927` mở rộng canonical registry v2 lên 139 metric; `ef77d55` audit false positives và chỉ giữ sáu table-verified ID; `3b8aa2e` ghi checkpoint `.2866`.

Checkpoint `.2866` ghi:

- TABLES_F2 `0.4777`;
- DOCS_F2 `0.8945`;
- Answer/Execution `0.2866`;
- 791/1.012 question canonical-linked;
- synthetic 40 execution `0.700 → 0.725`;
- 183 test pass.

Nhánh `tranhuy`, commit `75e5269`, chủ yếu thêm model server tương thích OpenAI API. Đây là tiến độ hạ tầng, không phải một cải thiện QA đã đo.

## Bảng milestone leaderboard

| Mốc | Phương pháp | TABLES_F2 | DOCS_F2 | Answer | Execution | Diễn giải an toàn |
|---|---|---:|---:|---:|---:|---|
| #3 | line, k=10, rule | .3641 | .8399 | .0850 | .0850 | baseline |
| #5 | line, k=5, rule | .4092 | .8093 | .0850 | .0850 | k giảm tăng table F2, không đổi answer |
| #6 | raw Qwen pandas | — | — | .1047 | .0613 | execution format thất bại |
| #8 | P1 rule-only | .4241 | .8628 | .1285 | .1285 | retrieval/rule gain |
| #10 | P1.5 composite | .4337 | .8777 | .1542 | .1542 | formula coverage gain |
| #11 | ablation | — | — | .1522 | .1522 | không tốt hơn control |
| #12 | Qwen/empty | — | — | .1561 | .1561 | gain rất nhỏ |
| #13 | weak targeting | — | — | .1443 | .1443 | regression |
| #14 | Selection 7B | — | — | .1838 | .1838 | structured selection gain |
| #15 | Selection 14B | .4334 | .8774 | .1957 | .1957 | model/selection gain |
| #16 | hybrid | — | — | không tăng | không tăng | ensemble rule chưa tốt |
| #17 | fact-aware 14B | .4406 | .8937 | .2115 | .2115 | frozen P2.1 control |
| #18 | year-only replay | .4426 | .8961 | .2253 | .2253 | conservative typed replay |
| #19 | all-types replay | .4439 | .8969 | .2292 | .2292 | frozen control mới |
| #20 | P2.2 v5.1 | .4443 | .8975 | .2312 | .2312 | 6 semantic-grounded targets |
| #21 | v5.2a | .4443 | .8975 | .2451 | .2451 | column/period/unit repair |
| #22 | v5.2b | — | — | .2451 | .2451 | multi-operand overlay không tăng |
| #23 | v5.3a | .4453 | .8975 | .2490 | .2490 | 3 exact-ID repairs |
| #24 | v5.3b | .4443 | .8975 | .2470 | .2470 | 2 exact-ID rescues |
| branch checkpoint | canonical direct24+semantic4 | .4761 | .8918 | .2806 | .2806 | repo-reported, artifacts absent |
| branch checkpoint | canonical v2 audited6 | .4777 | .8945 | .2866 | .2866 | best reported, artifacts absent |

Không so trực tiếp một chiều bằng Answer: nhánh `.2866` có TABLES_F2 cao hơn `main`, nhưng DOCS_F2 thấp hơn v5.2/v5.3. Cần nhìn vector metric và error diversity.

## Kết quả artifact chạy/kiểm lại trong lượt rà soát

### Source và tests

- local `main` trùng GitHub `origin/main`, working tree sạch trước khi thêm report;
- `301 passed in 15.27s` với cache bị tắt và basetemp riêng;
- `compileall` đạt;
- `git diff --check` đạt;
- export riêng nhánh cải tiến: `183 passed in 9.11s`.

### Dữ liệu và retrieval

- 1.012 question ID duy nhất;
- 1.973 report duy nhất, không null ID;
- 146.246 table row và 2.722.031 cell row;
- retrieval đủ 1.012 record, không empty candidate;
- 1.009 câu có 20 candidate, ba câu còn lại có 9/17/10;
- retrieval SHA khớp tài liệu: `96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a`.

### Main v5.2/v5.3

- v5.2a: 1.012 record, 798 `ok`, 214 `failed`, một run signature;
- v5.3a: chỉ khác `[245,329,730]`, 798 `ok`, 214 `failed`;
- v5.3b: chỉ khác `[158,213]`, 800 `ok`, 212 `failed`;
- mỗi ZIP có đúng một JSON + 1.575 CSV;
- các hash retrieval/codegen/ZIP kiểm lại khớp RUNBOOK.

### P2.4 human gold

| Nhóm | n | đúng | Accuracy |
|---|---:|---:|---:|
| Toàn tune | 100 | 32 | .320 |
| Executable | 83 | 32 | .386 conditional |
| Non-single | 61 | 10 | .164 |
| Non-single executable | 46 | 10 | .217 conditional |
| Number | 70 | 29 | .414 |
| Percent | 16 | 1 | .063 |
| Percentage point | 3 | 0 | .000 |
| Ratio | 6 | 0 | .000 |
| Count | 2 | 0 | .000 |

Kết luận: thêm coverage hoặc thêm LLM samples không thể tự giải quyết 51 câu executable-wrong. Cần sửa leaf grounding và plan semantics.

### P2.4 auto-silver

- 377 fact/report-pair; train/tune/locked = 118/123/136;
- ticker disjoint giữa các split;
- tune accepted 106/123, accepted precision 1.0;
- locked accepted 125/136, accepted precision 1.0.

Đây chỉ là single-cell resolver benchmark; không phải 91,9% QA accuracy.

## Mâu thuẫn mẫu số 506 và 1.012

`README.md`, `RUNBOOK.md` và submission builder trên `main` ghi BTC chấm 506 trong 1.012 câu. Các mức `.2451`, `.2490`, `.2470` lần lượt phù hợp với 124/506, 126/506 và 125/506 sau làm tròn.

Ngược lại, `CHECKPOINTS.md` của nhánh cải tiến diễn giải:

- `.2648 → .2806` là `+16/1.012`;
- `.2806 → .2866` là `+6/1.012`.

Nếu mẫu số là 506, hai mức tăng tương ứng xấp xỉ +8 và +3 câu; nếu là 1.012 thì xấp xỉ +16 và +6. Cả hai không thể đồng thời đúng cho cùng một leaderboard split. Do đó:

1. giữ nguyên score aggregate vì đây là số repo ghi;
2. không kết luận “tất cả sáu ID đúng” từ score;
3. yêu cầu export/screenshot leaderboard hoặc xác nhận BTC về sample count của checkpoint ngày 20/08;
4. không dùng số câu quy đổi để calibrate private candidate.

## Điều đã được chứng minh và điều chưa được chứng minh

### Đã được hỗ trợ tốt

- k=5 tốt hơn k=10 trong ablation đã nộp cho TABLES_F2;
- structured selection tốt hơn raw code generation;
- rule-first có precision tốt hơn LLM selection trên P2.4 tune;
- typed compiler/replay loại được nhiều false positive;
- canonical metric v2 tăng linkage coverage và retrieval metrics theo checkpoint;
- artifact main gần đây có integrity/replay tốt.

### Chưa được chứng minh

- `.2866` có thể replay từ repo hiện tại;
- sáu audited ID của `.2866` đều là câu được chấm và đều đúng;
- synthetic 40 hoặc auto-silver dự đoán private QA accuracy;
- P2.4 tune là đánh giá không bias public;
- Qwen 14B chắc chắn hợp rule tham số;
- v5.3c union sẽ generalize private;
- năm sửa public nhỏ là chiến lược tốt hơn một portfolio pre-registered.

## Nhận xét về tiến độ

Tiến độ engineering là mạnh: trong 18 ngày, dự án đi từ baseline rule sang pipeline typed, crash-safe và artifact-audited. Điểm yếu không phải thiếu ý tưởng mà là **quá nhiều thế hệ policy cùng tồn tại và validation phụ thuộc public questions**. Bước tiếp theo nên giảm số nhánh/overlay, hợp nhất abstraction và nâng chất lượng phép đo; không nên tăng thêm một v5.x chỉ để đổi vài public IDs.
