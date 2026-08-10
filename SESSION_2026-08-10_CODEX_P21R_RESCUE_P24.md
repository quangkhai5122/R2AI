# Session 2026-08-10 — P2.1r, shortlist rescue, P2.4 và Kaggle reproducibility

## 1. Phạm vi

Session này triển khai các bước đứng trước P2.2 atomic/nested IR:

1. khóa scorer và payload/run provenance;
2. sửa deterministic Selection + replay bảo thủ output #17;
3. cứu shortlist chỉ khi strict shortlist rỗng;
4. dựng dev set P2.4 có gold validator/evaluator;
5. sửa checkpoint/resume để không gọi lại attempt đã hoàn tất;
6. dựng candidate CPU, payload Kaggle và hướng dẫn chạy.

**Không triển khai atomic/nested IR trong session này.** Không upload Kaggle và không chạy
Qwen từ local. Sau session, người dùng đã nộp hai candidate; score và quyết định tiếp theo
được cập nhật tại §7–§8 bên dưới.

Mốc đầu vào được giữ nguyên là submission #17:

- TABLES_F2 `.4406`;
- DOCS_F2 `.8937`;
- ANSWER/EXEC `.2115`;
- retrieval control SHA-256
  `96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a`.

## 2. Thay đổi code

### 2.1 P0 — khóa tính tái lập Kaggle

- `vifinqa/utils/viet_text.py`
  - bỏ nhánh optional RapidFuzz;
  - cố định `difflib.SequenceMatcher`, contract version `1`.
- `scripts/04_make_kaggle_payload.py`
  - payload schema `3 → 4`;
  - manifest ghi scorer backend/version.
- `kaggle/kaggle_codegen.py`
  - fail-fast nếu schema/scorer/hash sai;
  - run signature bao scorer, rescue knobs, `batch_size`, `checkpoint_every`, `limit`.
- `vifinqa/codegen/generate.py`
  - ghi `llm_attempt_status="completed"` sau khi một response group đã xử lý xong;
  - resume bằng exact run signature cho cả accepted, rejected, no-candidates,
    accepted-but-rule-kept và code-mode failure;
  - backend/debug exception chưa hoàn tất không được đánh dấu.
- `kaggle/vifinqa-codegen.ipynb`
  - kiểm schema 4/scorer;
  - smoke `all` dùng file riêng;
  - full run `empty` + rescue, không resume artifact #17.

### 2.2 P2.1r — deterministic Selection và replay

- Cell expression dùng exact `(df, row, col)`, không còn label substring + `.iloc[0]`.
- Fixed-arity phải đúng tuyệt đối; cấm operand index lặp và cấm nhiều index cùng một
  stable cell.
- Bổ sung typed `argmax/argmin → year`, grounded `count`, output-type compatibility và
  hard guard cho unit explosion rõ ràng.
- Sửa precedence của ratio bằng ngoặc quanh từng scaled operand.
- Replay mặc định chỉ thay structural-none; không ghi đè rule/LLM success.
- `--output-types year` tạo candidate bảo thủ; `all` là ablation rộng hơn.
- Replay policy v3 fingerprint:
  - input #17 + retrieval;
  - 13 semantic code files;
  - `reports.parquet` + 100 table shards được retrieval tham chiếu;
  - k/top-n/policy/output-types.
- Audit có `replayed_records` để review question, answer, query, evidence và selection.

File chính:

- `vifinqa/codegen/selection.py`
- `vifinqa/codegen/selection_replay.py`
- `scripts/12_replay_selection_p21r.py`

### 2.3 Rescue strict-empty

- `QuestionBundle` chạy strict shortlist trước.
- Chỉ nếu rỗng:
  1. lazy-widen table pool lên tối đa 20 rồi chạy strict lại;
  2. nếu vẫn rỗng, schema scorer 2D dùng row label và `0.9 × col_name`, threshold 28;
  3. nếu vẫn không có candidate, giữ none.
- `selection_trace.shortlist` ghi mode/count để audit.
- `scripts/13_audit_shortlist_rescue.py` chỉ audit saved no-candidate IDs, không gọi LLM.

Kết quả CPU trên 142 no-candidate của #17:

- `widen_strict`: 24;
- `schema_2d`: 83;
- vẫn rỗng: 35;
- tổng có shortlist trở lại: 107/142.

Đây là candidate coverage, không phải 107 answer đúng.

### 2.4 Entity/retrieval ablation

- Thêm exact brand aliases MBBank/MBB, Eximbank/EIB, Sabeco/SAB.
- Sửa parent-company scope cho câu có nhiều cụm so sánh.
- Dựng `artifacts/retrieval_rescue.jsonl`, SHA-256
  `f41890fb38b3d9bb19ba835c4e639131b030d2ac69a93591c7badc58a71f6d33`.

Rule control cho thấy nhánh retrieval mới kém hơn:

| Retrieval | rule | composite | none | coverage |
|---|---:|---:|---:|---:|
| frozen schema 4 | 324 | 128 | 560 | 452 |
| reroute/retrieve mới | 316 | 110 | 586 | 426 |

Có 30 old-ok→none và chỉ 4 old-none→ok. Vì vậy retrieval mới **không được dùng cho
lượt GPU tiếp theo**; nó được giữ làm artifact chẩn đoán riêng.

### 2.5 P2.4 dev set

Thêm:

- `vifinqa/devset/p24.py`: sampler, split/hash/leakage guards, exact evidence/AST/replay
  validator và locked seal;
- `vifinqa/devset/evaluate.py`: safe hash filler + codegen evaluator;
- `scripts/14_p24_devset.py`;
- `schemas/p24_gold.schema.json`;
- `P2_4_LABELING_GUIDE.md`.

Bundle thật:

- source: 1.012;
- tune: 100;
- locked: 50;
- fingerprint:
  `311f17edcc8540d52b407c7ab84637f3052108bcb997adaf0fcf8fc04cb436d1`.

Evaluator yêu cầu codegen đủ 1.012 ID/một run signature, replay query và báo ANSWER,
EXEC, executable rate, coverage, breakdown theo stratum/output/source cùng
population-weighted aggregate. Codegen thiếu/rỗng run signature bị từ chối. Bundle và
locked seal đều one-shot, fail-closed nếu output đã tồn tại; locked evaluation bắt buộc seal.
Tune chỉ phủ 16/21 strata (thiếu 12/1.012 câu, represented mass 0,988142), nên weighted
tune chỉ có ý nghĩa có điều kiện trên strata được đại diện.

## 3. Artifact đã dựng

### 3.1 Candidate bảo thủ: year-only v3 (#18)

- codegen: `artifacts/codegen_p21r_year_only_v3.jsonl`;
- audit: `artifacts/codegen_p21r_year_only_v3.audit.json`;
- submission: `artifacts/submission_p21r_year_only_v3/submission.zip`;
- counts: 752 kept / 207 skipped output type / 25 replayed / 28 year unresolved;
- final coverage: 777/1.012, structural-none: 235;
- CSV: 1.550;
- run signature:
  `43bd291ff6f8a58ec4abfadbc2eb310ad7820f91f3d0deddb47bc548cb8a861d`;
- codegen SHA-256:
  `cec53f209909a8068526ae1fdce818da5a3319dd4e89073207df2d5f5e0d95b0`;
- ZIP SHA-256:
  `6bceddd20709ae354ded8b0aed2cdeb38579929e54743993010c1e239db04fa0`.

25 answer đều là năm hợp lệ 2015–2025. Leaderboard #18 đạt ANSWER/EXEC `.2253`, tăng
khoảng 7/506 câu đúng ròng so với #17; score aggregate không xác định ID nào đúng.

### 3.2 Candidate tốt nhất hiện tại: all-types v3 (#19)

- codegen: `artifacts/codegen_p21r_all_v3.jsonl`;
- audit: `artifacts/codegen_p21r_all_v3.audit.json`;
- submission: `artifacts/submission_p21r_all_v3/submission.zip`;
- counts: 752 kept / 40 replayed / 220 unresolved;
- breakdown: 25 year / 5 count / 4 number / 3 percent / 3 ratio;
- final coverage: 792/1.012;
- CSV: 1.569;
- run signature:
  `36469da02106a4b74aeea76bdb5d30dbe4a7e7407132bb2f98c2f6f22df43c93`;
- codegen SHA-256:
  `24203de5782a5f147c68c22fd0da3dbe15420a6a633981c3b9083bb9fdf66ad2`;
- ZIP SHA-256:
  `727a1e29b2e2bb24c043fd3ce60eaecb9859262476bcee047f15f63ca5f9aea1`.

Leaderboard #19 đạt ANSWER/EXEC `.2292`, thêm khoảng 2/506 câu đúng so với #18 và không
làm metric retrieval nào giảm. Đây là frozen control mới; manual review 5 count và 10
record number/percent/ratio vẫn hữu ích để hiểu precision theo operation.

### 3.3 Artifact cũ đã cách ly

Cả hai thế hệ pre-guard đã được cách ly: V1
`artifacts/submission_p21r_none_only/submission.zip` /
`artifacts/codegen_sel14b_factaware_p21r.jsonl` và V2
`artifacts/submission_p21r_none_only_v2/submission.zip` /
`artifacts/codegen_p21r_none_only.jsonl`. Đã thêm marker `DO_NOT_UPLOAD`/`DO_NOT_USE`;
không được nộp hay dùng làm primary.

### 3.4 Payload Kaggle schema 4

- path: `artifacts/kaggle_payload/`;
- dataset ID: `lequangkhai5122005/vifinqa-payload`;
- size: khoảng 101 MB;
- verified files: 249;
- fuzzy scorer: `difflib.SequenceMatcher/v1`;
- stable manifest digest:
  `91920b45e4184f3716d087ad4777047139a1b7e1fb4f146b61aeb20f74dc7905`;
- raw manifest SHA-256:
  `e7dea8c4d212d7519acd235bfe14e9a09703572612de44f3ad31e85032270d6e`;
- retrieval source/payload hash đều `96b71c5b31a193d…`;
- source code ↔ staged payload: 47/47 file khớp.

Payload đã dựng local nhưng **chưa upload** Kaggle.

## 4. Verification

- Full test suite: **192 passed**.
- Compileall: pass.
- `git diff --check`: pass; chỉ có cảnh báo line-ending CRLF của Windows.
- Notebook JSON: hợp lệ.
- P2.4 bundle validation: 1.012 → 100 tune / 50 locked, đúng fingerprint.
- Cả hai codegen v3: 1.012 unique ID, finite answer; mọi expression compile/replay.
- Reconstruct 40 replay operands từ frozen shortlist: 0 duplicate stable cell.
- Với 752 record không replay, answer/query/used-vars/source/status/selection-trace giữ
  nguyên tuyệt đối so với #17.

## 5. Thứ tự chạy tiếp theo

1. Giữ #19 all-types v3 làm frozen control; #18 chỉ là ablation year-only.
2. Nếu có thời gian thì gán P2.4 song song; không chờ P2.4 để bắt đầu implementation V2.
3. Chạy Selection v1 rescue hiện có làm control riêng trên payload/retrieval frozen.
4. Triển khai atomic metric-slot planner rồi typed nested IR theo §8.
5. Chỉ hybrid vào structural-none; đo rejected-non-year và rescued fact-complete thành hai
   ablation riêng, không gộp lần đầu.
6. Review accepted IR theo policy §8, build bằng cùng retrieval control rồi mới nộp.

## 6. Giới hạn còn lại

- 107 rescued shortlists chưa được Qwen chạy và chưa có gold precision.
- P2.4 sampler dùng route operation dự đoán; báo thêm breakdown theo gold AST root.
- Locked seal bảo vệ integrity, không chứng minh secrecy; freeze code/config/run signature
  trước khi mở locked.
- Payload khóa code/data/scorer nhưng chưa khóa exact Hugging Face model revision và exact
  package wheel; notebook có log runtime version. Không đổi dependency/model giữa resume.
- Gain P2.1r đã được leaderboard xác nhận ở mức aggregate; precision theo operation/ID và
  mọi gain tương lai của V2 vẫn chưa biết nếu không có P2.4 hoặc manual review.

## 7. Leaderboard #18/#19

| Metric | #17 P2.1 | #18 year-only v3 | #19 all-types v3 |
|---|---:|---:|---:|
| TABLES_F2MACRO | .4406 | .4426 | **.4439** |
| TABLES_PRECISION | – | .2759 | **.2764** |
| TABLES_RECALL | – | .6300 | **.6316** |
| TABLES_MRR5 | – | .5875 | .5875 |
| DOCS_F2MACRO | .8937 | .8961 | **.8969** |
| DOCS_PRECISION | – | .9488 | .9488 |
| DOCS_RECALL | – | .8933 | **.8943** |
| DOCS_MRR5 | – | .9654 | .9654 |
| ANSWER_ACCURACY | .2115 | .2253 | **.2292** |
| EXECUTION_ACCURACY | .2115 | .2253 | **.2292** |

<!-- Visual omission: only three discrete submission anchors are compared; the exact
table is more auditable than a chart and there is no meaningful time-series grain. -->

Public leaderboard chấm 506 câu. Các score làm tròn tương ứng #17 = 107, #18 = 114 và
#19 = 116 câu đúng: year-only tăng ròng 7 câu; 15 replay bổ sung của all-types tăng thêm
2 câu; tổng #17 → #19 tăng ròng 9 câu. Đây là phân rã aggregate, không biết ID gold nào
đúng. TABLES/DOCS tăng vì submission builder bổ sung execution evidence; không được diễn
giải thành router/retriever tự cải thiện.

Diff trực tiếp artifact: #17 → #18 đổi đúng 25 answer/query, 17 danh sách table và 8 danh
sách doc; #17 → #19 đổi 40 answer/query, 22 danh sách table và 10 danh sách doc; #18 →
#19 đổi đúng 15 answer/query gồm 5 count, 4 number, 3 percent và 3 ratio. ANSWER bằng EXEC
ở cả hai lượt, nên compiler không tạo thêm execution gap quan sát được.

## 8. Quyết định P2.2 sau leaderboard

**GO Structured Selection v2 typed nested IR**, không cần chờ hoàn tất P2.4 để bắt đầu
implementation. Tuy nhiên đây phải là fallback fill-only trên frozen control #19, không
phải chạy lại/thay thế 792 output đã thành công.

220 structural-none còn lại gồm:

- 142 `no_candidates`;
- 33 không có selection lưu được (`model_none`/parse rỗng);
- 45 có selection nhưng synthesis thất bại.

Trong 142 no-candidate, rescue tìm được candidate cho 107 nhưng chỉ 48 phủ đủ routed fact
slot; 59 thiếu slot và 35 vẫn rỗng. Trong 78 record đã có candidate, 23 output year chủ
yếu lỗi period/alias linking; target nested đầu tiên là 55 output non-year. Có 64 câu
ranking non-year trong toàn bộ 220, là nhóm rank/filter/project quan trọng nhất.

Thứ tự:

1. đóng băng #19;
2. chạy Selection v1 rescue làm control riêng;
3. thêm atomic metric-slot planner với role `filter/rank/project/numerator/denominator`;
4. typed bindings và nested arithmetic;
5. `argmin/argmax` + projection;
6. predicate/filter, `count_where` và temporal link;
7. chỉ sau đó mở rescue fact-complete, không gộp ablation đầu tiên.

Không cho model sinh/eval pandas, literal tùy ý, generic Python hoặc IR không giới hạn.
Compiler phải fail-closed theo schema/type/arity/depth/node/unit/entity-year/stable-cell,
grounding và execution replay. Nếu accepted IR ≤40, review toàn bộ; nếu nhiều hơn, review
ít nhất 30 mẫu phân tầng và toàn bộ edge case trước khi nộp.
