# P2.2 Structured Selection v2 — implementation log

Ngày bắt đầu: 2026-08-11; cập nhật gần nhất: 2026-08-16. OOM schema cũ đã retire.
P2.2 B+C v5.1 đã nộp và đạt `ANSWER_ACCURACY = EXECUTION_ACCURACY = 0.2312`.
V5.2a đã đạt leaderboard `.2451`. V5.2b là CPU overlay 6 multi-operand repair có
signed silver support độc lập cho mọi toán hạng, đã dựng/audit và chờ leaderboard.

## Thay đổi production

- Thêm `vifinqa/codegen/atomic_slots.py`: planner atomic metric-slot theo role,
  ticker/year/period; registry các ratio tài chính phổ biến; chỉ được gọi bởi v2.
- Thêm `vifinqa/codegen/selection_v2.py`: parser + typed nested IR + deterministic
  single-expression compiler + trace schema 2. Guard gồm exact schema/arity, strict type,
  unit normalization, grounded literal, no duplicate stable cell, no unused binding,
  cycle/node/depth/query limits, finite/semantic execution và year projection grounding.
- Thêm `vifinqa/codegen/selection_v2_prompt.py`: JSON-only contract, provenance-rich
  candidate renderer, named facts/bindings, conditional projection và scenario-percent
  operators.
- Tích hợp `llm_mode=select_v2` vào `generate.py`; Selection v1 giữ đường code/shortlist
  riêng. V2 có atomic-aware shortlist và rescue phần fact còn thiếu.
- Runner Kaggle thêm `--llm-mode select_v2`, `--llm-ids-file`, mask hash/count trong run
  signature, resume marker chính xác; recovery runtime hiện là payload schema 6.
- Payload builder copy/fingerprint target masks; verifier bắt buộc atomic/v2 runtime files.

## Công cụ và artifact kiểm soát

- `scripts/45_make_p22_target_masks.py`: freeze B=55, C=48, BC=103 từ #19 + rescue
  audit + retrieval control; idempotent và từ chối overwrite mask khác.
- `scripts/46_audit_p22_shortlists.py`: CPU audit atomic coverage/prompt length.
- `scripts/47_validate_p22_ir_on_p24_tune.py`: oracle representability trên tune; cấm
  đường dẫn locked; không phải model accuracy.
- `scripts/48_audit_p22_codegen.py`: audit fail-closed output 1,012 ID, question/signature,
  mask attempt, trace completion, finite answer và evidence refs.
- Target masks/audits ở `artifacts/p22_targets`; payload mới ở
  `artifacts/kaggle_payload`.
- Notebook mới `kaggle/vifinqa-codegen-p22.ipynb`; notebook Selection v1 cũ không bị
  thay thành v2.

## Số đo trước GPU

- Full suite trước OOM patch: 236 passed; focused v2 cuối: 35 passed; output audit: 3 passed; runner:
  23 passed.
- B shortlist: 55/55 nonempty, 15 atomic-complete, median prompt 7,092 chars.
- C shortlist: 48/48 nonempty, 31 atomic-complete theo planner v2, median prompt
  7,632.5 chars; rescue mode 32 schema, 12 widen, 4 none.
- P2.4 tune oracle: 82/100 compile/replay verified. 18 failures là hỗn hợp adapter
  scale/sentinel/flattened-derived-constant; không được diễn giải là 18 câu compiler chắc
  chắn không biểu diễn được hay là Qwen accuracy.
- Payload schema 5: 270 files, 61/61 runtime source hashes khớp; retrieval SHA
  `96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a`;
  stable manifest SHA
  `f174502ed8fe595e96023cfc6202e2be13fa2ea304fae8b3097180cb4908b70f`.

## Quyết định vận hành

1. Chạy smoke v2 riêng.
2. Chạy/submit B trước: 55 rejected non-year, không rescue.
3. Audit + hybrid fill-only vào frozen #19.
4. Chỉ sau đó chạy/submit BC bằng cách merge C=48 fact-complete rescue vào hybrid B.
5. Không chạy toàn bộ 1,012 câu, không dùng combined mask làm lượt đầu, không thay output
   đang thành công, không dùng locked để tune.

Lệnh đầy đủ và recovery/resume contract nằm trong
`RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md` và notebook P2.2.

## Cập nhật 2026-08-13 — Stage B chunk 7/7 CUDA OOM

### Hiện tượng và bảo toàn dữ liệu

- HF đã tự hạ batch về 1 nhưng một prompt vẫn OOM với `n=2` return sequences.
- `generate.py` flush toàn bộ 1.012-row checkpoint trước khi ném exception. Chunk 7 không
  được commit; với chunk size 8, checkpoint kỳ vọng giữ 48/55 attempt, còn 7 pending.
- Không dùng kỳ vọng này thay audit. File Kaggle phải được tải/đổi tên `.partial.jsonl` và
  kiểm bằng `scripts/48_audit_p22_codegen.py --allow-incomplete`.

### Code đã thêm

- `HfBatchClient`: nếu batch 1 với `n>1` OOM, giải phóng tensor/cache rồi sinh từng sample
  tuần tự với `num_return_sequences=1`; nếu một sample đơn vẫn OOM thì fail với hướng dẫn
  tạo tail mask/token cap thấp hơn.
- `scripts/49_make_p22_oom_tail_mask.py`: fail-closed trên universe/question/single
  signature/marker+trace-v2, lấy `parent - completed`, từ chối completed ngoài mask và
  ghi mask idempotent có SHA-256 provenance.
- `kaggle/vifinqa-codegen-p22-oom-tail.ipynb`: chỉ chạy tail với output mới,
  batch/checkpoint 1, input 6000 và output 512 token.
- Payload contract nâng 5 → 6 để payload cũ bị từ chối rõ ràng. Checkpoint schema 5 vẫn
  được giữ nguyên làm provenance; tail schema 6 được merge chứ không resume.
- Payload builder loại `*.orig`, `*.rej`, `*.patch`, `*.pyc` và `__pycache__`, tránh đưa
  file backup ngoài ý muốn vào manifest/runtime Kaggle.

### Validation

- 27 focused tests cho runner/tail/audit pass.
- Full suite: **243 passed**.
- Payload schema 6 base (chưa có tail): 270 files, source/runtime không lệch; stable
  manifest `1f0a869eefd5b8276640d6197b63f2afb142dbda0ae4e2b14318d0a07a32cfcd`.
- Checkpoint Kaggle SHA `4f6c0bee...` đã xác nhận 1.012 rows, 48 completed, 7 pending,
  10 accepted và 38 rejected; audit SHA `96b2ea41...`.
- Tail mask đã freeze IDs `909,926,939,966,992,1001,1012`, SHA `2ab36ca7...`.
- Post-hoc semantic review dựng lại exact shortlist từ retrieval/store: chỉ ID 855
  plausible nhưng chưa gold-verified; 9/10 accepted còn lại thiếu grounding rõ ràng.
  Review SHA `db0c9381...` và được ghi riêng, không trộn với machine audit/gold.
- HOLD trước GPU: chưa rebuild/upload tail payload, chưa chạy tail/C và chưa hybrid. Cần
  thêm deterministic provenance/coverage guards, replay raw responses đã lưu, rồi mới
  quyết định chạy bảy ID pending.

## Cập nhật 2026-08-13 — grounded compiler, replay checkpoint và schema 7

### Guard production đã thêm

- Policy compiler tăng thành `typed_nested_ir_v2_grounded`.
- Cấm một candidate ref bind thành nhiều semantic fact; tái sử dụng phải qua cùng một
  `var`.
- `generate.py` truyền atomic plan vào compiler. Compiler fail closed nếu shortlist thiếu
  bất kỳ `F1..Fn`, program bỏ slot/dùng một slot nhiều lần, hoặc candidate lệch
  ticker/year.
- Route `ranking` bắt buộc root projection; thêm exact metric anchor bảo thủ cho
  `tổng tài sản` và `thuế thu nhập ... phải nộp`; comparative gap dạng
  “bé/lớn hơn ... bao nhiêu” không nhận answer âm.
- Prompt bắt model khai báo `slot` ở mỗi fact, dùng đủ từng routed F-slot đúng một lần,
  và trả `op=none` nếu shortlist không đủ.
- Payload contract tăng 6 → **7** để runner từ chối mọi dataset chứa compiler/prompt cũ.

### Công cụ và mask mới

- `vifinqa/codegen/selection_v2_replay.py` và
  `scripts/50_replay_p22_checkpoint.py`: kiểm universe/question/signature,
  completion marker + trace schema 2, raw response SHA/truncation, dựng lại exact
  shortlist, compile/execute/semantic-validate trên CPU và xuất artifact mới từ frozen
  #19. Không ghi đè checkpoint/control.
- `scripts/51_make_p22_groundable_mask.py`: freeze subset atomic-complete,
  non-truncated từ shortlist audit; idempotent, hash retrieval/source mask/audit và không
  đọc gold.
- B mới: `p22b_groundable_v2.json`, **15/55** ID.
- C mới: `p22c_groundable_v2.json`, **31/48** ID.
- Notebook chính dùng batch/checkpoint 1, input 6000, output 512 token; Run All dừng
  fail-closed trước C bằng `APPROVE_STAGE_C=False`.
- Notebook OOM-tail schema 6 đã retire và ném lỗi ngay ở code cell đầu.

### Kết quả replay thực tế

- Checkpoint gốc bất biến: SHA-256
  `4f6c0bee9879c61eb8ede2a98fd4e9287fc5542c2e6b068f7a0555a4488a6caa`.
- Raw trace: 48 completed, 0 response bị truncate.
- Grounded replay: 1 accepted, 47 rejected, 7 pending; transition
  `accepted→accepted=1`, `accepted→rejected=9`, `rejected→rejected=38`.
- Accepted duy nhất: ID 855, answer 2.0; vẫn chỉ plausible, không gold-verified.
- Replay SHA-256:
  `7e01b2bf848762452eb0b4ff722e8f3eed849a5ff7e68d84c6e417c43077ba15`.
- Hybrid smoke giữ 792 primary, chọn đúng 1 fallback, unresolved 219.
- Build submission smoke pass: 1.012 entry, 1.572 CSV, mọi query eval-compilable.

### Quyết định vận hành

Không chạy tail 7 câu và không chạy lại B=55. Rebuild/upload payload schema 7, chạy
B-groundable 15 trước, tải về rồi bắt buộc CPU replay/audit/hybrid. Chỉ sau review các
accepted output mới cân nhắc bật C-groundable 31. Lệnh authoritative ở mục 10 của
`RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md`.

## Cập nhật 2026-08-13 — kết quả B-groundable schema 7

Run Kaggle mới hoàn tất 15/15 không OOM. Checkpoint SHA-256
`b582c8eb7b25a893b20adf2fb9645530bdfaa573ff47692cbbdf9753e64f9d47`.
CPU replay khớp Kaggle: accepted 1 (ID 855), rejected 14, pending 0. Hybrid fill-only
dùng đúng một fallback và còn 219 structural-none.

Kết quả này xác nhận compiler guard đang hữu ích nhưng atomic planner/shortlist chưa
đủ điều kiện cho full nested IR:

- bốn complex program chạm giới hạn 512 token và kết thúc giữa JSON;
- nhiều slot được gọi là complete chỉ vì có đúng ticker/year, trong khi metric label sai;
- hai câu multi-entity C điển hình (750, 783) bị route thành một ticker/một slot, nên
  chương trình đúng là bất khả thi dù mask báo complete;
- nới type/literal guard không phải giải pháp: các chương trình bị chặn tương ứng đang
  dùng candidate sai metric.

Do đó Stage C schema 7 bị HOLD. Thiết kế kế tiếp phải chuyển từ
`slot presence` sang `semantic slot grounding`:

1. fact planner mở đủ entity × period × metric × role;
2. candidate phải có metric evidence từ label/code/table context và lưu score/anchor
   vào trace;
3. mask chỉ nhận câu có mọi semantic slot qua threshold fail-closed;
4. JSON output cần contract ngắn hơn hoặc deterministic fact bindings để model chỉ
   sinh AST; đồng thời log riêng generation termination do max token.


## Cập nhật 2026-08-14 — semantic-grounded v5 / schema 8

### Nguyên nhân và phạm vi sửa

Checkpoint B-groundable schema 7 chứng minh `atomic_fact_complete` là điều kiện quá yếu:
slot có thể đủ theo ticker/year nhưng sai metric, sai entity hoặc sai period. Bản sửa này
không mở rộng sang generic atomic/nested IR mới; nó siết evidence contract của P2.2 đã có
và giảm lượng GPU xuống subset có thể ground deterministically.

### Code đã triển khai

- `vifinqa/codegen/atomic_slots.py`: planner tách metric/entity/period/role, nhận thêm
  formula family, named target, date/period và route grounding; metric requirement registry
  fail-closed khi câu nested thiếu atomic leaf bắt buộc.
- `vifinqa/retrieval/shortlist.py`: mỗi candidate có `metric_grounded`, grounding score/
  reason và period role. Chỉ direct row/column evidence được dùng; context-only, sai VAS
  code, sai entity, sai năm và các alias domain nguy hiểm bị loại.
- `vifinqa/codegen/generate.py`: Selection v2 chỉ thấy grounded candidates; trace tách
  raw/grounded count, planner/entity guard và semantic completeness.
- `vifinqa/codegen/selection_v2_prompt.py`: JSON contract compact, fact dùng tên `F1..Fn`,
  ví dụ minified; thêm typed money literal.
- `vifinqa/codegen/selection_v2.py`: policy
  `typed_nested_ir_v2_semantic_grounded`; compiler từ chối ungrounded fact/candidate,
  hỗ trợ money literal theo đơn vị VND và giữ type/unit checks.
- `vifinqa/codegen/llm_client.py`: response vẫn tương thích `str` nhưng mang thêm
  `finish_reason`, generated token count, max token và `hit_max_tokens`.
- `vifinqa/codegen/selection_v2_replay.py` + `scripts/50_replay_p22_checkpoint.py`:
  replay subset-safe. Mặc định checkpoint attempted IDs phải exact mask; tùy chọn
  `--allow-checkpoint-superset` chỉ replay giao và audit mọi ID bị bỏ.
- Audit/mask tăng thành `p22_shortlist_audit_v2_semantic` và
  `p22_groundable_mask_v2_semantic`; payload/runner tăng schema 7 → 8.
- `kaggle/vifinqa-codegen-p22.ipynb`: B dùng mask semantic-v5 2 câu, C 4 câu,
  `APPROVE_STAGE_C=False`, output mới tách khỏi mọi checkpoint cũ.

### Kết quả đo local

- B shortlist: 55 source IDs, 15 atomic-complete nhưng chỉ **2 semantic-complete**:
  `[855,966]`.
- C rescue shortlist: 48 source IDs, 42 atomic-complete nhưng chỉ **4
  semantic-complete**: `[102,183,355,591]`.
- Checkpoint B schema 7 replay subset-safe trên B-v5: target 2, attempted 2,
  accepted 0, rejected 2; 13 attempted ID ngoài mask bị bỏ. Đây là diagnostic, không
  phải candidate submission.
- Focused tests 80/80; full suite 276/276; compileall pass.
- Payload schema 8: 288 files, source/runtime 62/62 exact, frozen retrieval hash giữ
  nguyên. Stable manifest SHA
  `b61cf8206c5802863ba36d0c7e41976d81ce2e97c083de49f5150d27b221dc67`.

### Artifact mới

- `artifacts/p22_targets/p22b_shortlist_audit_semantic_v5.json`
- `artifacts/p22_targets/p22c_shortlist_audit_semantic_v5.json`
- `artifacts/p22_targets/p22b_semantic_groundable_v5.json`
- `artifacts/p22_targets/p22c_semantic_groundable_v5.json`
- `artifacts/codegen_p22b_semantic_v5_replay_final.jsonl` và audit đi kèm
- `artifacts/kaggle_payload/` schema 8

### Giới hạn còn lại

Semantic completeness chỉ chứng minh evidence đủ để program có thể được ground; nó
không phải gold answer và không đảm bảo LLM chọn đúng AST. Vì vậy B phải audit/replay
trước, C vẫn khóa. Không dùng public leaderboard để nới threshold hoặc đưa lại 59/35
câu evidence thiếu; hướng atomic/nested IR tiếp theo chỉ nên thiết kế sau khi có output
B/C semantic-v5 và taxonomy accepted/rejected mới.
## Cập nhật 2026-08-15 — v5.1 terminal-VND + replay B/C

### Workspace cleanup

- Xác minh không tồn tại thư mục cấu hình Codex của dự án; vấn đề thực tế là 164 file
  untracked dạng .codex_*.patch/.py ở repo root, tổng 269,262 byte.
- Không source/doc nào tham chiếu các file này. Đã xóa đúng 164 file và thêm
  .codex_* vào .gitignore để tránh tái xuất hiện.
- .pytest_cache cũ có Windows deny-ACL và không thể xóa bằng quyền user hiện tại.
  Đây chỉ là cache, không ảnh hưởng source/artifact. Test hiện chạy với
  -p no:cacheprovider và --basetemp nằm trong artifacts.

### Nguyên nhân và code sửa

Store cũ gán GEG 2025 cash-flow table unit_scale=1e9, unit_source=sticky vì report có
một bảng tóm tắt tỷ VND trước báo cáo tài chính. Context bảng hiện tại kết thúc bằng bare
VND và raw value đã là VND. ID 966 vì thế bị nhân thêm 1e9 và count thành 3 thay vì 2.

- Thêm vifinqa/extraction/unit_policy.py với policy terminal_bare_vnd_v1.
- report_parser.py nhận terminal bare VND cho store build mới; multiplier tiếng Việt/
  tiếng Anh đứng trước VND không bị coi là bare.
- shortlist.py lưu stored/effective unit scale/source, resolution, terminal flag và
  SHA-256 context; frozen store được sửa ở runtime chỉ cho sticky conflict.
- selection_v2.py bump policy thành typed_nested_ir_v2_semantic_grounded_v5_1_unit;
  compiler reject mọi scale change không có provenance hợp lệ và ghi provenance vào trace.
- selection_v2_replay.py bump replay policy v5.1 và fingerprint cả report_parser.py lẫn
  unit_policy.py.
- scripts/48_audit_p22_codegen.py thêm repeatable --allow-attempted-from-mask để audit
  chained stage. Target counters không trộn inherited attempts; ID ngoài target/upstream
  vẫn fail-closed.
- Thêm tests/test_p22_unit_policy.py và hồi quy auditor Stage B→C.

Phạm vi đo trên frozen store: 13/146,246 bảng bị override, thuộc 3 report GEG/TTF.
Không rebuild store và không chạy lại Qwen.

### Replay và artifact

- B: target/attempted/accepted 2/2/2, IDs 855=2 và 966=2; 1,010 ID ngoài mask không
  semantic drift.
- C: target/attempted/accepted 4/4/4, IDs 102=1,075,116; 183=27,702,541;
  355=48.4; 591=25,479,031. Audit ghi 2 inherited attempts hợp lệ từ mask B.
- Final hybrid: 1,012 rows; chỉ sáu ID trên thay đổi so với #19; 214 unresolved.
- Submission: 1 results.json + 1,575 CSV; SHA-256
  58dd6948f1537ffed541dd52b0a3467375b025e72ff46f8c13a46ca0910577b2.
- Submission builder canonicalize text của 40 query kế thừa; AST tương đương
  40/40, question/answer không đổi, và không target v5.1 nào bị đổi query text.

- Full suite trên môi trường base: 283 passed. Target modules py_compile pass.

### Ranh giới diễn giải

Sáu kết quả là evidence-backed local candidates, chưa phải gold và chưa có leaderboard
score. ID 71/271 có dấu hiệu unit/column lỗi trong output status=ok của #19 nhưng không
được trộn vào candidate v5.1; chúng cần deterministic overlay riêng với exact allowlist,
question/cell hash và submission ablation riêng.

## Cập nhật 2026-08-16 — v5.2a column-role + period + unit repair

### Động cơ và phạm vi

- Leaderboard v5.1: TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC `.2312`;
  mức tăng so với all-types v3 là khoảng một câu đúng trên tập chấm ẩn.
- V5.2a không gọi lại Qwen và không thay 214 structural-none. Nó chỉ quét các record
  `status=ok`, route `lookup`, output `number`, đúng một fact/một DataFrame.
- Repair chỉ được kích hoạt khi ô cũ là cột note/code hoặc effective-unit policy thay đổi.
  Cột đích phải là duy nhất cho target period, hàng metric phải khớp và câu trả lời phải
  có silver support cùng dấu sau khi chuẩn hóa unit. Đây là verifier bảo thủ, không phải gold.

### Code và guard

- `vifinqa/codegen/semantic_repair.py`: policy
  `v52a_column_period_unit_silver_v1`, column-role/period resolver, effective-unit
  normalization, signed silver verifier, exact-cell query và overlay fail-closed.
- `scripts/52_build_v52a_semantic_repair.py`: CLI bắt buộc exact ID allowlist,
  primary signature/SHA-256 và retrieval SHA-256; output/audit là exclusive-create.
- `tests/test_semantic_repair_v52a.py`: hồi quy role, parser, note+unit repair,
  opposite-sign rejection, ID guard và overwrite refusal.

### Kết quả deterministic trên frozen control

- Đã chọn đúng 13 ID:
  `61,71,101,139,176,201,255,271,278,289,307,310,337`.
- Answers/support-count lần lượt:
  `1997.4/2`, `2998.87/2`, `1.33/4`, `3176645956/1`, `0.9/2`,
  `3.98/1`, `17.66/2`, `46.14/2`, `1.74/3`, `271.51/1`,
  `1.08/3`, `2.26/1`, `81.57/3`.
- ID 67 bị loại đúng chủ đích: candidate `-890.93` nhưng silver chỉ hỗ trợ
  `+890.93`; verifier không cho phép bỏ qua dấu.
- Output giữ nguyên semantic cho 999/1,012 record; 13/13 thay đổi trùng exact allowlist;
  một run signature, mọi answer hữu hạn, 214 structural-none giữ nguyên.

### Artifact và xác minh

- Codegen: `artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl`,
  SHA-256 `e339da82b8a49a3160427946d1f05ba59269c6f730e2ec4bf5d4e22864351ab4`.
- Run signature:
  `dc34176abba043ff3a0b42f1e8c5861067c82ba165bf36c29f0a641eb33b69d0`.
- Submission: `artifacts/submission_p22bc_semantic_v52a/submission.zip`,
  SHA-256 `d679eda29919fba677ec3eadb7a68fcece0142d97696d3830a89175347c5b8c7`.

## Cập nhật 2026-08-16 — v5.2b multi-operand signed-silver repair

### Động cơ và phạm vi

- V5.2a đã đạt TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC `.2451`;
  tăng `.0139`, khoảng +7/506 câu đúng ròng so với v5.1, trong khi retrieval metrics
  không đổi. Đây là bằng chứng leaderboard cho hướng deterministic semantic repair.
- V5.2b vẫn là CPU-only overlay, dùng frozen v5.2a làm primary, không gọi Qwen và không
  thay 214 structural-none.
- Phạm vi chỉ gồm historical `difference`, `growth_pct`, `ratio`, `average`;
  không mở nested/filter/ranking hoặc generic AST rewrite.

### Code và guard

- `vifinqa/codegen/semantic_repair_v52b.py`: strict AST recognizer, exact fact-to-leaf
  assignment, period/metric/unit resolution, signed silver verifier cho từng operand,
  duplicate-cell guard và formula registry cố định.
- `scripts/53_build_v52b_multi_operand_repair.py`: read-only `--preflight`; build
  bắt exact allowlist, primary signature/SHA, retrieval SHA và exclusive-create output.
- `tests/test_semantic_repair_v52b.py`: khóa AST shape, growth year ordering,
  signed formula semantics, independent support cho mọi toán hạng, opposite-sign reject
  và overwrite refusal.
- Growth luôn sắp `(end, base)` theo năm giảm dần. Ratio giữ dấu theo
  `numerator/denominator*100`; không áp dụng `abs` hậu nghiệm.

### Kết quả deterministic trên frozen v5.2a

- Preflight chấp nhận đúng 6 ID: `605,718,721,771,827,927`.
- Answer mới: `-20.73`, `-39.92`, `1.40`, `30436754.00`, `602.38`,
  `37.25`.
- Silver support count theo operand:
  `4/3`, `1/2`, `2/3`, `1/1`, `1/1/2/1`, `2/1/1/2/2`.
- Ba triggered proposal 665/667/762 bị loại vì ít nhất một toán hạng không có signed
  support. Các failure khác fail-closed, không được đưa vào allowlist.
- Output giữ nguyên semantic cho 1,006/1,012 record; đúng 6 thay đổi; 214
  structural-none giữ nguyên.

### Artifact và xác minh

- Codegen: `artifacts/codegen_p22bc_semantic_v52b_overlay.jsonl`,
  SHA-256 `51287d094488edac7b376bf6648dac289218fd1ffd69d28f9e097a1290580f4b`.
- Run signature:
  `98a638a1d0b5b58f763195578799fee2199e5c54149c28c740a21353faff242f`.
- Submission: `artifacts/submission_p22bc_semantic_v52b/submission.zip`,
  SHA-256 `90a766fa5860d6efc1a07afaf5967de4ca28a2ec963d993e4b018265b5401209`.
- Audit: 1,012 unique/finite, một signature; builder compile/replay đủ 1,012 query;
  ZIP có 1 results + 1,575 CSV; 6 canonicalized query AST-equivalent và answer khớp.
- Focused v5.2a+v5.2b: 11 passed; full suite cuối: **294 passed**.

## Cập nhật 2026-08-20 — P2.4-silver tự động và v5.3

### Kết quả v5.2b và quyết định thiết kế

- Submission v5.2b giữ nguyên TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC
  `.2451`, bằng đúng v5.2a. Vì vậy multi-operand signed-silver overlay chưa cho thấy lợi ích
  leaderboard; v5.3 quay về các thay đổi single-cell có bằng chứng độc lập và ablation hẹp.
- Frozen primary của cả hai lượt là
  `artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl`, SHA-256
  `e339da82b8a49a3160427946d1f05ba59269c6f730e2ec4bf5d4e22864351ab4`.

### P2.4-silver tự động

- `vifinqa/devset/p24_silver.py` và `scripts/54_p24_auto_silver.py` tạo fact lookup
  từ cùng metric/period xuất hiện ở hai báo cáo kề nhau, chuẩn hóa unit rồi tách ticker
  disjoint thành train/tune/locked. Bundle/output dùng exclusive-create và manifest có hash.
- Bundle canonical `artifacts/p24_silver_auto_v53` có 377 fact từ 377 report-pair,
  8 ticker mỗi split, fingerprint
  `15be1d901009ee769883552f4e4132af2d4c13da55dcc8b2e6610715923eabb5`.
- Tune: 123 fact, coverage/cell/answer accuracy `.8617886178861789`, accepted-answer
  precision `1.0`. Locked: 136 fact, các metric tương ứng `.9191176470588235`, precision
  `1.0`. Các số này đo resolver cell/period/unit trên silver lặp, không phải accuracy trên
  1,012 câu và không được diễn giải như gold ẩn.
- Bundle stress cũ `artifacts/p24_silver_auto` có 4,482 fact nhưng lượt evaluate bị dừng
  vì lặp nhiều report; không dùng nó làm bundle canonical và không báo metric từ bundle đó.

### V5.3a — single-cell consensus repair

- `vifinqa/codegen/single_cell_consensus.py` và
  `scripts/55_build_v53a_single_cell_consensus.py` chỉ xét status-ok lookup một fact,
  yêu cầu primary đáng ngờ, metric identity chặt, target period/unit rõ và support độc lập
  cùng dấu ở same/next report. Exact query/cell và toàn bộ provenance được ghi vào audit.
- Exact allowlist là `[245,329,730]`; ID 91 bị loại sau metric-identity guard vì câu hỏi
  “nguyên giá BĐS đầu tư” không đồng nhất với hàng “giá vốn cho thuê BĐS”.
- Codegen `artifacts/codegen_p22bc_semantic_v53a_overlay.jsonl`, SHA-256
  `77906d6c4dfd3adf88e7d882d45f34d6a2040da934f275d4b3ae3c2c5c44cee1`, run signature
  `439782fd55542e2269a4415a2a6c970accffd1f3a06bebaee5c0742adbe9c5b7`.
- Submission ZIP `artifacts/submission_p22bc_semantic_v53a/submission.zip`, SHA-256
  `c538f805411a7cc540f3e38d3712b8cdb0c83d315748e583f7a37690de953e88`.

### V5.3b — structural-none lookup rescue

- `scripts/56_build_v53b_lookup_rescue.py` khóa đúng 30 structural-none lookup ID và
  fail-closed. Chỉ ID `[158,213]` qua guard; 17/30 là multi-fact, 10 thiếu candidate được
  support, 1 thiếu exact report. Số structural-none giảm 214 xuống 212.
- ID 158 dùng direct-exact ownership entity/percentage-column; ID 213 dùng consensus
  period trước ở next report. Hai cơ chế được phân biệt bằng `evidence_mode` trong audit.
- Codegen `artifacts/codegen_p22bc_semantic_v53b_lookup_rescue.jsonl`, SHA-256
  `18a8adebd87c8e5b947f198cf27073aa05f5ee4cc36d3c02b37a7b29c872cf00`, run signature
  `4e794f7e4accae4e48ce8ff44e7ec5abdc700b2f1df00e748166bb58228c85e3`.
- Submission ZIP `artifacts/submission_p22bc_semantic_v53b_lookup_rescue/submission.zip`,
  SHA-256 `dbca0c56f825ed4806389f19326d7e986f0d88ffb6e3e8e7c2efc875e2317cc8`.

### Gate cuối

- Focused v5.2a/v5.2b/P2.4/v5.3: 18 passed; full suite: **301 passed**.
- Mỗi codegen có 1,012 ID duy nhất, một run signature và finite answers. Diff semantic
  đúng exact allowlist (3 cho v5.3a, 2 cho v5.3b); mọi row khác giữ nguyên.
- Mỗi submission có 1 `results.json` + 1,575 CSV; embedded results khớp file trên đĩa,
  mọi query compile/replay và answer khớp. Thứ tự leaderboard được freeze: v5.3a trước,
  sau đó v5.3b; báo đủ retrieval/answer/execution metrics cho từng lượt.

## Cập nhật 2026-08-20 — leaderboard v5.3a/v5.3b

### Kết quả đã xác nhận

- V5.3a: TABLES_F2 `.4453`, DOCS_F2 `.8975`, TABLES precision/recall/MRR5
  `.2770/.6342/.5875`, DOCS precision/recall/MRR5 `.9488/.8949/.9654`,
  ANSWER/EXEC `.2490`.
- V5.3b: TABLES_F2 `.4443`, DOCS_F2 `.8975`, TABLES precision/recall/MRR5
  `.2766/.6323/.5875`, DOCS precision/recall/MRR5 `.9488/.8949/.9654`,
  ANSWER/EXEC `.2470`.
- Frozen v5.2a là ANSWER/EXEC `.2451`. Trên 506 câu chấm, các score làm tròn khớp
  124/506, 126/506 và 125/506; vì vậy v5.3a tăng khoảng hai câu đúng ròng, v5.3b
  tăng khoảng một câu đúng ròng.

### Diễn giải và giới hạn

- V5.3a thay ba ID `[245,329,730]` nhưng score tổng chỉ cho biết tổng transition accuracy
  là +2; không thể kết luận ID nào đúng nếu không có gold riêng. TABLES_F2/recall tăng cho
  thấy nhánh này đồng thời cải thiện evidence ở tập chấm, nhưng không định danh row.
- V5.3b thay `[158,213]` và tăng một câu đúng ròng; retrieval metric giữ nguyên. Không thể
  suy ra 158 hay 213 là câu đúng, hoặc câu còn lại nằm ngoài 506 câu chấm.
- Hai overlay dùng cùng frozen v5.2a và tập ID rời nhau. Với metric accuracy cộng theo row,
  union exact năm repair có dự báo `127/506 = .250988...`, hiển thị `.2510`. Dự báo này
  phụ thuộc audit no-drift đã có nhưng vẫn cần một submission union để xác nhận.

### Quyết định tiếp theo

Tạo v5.3c bằng cách ghép nguyên vẹn ba record từ artifact v5.3a và hai record từ artifact
v5.3b lên frozen v5.2a. Không chạy lại resolver, không mở threshold và không dùng score để
chọn bớt ID. Audit phải chứng minh chỉ `[158,213,245,329,730]` thay đổi; sau khi nộp control
union này mới chuyển sang fact-slot table reranker.
