# P2.2 Structured Selection v2 — implementation log

Ngày bắt đầu: 2026-08-11; cập nhật gần nhất: 2026-08-15. OOM schema cũ đã retire;
raw semantic-v5 B=2/C=4 đã chạy xong trên Kaggle và được replay bằng compiler v5.1
ở local. P2.2 vẫn chưa có leaderboard score.

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
