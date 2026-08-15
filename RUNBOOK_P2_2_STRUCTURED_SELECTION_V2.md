# RUNBOOK P2.2 — Structured Selection v2

> **CURRENT (2026-08-15): dùng mục 12, v5.1 đã replay/build hoàn tất trên local.**
> Không chạy lại Kaggle: raw checkpoint B=2 và C=4 đã đủ. Các mục 2–10 là lịch sử
> schema 5/6/7; mục 11 là quy trình sinh raw semantic-v5 đã hoàn thành. Mọi lệnh local
> hiện hành dùng môi trường `(base)`, không activate `.venv`.


Tài liệu này là hướng dẫn runnable cho P2.2. Control bất biến là submission #19
`artifacts/codegen_p21r_all_v3.jsonl` (ANSWER/EXEC `0.2292`). P2.2 chỉ được phép
điền các `structural-none`; không chạy lại hay ghi đè 792 output đang thành công.

## 1. Thiết kế đã khóa

- `select_v2` không sinh pandas. Qwen chỉ trả JSON gồm fact có tên, binding trung gian
  và cây typed IR lồng nhau.
- Atomic metric-slot planner tách `filter`, `rank`, `project`, `numerator`,
  `denominator`, `base`, `end` theo ticker/year/period trước khi shortlist.
- Compiler xác định kiểm schema, kiểu, unit, literal grounding, stable-cell uniqueness,
  entity/year projection, giới hạn 36 fact / 48 binding / 128 node / depth 14, execution
  và semantic validation. Policy grounded v2 còn bắt mỗi candidate-ref chỉ có một fact,
  phủ đủ và đúng một lần mọi F-slot, khớp ticker/year, ranking phải dùng projection và
  kiểm metric anchor độ chính xác cao. Bất kỳ lỗi nào đều giữ placeholder của #19.
- Mỗi candidate chỉ được khai báo ở top-level `facts`; root/binding chỉ tham chiếu tên.
  Query cuối vẫn là một expression có thể replay trong submission.
- `n=2` chỉ tăng cơ hội có JSON hợp lệ theo chính sách first-valid; đây không phải
  majority vote hay self-consistency accuracy.

Hai lượt chạy bắt buộc tách riêng:

| Lượt | Mask | Số câu | Rescue | Mục đích |
|---|---|---:|---|---|
| B | `p22b_semantic_groundable_v5.json` | 2 | tắt | mọi slot qua entity/period/metric grounding fail-closed |
| C | `p22c_semantic_groundable_v5.json` | 4 | bật | rescue nhưng mọi slot vẫn phải qua semantic grounding |

Không chạy mask BC trước B. B và C rời nhau; C chỉ chạy sau khi đã download/audit B.

## 2. Gate local trước khi upload

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
.venv\Scripts\activate

python scripts/45_make_p22_target_masks.py --dry-run
python -m pytest -q --basetemp artifacts/pytest_tmp_p22_release
python -m compileall -q vifinqa kaggle scripts tests

python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --dataset-id lequangkhai5122005/vifinqa-payload `
  --dry-run
```

Gate đã đo ở checkout hiện tại:

- full suite sau bản vá OOM: **243 passed**;
- retrieval SHA-256:
  `96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a`;
- payload schema **5**, 270 manifest files, source/runtime **61/61 khớp** là payload đã
  sinh checkpoint Stage B ban đầu;
- stable manifest SHA-256:
  `f174502ed8fe595e96023cfc6202e2be13fa2ea304fae8b3097180cb4908b70f`;
- raw `payload-manifest.json` SHA-256:
  `15d6bcff3bcbb849eecee88327de699f93bb9cd856ecbd4098532b584c64fdb7`;
- mask B SHA-256:
  `5e6b98da39a40537a7c4509fcb3583f3b1dc0ede9076749934604edab2f851a8`;
- mask C SHA-256:
  `ac8a807680a9d958bdb2759f764b3ff6c112d4aec8bfa470ce8571fe9fb65c17`.

Runtime hiện hành đã nâng lên payload schema **6** để thêm fallback `n=2 -> n=1+n=1`
khi một prompt gây OOM. Schema 5 ở trên chỉ là provenance của checkpoint cũ; không được
dùng lại cho tail/C sau khi source đã đổi.

CPU shortlist audit (đã chạy, không phải model accuracy):

- B: 55/55 shortlist nonempty; 15/55 phủ đủ mọi atomic slot; prompt median 7,092 ký
  tự; không rescue.
- C: 48/48 shortlist nonempty; 31/48 phủ đủ mọi atomic slot sau atomic rescue;
  prompt median 7,632.5 ký tự. Mask C được freeze bằng routed fact coverage của rescue
  v1; metric atomic mới nghiêm hơn nên 17 câu vẫn partial và compiler phải fail closed.
- P2.4 tune oracle: 82/100 gold AST đã compile/replay trực tiếp. 18 failure còn lại
  chủ yếu do bộ chuyển gold bảo thủ giữ scale/sentinel hoặc median đã làm phẳng; con số
  này đo representability của adapter, không phải accuracy của Qwen.

## 3. Rebuild và upload payload Kaggle

```powershell
python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --dataset-id lequangkhai5122005/vifinqa-payload

kaggle datasets version `
  -p artifacts/kaggle_payload `
  -m "P2.2 schema6 sequential OOM fallback" `
  --dir-mode zip
```

Nếu upload qua giao diện web, upload nguyên thư mục `artifacts/kaggle_payload`. Trên
Kaggle chỉ attach đúng một version payload; cell đầu notebook phải in schema 6. Số file
là 270 trước khi có tail mask và 271 sau khi `p22b_oom_tail.json` được thêm. Không dùng
`--skip-payload-verification` cho full run/submission.

Notebook chuẩn: `kaggle/vifinqa-codegen-p22.ipynb`.

## 4. Chạy Local (tùy chọn)

Runner Local phải là bản nằm **trong payload**, để runtime directory trùng manifest:

```powershell
python artifacts/kaggle_payload/code/kaggle_codegen.py `
  --payload artifacts/kaggle_payload --backend hf `
  --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit `
  --llm-mode select_v2 --llm-target empty `
  --llm-ids-file targets/p22b_rejected_non_year.json `
  --out artifacts/codegen_p22b_sel14b.jsonl `
  --n 2 --temperature 0.2 --k 0 `
  --max-tokens 768 --max-input-tokens 7000 `
  --batch-size 2 --checkpoint-every 8 `
  --time-budget-min 240 --seed 13
```

Lượt C dùng output mới và thêm rescue:

```powershell
python artifacts/kaggle_payload/code/kaggle_codegen.py `
  --payload artifacts/kaggle_payload --backend hf `
  --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit `
  --llm-mode select_v2 --llm-target empty `
  --llm-ids-file targets/p22c_rescue_fact_complete.json `
  --out artifacts/codegen_p22c_sel14b.jsonl `
  --n 2 --temperature 0.2 --k 0 `
  --rescue-no-candidates --rescue-table-k 20 --rescue-min-score 28 `
  --max-tokens 768 --max-input-tokens 7000 `
  --batch-size 2 --checkpoint-every 8 `
  --time-budget-min 240 --seed 13
```

Log full B phải có `LLM queue: 55`; log full C phải có `LLM queue: 48`. Nếu khác, dừng
và kiểm payload/mask/output cũ trước khi chạy GPU tiếp.

## 5. Chạy Kaggle

1. Import `kaggle/vifinqa-codegen-p22.ipynb`.
2. Chọn GPU T4 x2, Internet On, attach payload schema 6 mới nhất.
3. Chạy smoke. Smoke dùng file riêng, `limit=12`, `target=all`, không resume vào B/C.
4. Chạy B đến khi QA báo `target 55 / attempted 55 / pending 0`; download
   `codegen_p22b_sel14b.jsonl` ngay.
5. Chỉ sau khi giữ được B mới chạy C; QA phải báo `48 / 48 / 0`; download
   `codegen_p22c_sel14b.jsonl`.

Nếu phiên hết giữa chừng, giữ checkpoint và chạy lại **đúng toàn bộ command** vào cùng
`--out`. Không đổi batch size, checkpoint size, max token, mask, model, package/model
revision, temperature hay rescue flags trong một checkpoint. Nếu buộc phải đổi để xử lý
OOM, dùng tên output mới; run signature cố ý khác.

Riêng OOM ở Stage B chunk 7/7 ngày 2026-08-13, không resume checkpoint schema 5 bằng
runtime schema 6. Thực hiện đúng mục 9 để chỉ chạy tail đã fingerprint.

## 6. Audit, hybrid và build submission B

Đặt file Kaggle đã download tại `artifacts/codegen_p22b_sel14b.jsonl`, rồi chạy:

```powershell
python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_rejected_non_year.json `
  --out artifacts/codegen_p22b_sel14b.audit.json

python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p21r_all_v3.jsonl `
  --fallback artifacts/codegen_p22b_sel14b.jsonl `
  --out artifacts/codegen_p22b_hybrid.jsonl `
  --audit artifacts/codegen_p22b_hybrid.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22b_hybrid.jsonl `
  --out-dir artifacts/submission_p22b `
  --sub-k 5
```

Chỉ nộp `artifacts/submission_p22b/submission.zip` sau khi audit không có pending/outside
mask và build báo 1,012 query eval-compilable. Không dùng `--expand-docs`.

## 7. Audit, merge C vào B và build submission BC

```powershell
python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22c_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22c_rescue_fact_complete.json `
  --out artifacts/codegen_p22c_sel14b.audit.json

python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p22b_hybrid.jsonl `
  --fallback artifacts/codegen_p22c_sel14b.jsonl `
  --out artifacts/codegen_p22bc_hybrid.jsonl `
  --audit artifacts/codegen_p22bc_hybrid.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22bc_hybrid.jsonl `
  --out-dir artifacts/submission_p22bc `
  --sub-k 5
```

Nộp B trước BC để tách causal gain của typed IR không-rescue và rescue. Không dùng public
leaderboard để lặp threshold/prompt nhiều lần.

## 8. P2.4 tune và báo cáo kết quả

Nếu muốn so sánh offline trên tune đã hoàn tất, dùng output hybrid đầy đủ 1,012 ID và tên
report mới (evaluator không ghi đè):

```powershell
python scripts/14_p24_devset.py evaluate --split tune `
  --gold artifacts/devset_p24/p24_tune_gold.final.jsonl `
  --codegen artifacts/codegen_p22b_hybrid.jsonl `
  --output artifacts/devset_p24/eval_p22b_tune.json

python scripts/14_p24_devset.py evaluate --split tune `
  --gold artifacts/devset_p24/p24_tune_gold.final.jsonl `
  --codegen artifacts/codegen_p22bc_hybrid.jsonl `
  --output artifacts/devset_p24/eval_p22bc_tune.json
```

Không mở locked split để chọn prompt/threshold. Khi báo kết quả, gửi bốn file audit B/C,
hybrid B/BC, hai leaderboard JSON và hai codegen Kaggle; phân tích cần tách coverage,
accepted/rejected reason, output type và B so với BC.

## 9. Phục hồi Stage B sau CUDA OOM ở chunk 7/7

Checkpoint Stage B được flush ngay trước khi runner ném lỗi. Với mask 55 và
`--checkpoint-every 8`, sáu chunk đầu tương ứng 48 ID; cả chunk cuối có 7 ID không được
commit. Con số 48/7 là kỳ vọng từ lịch chunk, nhưng phải để audit đọc marker thực tế xác
nhận; không tự viết danh sách ID bằng tay.

### 9.1. Giữ và audit checkpoint gốc

Tải `/kaggle/working/codegen_p22b_sel14b.jsonl` và giữ nguyên tại:

```powershell
# artifacts/codegen_p22b_sel14b.jsonl

python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_rejected_non_year.json `
  --allow-incomplete `
  --out artifacts/codegen_p22b_sel14b.checkpoint.audit.json

python scripts/49_make_p22_oom_tail_mask.py `
  --checkpoint artifacts/codegen_p22b_sel14b.jsonl `
  --parent-mask artifacts/p22_targets/p22b_rejected_non_year.json `
  --retrieval artifacts/retrieval.jsonl `
  --expect-pending 7 `
  --out artifacts/p22_targets/p22b_oom_tail.json
```

Hai lệnh phải báo 1.012 row, đúng một run signature, `attempted=48`, `pending=7` và
cùng bảy `pending_ids`. Nếu khác, dừng và không dùng `--expect-pending` khác để ép pass;
gửi checkpoint cùng audit để kiểm tra.

Kết quả thực tế đã xác nhận ngày 2026-08-13:

- checkpoint SHA-256 `4f6c0bee9879c61eb8ede2a98fd4e9287fc5542c2e6b068f7a0555a4488a6caa`;
- `attempted=48`, `pending=7`, `accepted=10`, `rejected=38`;
- pending IDs: `909, 926, 939, 966, 992, 1001, 1012`;
- tail-mask SHA-256 `2ab36ca76fafbb64666892383ffc9aace652158af6a474da0fbd301afe1dcf13`;
- semantic review: chỉ ID 855 plausible nhưng chưa gold-verified; 9/10 accepted còn lại
  thiếu grounding metric/entity/year rõ ràng.

**HOLD:** không merge checkpoint, không upload payload tail và không chạy Stage C. Trước
hết phải thêm deterministic provenance/coverage guards và replay raw responses đã lưu.

### 9.2. Rebuild/upload payload schema 6 chứa tail mask — chỉ sau khi HOLD được gỡ

```powershell
python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --dataset-id lequangkhai5122005/vifinqa-payload

kaggle datasets version `
  -p artifacts/kaggle_payload `
  -m "P2.2 schema6 Stage-B OOM tail" `
  --dir-mode zip
```

Payload recovery phải có schema 6, `targets/p22b_oom_tail.json` nằm trong manifest và,
nếu target directory chỉ thêm đúng mask này, 271 manifest files. Không attach đồng thời
schema 5 và schema 6.

### 9.3. Chạy tail trên Kaggle

Import `kaggle/vifinqa-codegen-p22-oom-tail.ipynb`, attach đúng một payload schema 6 và
chạy toàn bộ cell. Notebook dùng:

- output mới `codegen_p22b_oom_tail_sel14b.jsonl`;
- mask tail đã fingerprint, không phải mask B=55;
- `batch-size=1`, `checkpoint-every=1`, `max-input-tokens=6000`, `max-tokens=512`;
- vẫn `n=2`; nếu hai return sequence cùng lúc OOM, runner sinh tuần tự `n=1 + n=1`.

Không copy checkpoint gốc trở lại `/kaggle/working` và không resume vào nó. Download
`codegen_p22b_oom_tail_sel14b.jsonl` sau khi QA notebook báo `attempted == target`.

### 9.4. Audit và hybrid hai bước

```powershell
python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_oom_tail_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_oom_tail.json `
  --out artifacts/codegen_p22b_oom_tail_sel14b.audit.json

# Giữ mọi output #19; chỉ lấy các câu B đã thành công trong checkpoint 48-ID.
python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p21r_all_v3.jsonl `
  --fallback artifacts/codegen_p22b_sel14b.partial.jsonl `
  --out artifacts/codegen_p22b_partial_hybrid.jsonl `
  --audit artifacts/codegen_p22b_partial_hybrid.audit.json

# Giữ kết quả trên, chỉ lấy các câu tail thành công cho structural-none còn lại.
python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p22b_partial_hybrid.jsonl `
  --fallback artifacts/codegen_p22b_oom_tail_sel14b.jsonl `
  --out artifacts/codegen_p22b_recovered_hybrid.jsonl `
  --audit artifacts/codegen_p22b_recovered_hybrid.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22b_recovered_hybrid.jsonl `
  --out-dir artifacts/submission_p22b_recovered `
  --sub-k 5
```

Chỉ sau khi audit tail pass, hai audit hybrid tồn tại và submission replay đủ 1.012 ID mới
được chạy Stage C. Khi chạy C, dùng notebook chính đã cập nhật schema 6 và output C mới.

Artifact `p22b_recovered` là recovery có kiểm soát: 48 attempt đầu đến từ payload schema 5,
tail đến từ schema 6 với token cap thấp hơn. Nó hợp lệ để cứu GPU và đo leaderboard, nhưng
không phải một run đồng nhất bitwise. Nếu cần thí nghiệm xác nhận sạch, chạy lại toàn bộ
B=55 dưới schema 6 trong một output mới ở lượt sau.

## 10. Grounded compiler v2 — quy trình hiện hành (authoritative)

> Mục này thay thế quy trình tail ở mục 9 và các lệnh B=55/C=48 cũ. Notebook
> `vifinqa-codegen-p22-oom-tail.ipynb` đã **retire và fail-fast**. Không resume
> checkpoint schema 5, không chạy tail 7 câu và không merge trực tiếp checkpoint cũ.

### 10.1. Kết quả replay checkpoint đã lưu

Compiler policy hiện hành là `typed_nested_ir_v2_grounded`. Ngoài type/schema/unit guard,
compiler bắt buộc:

- một candidate ref chỉ được bind vào một fact;
- shortlist phải có đủ mọi atomic slot `F1..Fn`, và program dùng mỗi slot đúng một lần;
- candidate phải khớp ticker/year của slot;
- route `ranking` phải có root `argmax_project` hoặc `argmin_project`;
- metric anchor độ chính xác cao và comparative-gap âm bị fail closed.

CPU replay toàn bộ raw response trong
`artifacts/codegen_p22b_sel14b.jsonl` cho kết quả:

- 1.012 row; target 55; attempted 48; pending 7;
- old accepted 10 → **1 accepted (ID 855), 9 rejected**;
- old rejected 38 → 38 rejected;
- mọi raw response đều nguyên vẹn, `raw_truncated=0`, SHA-256 khớp;
- artifact replay:
  `artifacts/codegen_p22b_sel14b_guarded_replay.jsonl`;
- audit:
  `artifacts/codegen_p22b_sel14b_guarded_replay.audit.json`;
- replay output SHA-256:
  `7e01b2bf848762452eb0b4ff722e8f3eed849a5ff7e68d84c6e417c43077ba15`.

Hybrid smoke với #19 chỉ chọn đúng một fallback (ID 855), và submission smoke đã build
đủ 1.012 entry / 1.572 CSV. Đây là validation kỹ thuật, **không phải submission được
khuyến nghị** và không chứng minh ID 855 đúng gold.

### 10.2. Mask GPU mới

Mask được freeze chỉ từ retrieval/store/shortlist, không dùng gold hay leaderboard:

| Stage | Mask | Số câu | Loại |
|---|---|---:|---|
| B | `artifacts/p22_targets/p22b_groundable_v2.json` | 15 | no-rescue, atomic-complete, không truncated |
| C | `artifacts/p22_targets/p22c_groundable_v2.json` | 31 | rescue, atomic-complete, không truncated |

B loại 40/55 câu có shortlist thiếu atomic slot. C loại 17/48 câu cùng lý do. Gửi các
câu thiếu evidence cho LLM chỉ tốn GPU và buộc compiler reject, nên không nằm trong lượt
hiện hành.

### 10.3. Gate local, build và upload payload schema 7

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
.venv\Scripts\activate

python -m pytest tests -q --basetemp artifacts/pytest_tmp_p22_schema7
python -m compileall -q vifinqa kaggle scripts tests

python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --dataset-id lequangkhai5122005/vifinqa-payload

kaggle datasets version `
  -p artifacts/kaggle_payload `
  -m "P2.2 schema7 grounded B15 C31" `
  --dir-mode zip
```

Không upload nếu manifest không có `schema_version=7`, fuzzy scorer
`difflib.SequenceMatcher/v1`, frozen retrieval SHA
`96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a`, hoặc
hai mask mới không nằm trong `manifest.files`.

Bản local đã khóa ngày 2026-08-13:

- 259 test pass; compileall pass;
- schema 7, 276 file; source Python 61/61 và runner khớp chính xác payload;
- stable manifest SHA-256
  `fcee7cf39071e27e931526846408181a415ed4f1732bf315fa0a4834c2743abb`;
- raw `payload-manifest.json` SHA-256
  `80194dd37c8697f410390b57ad787c0c458fcc219925d057eeea90e9f00501ea`;
- B-mask SHA-256 `cda759d6ae2cfeca054c14d6fefe1aa75bc75abd88f25dcf16bb005f5b2b02a8`;
- C-mask SHA-256 `e8581f7615a52d6808ae1efbe15ec7d01c0b97e3b9457eecf9d91d57ac34dc5b`.


### 10.4. Chạy Kaggle — chỉ B trước

1. Import `kaggle/vifinqa-codegen-p22.ipynb`.
2. Attach đúng **một** version payload schema 7, GPU T4 x2, Internet On.
3. Run All. Notebook dùng 14B NF4/HF, `n=2`, batch/checkpoint 1,
   `max-input-tokens=6000`, `max-tokens=512`.
4. Run All phải tự dừng ở gate `APPROVE_STAGE_C=False`. Đây là hành vi đúng.
5. Log B phải có `payload verified: schema=7`, `LLM queue: 15` và các chunk.
6. QA B phải báo `target 15 / attempted 15 / pending 0`.
7. Download `codegen_p22b_groundable_sel14b.jsonl` rồi dừng; chưa bật Stage C.

Schema 7 vẫn tự hạ batch và, nếu `n=2` OOM, sinh tuần tự `n=1 + n=1`. Nếu một
sample đơn vẫn OOM, giữ checkpoint, gửi log và file về local; không tự giảm token hay đổi
flag trong cùng output/signature.

### 10.5. Gate local bắt buộc sau khi download B

```powershell
python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_groundable_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_groundable_v2.json `
  --out artifacts/codegen_p22b_groundable_sel14b.audit.json

python scripts/50_replay_p22_checkpoint.py `
  --retrieval artifacts/retrieval.jsonl `
  --checkpoint artifacts/codegen_p22b_groundable_sel14b.jsonl `
  --control artifacts/codegen_p21r_all_v3.jsonl `
  --mask artifacts/p22_targets/p22b_groundable_v2.json `
  --store artifacts/store `
  --out artifacts/codegen_p22b_groundable_guarded.jsonl `
  --audit artifacts/codegen_p22b_groundable_guarded.audit.json `
  --k 0 --top-n 24

python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_groundable_guarded.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_groundable_v2.json `
  --out artifacts/codegen_p22b_groundable_guarded.codegen_audit.json

python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p21r_all_v3.jsonl `
  --fallback artifacts/codegen_p22b_groundable_guarded.jsonl `
  --out artifacts/codegen_p22b_groundable_hybrid.jsonl `
  --audit artifacts/codegen_p22b_groundable_hybrid.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22b_groundable_hybrid.jsonl `
  --out-dir artifacts/submission_p22b_groundable `
  --sub-k 5
```

Hai audit đầu phải báo target/attempted 15, pending 0. Chỉ output CPU replay mới được
đưa vào hybrid; không merge trực tiếp file Kaggle. Sau gate, gửi codegen B và hai audit
để review accepted output. **Chưa nộp và chưa chạy C** cho tới khi review này hoàn tất.

### 10.6. Stage C sau khi B được duyệt

Khi B đã audit xong, mở lại notebook, đổi duy nhất `APPROVE_STAGE_C=True`, rồi chạy
cell C và QA C. Expected: `LLM queue: 31`, target/attempted 31, pending 0, output
`codegen_p22c_groundable_sel14b.jsonl`. C cũng phải qua cùng chuỗi
`48_audit → 50_replay → 48_audit → hybrid fill-only → build submission`, với primary
là hybrid B đã duyệt và mask `p22c_groundable_v2.json`. Riêng replay C bắt buộc giữ
đúng contract rescue đã dùng trên Kaggle:

```powershell
python scripts/50_replay_p22_checkpoint.py `
  --retrieval artifacts/retrieval.jsonl `
  --checkpoint artifacts/codegen_p22c_groundable_sel14b.jsonl `
  --control artifacts/codegen_p22b_groundable_hybrid.jsonl `
  --mask artifacts/p22_targets/p22c_groundable_v2.json `
  --store artifacts/store `
  --out artifacts/codegen_p22c_groundable_guarded.jsonl `
  --audit artifacts/codegen_p22c_groundable_guarded.audit.json `
  --k 0 --top-n 24 `
  --rescue-no-candidates `
  --rescue-table-k 20 `
  --rescue-min-score 28
```

### 10.7. Kết quả B-groundable thực tế ngày 2026-08-13 — HOLD Stage C

Checkpoint Kaggle mới:

- file `artifacts/codegen_p22b_groundable_sel14b.jsonl`;
- SHA-256 `b582c8eb7b25a893b20adf2fb9645530bdfaa573ff47692cbbdf9753e64f9d47`;
- 1.012 row, đúng một run signature; target 15, attempted 15, pending 0;
- Kaggle trace: accepted 1, rejected 14.

CPU replay bằng frozen retrieval/store và compiler `typed_nested_ir_v2_grounded` khớp
hoàn toàn với Kaggle:

- accepted ID `[855]`, rejected 14, pending 0;
- replay SHA-256 `b2bfe8471a263c6954fb8a18b0dbec085e1de91ad14f26ac062a64ce87d9bb35`;
- hybrid giữ 792 primary, dùng 1 fallback, unresolved 219;
- hybrid signature `8b3e8fbbd4512007585e9ab74e3fd9cc96ae70e33fa56ee2411b5cd9e0f5610d`;
- hybrid SHA-256 `216c561e1cb59dcaaddbbd1f5aa1d98c00ae23194074c31f8cf1be85c70ca062`.

ID 855 chọn đúng bốn dòng `Khấu hao tài sản ngưng sử dụng`: 2015/2016 là số âm
trong ngoặc, 2021/2022 dương, nên `count(value > 0) = 2`. Kết quả có evidence và
logic nhất quán nhưng vẫn không có gold; không diễn giải đây là một câu đúng đã biết.
Không build/nộp submission chỉ từ gain một câu này.

#### Vì sao chưa bật C

Bộc lộ ba failure mode không thể giải quyết bằng chạy thêm GPU:

1. Bốn ID 391/457/527/571 có cả hai response dừng giữa JSON ở giới hạn 512 output
   token. `raw_truncated=false` chỉ nói logger không cắt chuỗi; model generation vẫn
   kết thúc trước khi đóng JSON.
2. `atomic_fact_complete` hiện chỉ chứng minh mỗi F-slot có một candidate theo
   ticker/year. Nó chưa chứng minh candidate khớp metric. Audit B cho thấy guard từ
   chối đúng các evidence sai (ví dụ ID 158 chọn dòng lợi nhuận chưa phân phối thay
   tỷ lệ sở hữu; ID 966 không có CFO đúng cho GEG).
3. Preflight 31 câu C có mismatch cứng: ID 37 không có dòng cam kết thuê; ID 102
   ghép quyền sử dụng đất vào `Vốn của tổ chức tín dụng`; ID 355 ghép 3F Việt vào
   Vissan; ID 750 chỉ có DBC dù câu hỏi cần SAB trừ DBC; ID 783 route/candidates là
   SSB dù câu hỏi cần MBB và EIB. Vì các câu C đơn giản dễ compile, chúng có nguy cơ
   được accepted với evidence sai cao hơn B.

Quyết định: **giữ `APPROVE_STAGE_C=False`; không chạy 31 câu C bằng payload schema 7**.
Bước code trước C:

1. atomic planner phải mở đủ `(entity, period, metric, role)` cho multi-entity và
   nested questions;
2. compiler/mask phải fail closed khi label/code/context không ground metric của slot;
3. prompt/IR phải giảm việc model lặp toàn bộ fact envelope và phải ghi riêng dấu hiệu
   generation chạm `max_new_tokens`;

## 11. Semantic-grounded v5 / payload schema 8 — CURRENT

Mục này thay thế toàn bộ lệnh GPU ở mục 2–10. Control vẫn là
`artifacts/codegen_p21r_all_v3.jsonl` (#19, ANSWER/EXEC `0.2292`); P2.2 chỉ được
điền `structural-none`, không được ghi đè 792 output thành công của control.

### 11.1. Frozen inputs và kết quả preflight

| Thành phần | Giá trị hiện hành |
|---|---|
| Retrieval | `artifacts/retrieval.jsonl`, SHA `96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a` |
| B mask | `artifacts/p22_targets/p22b_semantic_groundable_v5.json`, 2 IDs `[855,966]` |
| C mask | `artifacts/p22_targets/p22c_semantic_groundable_v5.json`, 4 IDs `[102,183,355,591]` |
| B mask SHA | `a12ea224b2a38f19f768e0d81be27f73f2a9281c26b72b1fe2c378ad2f12bf60` |
| C mask SHA | `32d8e21de24b8613cec04ec9903b0dd5248dd77f8e2db939e36ba94e89abda5d` |
| Payload | schema 8, 288 manifest files, source/runtime 62/62 exact |
| Stable manifest SHA | `b61cf8206c5802863ba36d0c7e41976d81ce2e97c083de49f5150d27b221dc67` |
| Raw manifest SHA | `0b31d13d6e120c0819a70cf678e52c68b3ed2948587ce8d7c9fece6cdfd56f50` |

Shortlist audit không dùng gold/leaderboard: B 55 câu chỉ còn 2 semantic-complete; C
48 câu chỉ còn 4 semantic-complete. “Complete” ở đây nghĩa là mọi slot qua entity,
period và metric grounding bảo thủ; nó vẫn chưa chứng minh answer đúng.

Checkpoint cũ `artifacts/codegen_p22b_groundable_sel14b.jsonl` (schema 7, mask 15)
đã được replay chỉ trên giao với mask mới bằng `--allow-checkpoint-superset`: target 2,
attempted 2, accepted 0, rejected 2; 13 attempted ID ngoài mask bị bỏ. Output đối chứng
là `artifacts/codegen_p22b_semantic_v5_replay_final.jsonl`, SHA
`8aa80dc9375d246c27e3e0b86f8914c5fe1bbe5e046415d1261f1d24a24bf6d4`.
Không hybrid hoặc submit artifact đối chứng này.

### 11.2. Gate local, rebuild và upload payload

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
.venv\Scripts\activate

python -m compileall -q vifinqa scripts kaggle\kaggle_codegen.py
python -m pytest -q -p no:cacheprovider `
  --basetemp artifacts/pytest_tmp_p22_schema8_release

python scripts/04_make_kaggle_payload.py `
  --store-dir artifacts/store `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --out artifacts/kaggle_payload `
  --dry-run

python scripts/04_make_kaggle_payload.py `
  --store-dir artifacts/store `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --out artifacts/kaggle_payload

kaggle datasets version `
  -p artifacts/kaggle_payload `
  -m "P2.2 schema8 semantic v5 B2 C4" `
  --dir-mode zip
```

Gate đo trên checkout này: compileall pass, focused P2.2 **80 passed**, full suite
**276 passed**. Không upload nếu builder không in `schema=8`, `verified files=288`,
hoặc notebook/payload không chứa đúng hai mask v5 nêu trên. Sau khi upload, attach
đúng **một version** dataset `vifinqa-payload`; gỡ mọi version schema 7 khỏi notebook.

### 11.3. Kaggle Stage B — chạy đúng 2 câu rồi dừng

1. Import `kaggle/vifinqa-codegen-p22.ipynb`.
2. Chọn GPU T4 x2, Internet On, attach đúng một payload schema 8.
3. Giữ nguyên `APPROVE_STAGE_C=False` và Run All.
4. Log bắt buộc có `payload verified: schema=8`, fuzzy scorer v1,
   `LLM queue: 2`, rồi chunk. Nếu queue là 15/31/55/1012 thì dừng: đang dùng
   notebook hoặc payload cũ.
5. QA B phải báo target 2, attempted 2, pending 0. Download
   `codegen_p22b_semantic_v5_sel14b.jsonl`; chưa chạy C.

Đặt file tải về tại `artifacts/codegen_p22b_semantic_v5_sel14b.jsonl`, rồi chạy:

```powershell
python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_semantic_v5_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_semantic_groundable_v5.json `
  --out artifacts/codegen_p22b_semantic_v5_sel14b.audit.json

python scripts/50_replay_p22_checkpoint.py `
  --retrieval artifacts/retrieval.jsonl `
  --checkpoint artifacts/codegen_p22b_semantic_v5_sel14b.jsonl `
  --control artifacts/codegen_p21r_all_v3.jsonl `
  --mask artifacts/p22_targets/p22b_semantic_groundable_v5.json `
  --store artifacts/store `
  --out artifacts/codegen_p22b_semantic_v5_guarded.jsonl `
  --audit artifacts/codegen_p22b_semantic_v5_guarded.audit.json `
  --k 0 --top-n 24

python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22b_semantic_v5_guarded.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22b_semantic_groundable_v5.json `
  --out artifacts/codegen_p22b_semantic_v5_guarded.codegen_audit.json
```

Không thêm `--allow-checkpoint-superset` cho run mới. Raw audit và replay audit phải
đều target/attempted 2, pending 0, đúng một run signature. Chỉ CPU replay được phép
đi vào hybrid. Nếu replay `accepted=0`, dừng và giữ #19; không có submission P2.2 B.
Nếu có accepted, kiểm riêng evidence/query của từng ID rồi mới chạy:

```powershell
python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p21r_all_v3.jsonl `
  --fallback artifacts/codegen_p22b_semantic_v5_guarded.jsonl `
  --out artifacts/codegen_p22b_semantic_v5_hybrid.jsonl `
  --audit artifacts/codegen_p22b_semantic_v5_hybrid.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22b_semantic_v5_hybrid.jsonl `
  --store-dir artifacts/store `
  --out-dir artifacts/submission_p22b_semantic_v5 `
  --sub-k 5
```

### 11.4. Stage C — chỉ sau khi B được audit và duyệt

Chỉ bật `APPROVE_STAGE_C=True` sau khi B đã qua toàn bộ gate trên. C phải in
`LLM queue: 4`, target/attempted 4, pending 0 và tạo
`codegen_p22c_semantic_v5_sel14b.jsonl`. Đặt file vào `artifacts/`, rồi audit/replay
với đúng contract rescue:

```powershell
python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22c_semantic_v5_sel14b.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22c_semantic_groundable_v5.json `
  --out artifacts/codegen_p22c_semantic_v5_sel14b.audit.json

python scripts/50_replay_p22_checkpoint.py `
  --retrieval artifacts/retrieval.jsonl `
  --checkpoint artifacts/codegen_p22c_semantic_v5_sel14b.jsonl `
  --control artifacts/codegen_p22b_semantic_v5_hybrid.jsonl `
  --mask artifacts/p22_targets/p22c_semantic_groundable_v5.json `
  --store artifacts/store `
  --out artifacts/codegen_p22c_semantic_v5_guarded.jsonl `
  --audit artifacts/codegen_p22c_semantic_v5_guarded.audit.json `
  --k 0 --top-n 24 `
  --rescue-no-candidates `
  --rescue-table-k 20 `
  --rescue-min-score 28

python scripts/48_audit_p22_codegen.py `
  --codegen artifacts/codegen_p22c_semantic_v5_guarded.jsonl `
  --retrieval artifacts/retrieval.jsonl `
  --mask artifacts/p22_targets/p22c_semantic_groundable_v5.json `
  --allow-attempted-from-mask artifacts/p22_targets/p22b_semantic_groundable_v5.json `
  --out artifacts/codegen_p22c_semantic_v5_guarded.codegen_audit.json

python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p22b_semantic_v5_hybrid.jsonl `
  --fallback artifacts/codegen_p22c_semantic_v5_guarded.jsonl `
  --out artifacts/codegen_p22bc_semantic_v5_hybrid.jsonl `
  --audit artifacts/codegen_p22bc_semantic_v5_hybrid.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22bc_semantic_v5_hybrid.jsonl `
  --store-dir artifacts/store `
  --out-dir artifacts/submission_p22bc_semantic_v5 `
  --sub-k 5
```

Nếu B replay accepted 0 thì chưa tồn tại B hybrid: dừng ở B và không chạy C. Nếu C
replay accepted 0, giữ nguyên B hybrid; không tạo/nộp một ZIP giống hệt chỉ đổi tên.

### 11.5. Resume/OOM contract

B=2/C=4 và prompt compact làm giảm mạnh áp lực GPU, nhưng vẫn giữ batch 1,
`max-input-tokens=6000`, `max-tokens=512`, `n=2`. Resume chỉ hợp lệ khi exact
run signature và completion marker khớp. Không đổi batch/token/mask/flags trong cùng
output. Nếu vẫn OOM ở batch 1, download checkpoint + log và dùng output mới sau khi
## 12. Semantic-grounded v5.1 terminal-VND repair — CURRENT

### 12.1. Kết luận vận hành

Hai raw checkpoint Kaggle đã hoàn tất và không cần chạy GPU lại:

- Stage B: IDs 855, 966; 2/2 mẫu cho mỗi câu đã lưu đầy đủ.
- Stage C: IDs 102, 183, 355, 591; 2/2 mẫu cho mỗi câu đã lưu đầy đủ.
- Control bất biến vẫn là artifacts/codegen_p21r_all_v3.jsonl (#19).
- Mọi output v5.1 dùng tên mới; không ghi đè artifact semantic-v5.

Stage B semantic-v5 cho ID 966 bằng 3 vì bảng GEG 2025 bị kế thừa
unit_scale=1e9 từ một bảng tóm tắt trước đó. Heading bảng dòng tiền hiện tại kết thúc
bằng bare VND và giá trị gốc 932,670,196,340 VND, nên công ty này không vượt ngưỡng
1e12. Đáp án đúng theo evidence là 2.

### 12.2. Contract code v5.1

- Parser nhận terminal bare VND là scale 1 cho store build mới, nhưng không nhầm
  million/billion/thousand hoặc triệu/tỷ/nghìn VND.
- Runtime shortlist sửa frozen store chỉ khi unit_source=sticky và context kết thúc bằng
  bare VND. Explicit/header/bare multiplier vẫn giữ nguyên.
- Compiler policy typed_nested_ir_v2_semantic_grounded_v5_1_unit ghi stored/effective
  scale, source, resolution và SHA-256 context vào selection_trace; mọi thay đổi scale
  không có provenance hợp lệ bị reject với unit_provenance_error.
- Auditor chained-stage chỉ nhận marker kế thừa khi operator truyền đúng
  --allow-attempted-from-mask. Mọi attempted ID ngoài target và upstream mask vẫn fail.
- Đo toàn frozen store: 13/146,246 bảng thỏa override, thuộc 3 report và 2 ticker
  GEG/TTF. Đây là rule hẹp, không phải thay đổi unit toàn cục.

### 12.3. Kết quả replay/build đã khóa

| Nhánh | Accepted | Kết quả |
|---|---:|---|
| B v5.1 | 2/2 | ID 855=2; ID 966=2 |
| C v5.1 | 4/4 | ID 102=1,075,116; 183=27,702,541; 355=48.4; 591=25,479,031 |
| Final hybrid | 6 fill-only | 794 kept primary, 4 fallback C, 214 unresolved |

Bất biến đã kiểm: đủ 1,012 ID, answer finite, chỉ sáu ID trên thay đổi so với #19 theo
question/answer/query/source/status/evidence, 1,006 ID còn lại không semantic drift.
Submission có 1 results.json + 1,575 CSV, không có path unsafe; mọi query compile/replay.
Lưu ý kiểm toán text: submission builder đã canonicalize cú pháp của 40 query kế
thừa; AST tương đương 40/40, question/answer không đổi và không có query nào trong
sáu target v5.1 bị đổi text.

| Artifact | SHA-256 |
|---|---|
| codegen_p22b_semantic_v51_guarded.jsonl | 4c2c4a065d9dea66bea60c4f8db704b6156776e5a6f52c0a47768f6564526615 |
| codegen_p22b_semantic_v51_hybrid.jsonl | 9df01b1b89330f14ccf1353cde95f0793fb97c4632983c12be90c34babf3dc75 |
| codegen_p22c_semantic_v51_guarded.jsonl | adf138448161f38c368c09c743fb47c38f1efa6b1b83bfb3898a0e50841f012a |
| codegen_p22bc_semantic_v51_hybrid.jsonl | 8a4b19754c16cdc6f1c1d6eeb8a5c996cb17c3bb02d5c0f739d6317b199db877 |
| submission_p22bc_semantic_v51/submission.zip | 58dd6948f1537ffed541dd52b0a3467375b025e72ff46f8c13a46ca0910577b2 |

Run signatures: B replay
9c8ec24543b9892e63164c90d56cbec9609f829ce86ee15182853b454f335275;
C replay b26d49413f4dbc826430c06a8273a52ff0edef28c80281d918f8ddb91a1ab08c;
final hybrid b70e9e2075714774f09924c61918edede28cf3d600d697b9c73c46abd76a5d6d.

### 12.4. Lệnh kiểm tra hiện hành trên môi trường (base)

Không activate env. Các artifact trên đã tồn tại; block này chỉ kiểm tra, không replay
hay ghi đè:

~~~powershell
cd D:\Python_Project\Hackathon\R2AI_2026
python -m pytest -p no:cacheprovider --basetemp artifacts\pytest_tmp_runbook tests -q
python scripts/48_audit_p22_codegen.py --codegen artifacts/codegen_p22b_semantic_v51_guarded.jsonl --retrieval artifacts/retrieval.jsonl --mask artifacts/p22_targets/p22b_semantic_groundable_v5.json --out artifacts/codegen_p22b_semantic_v51_guarded.codegen_audit.json
python scripts/48_audit_p22_codegen.py --codegen artifacts/codegen_p22bc_semantic_v51_hybrid.jsonl --retrieval artifacts/retrieval.jsonl --mask artifacts/p22_targets/p22c_semantic_groundable_v5.json --allow-attempted-from-mask artifacts/p22_targets/p22b_semantic_groundable_v5.json --out artifacts/codegen_p22bc_semantic_v51_hybrid.codegen_audit.json
Get-FileHash -Algorithm SHA256 artifacts\submission_p22bc_semantic_v51\submission.zip
~~~

Expected test gate: 283 passed. Expected ZIP hash phải đúng giá trị ở bảng trên. Hai lệnh
audit có thể ghi lại JSON audit cùng nội dung; không chạy replay/build khi chưa đổi suffix
output hoặc di chuyển artifact hiện tại.

### 12.5. Quyết định tiếp theo

Candidate để nộp một lần là:

artifacts/submission_p22bc_semantic_v51/submission.zip

Chưa gộp sửa deterministic cho ID 71/271 vào candidate này. Hai ID đó đang là output
status=ok của #19 nên cần overlay allowlist/provenance riêng và một submission ablation
riêng; không được âm thầm thay bằng hybrid fallback. Sau khi có score v5.1, ưu tiên
column-role/unit overlay CPU trước khi mở rộng mask hoặc chạy thêm Qwen.
quyết định cấu hình; không tự giảm token rồi resume vào file cũ.
