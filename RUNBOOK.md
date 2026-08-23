# RUNBOOK — sổ tay lệnh ViFinQA

> **File này là nguồn sự thật duy nhất về LỆNH CHẠY.** Mỗi khi pipeline đổi,
> ghi đè trực tiếp vào đây (đừng tạo file mới) để không bị lạc phiên bản.
>
> Cập nhật lần cuối: **2026-08-21** — khóa formula-aware retrieval trên
> leaderboard và chuyển Qwen unresolved sang majority self-consistency.

---

## 0. Trạng thái hiện tại (đọc trước khi chạy bất cứ thứ gì)

| Thành phần | Phiên bản/giá trị đang dùng |
|---|---|
| `vifinqa` package | 0.2.0 |
| Payload schema | 2 (có `payload-manifest.json`, SHA-256) |
| `TABLE_POS_MODE` | `line` (BTC xác nhận: vị trí = **số dòng** của `<table>`) |
| `SUBMISSION_K` | 5 |
| Retrieval chuẩn hiện tại | `artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl` |
| Rule checkpoint tương ứng | `artifacts/codegen_rule_canonical_v2_formula_operand_exact_w010_k15.jsonl` |
| Điểm tốt nhất đã nộp | TABLES_F2 .4892 / DOCS_F2 .9068 / ANSWER .2866 / EXEC .2866 |
| Backend LLM Kaggle | `hf` (transformers). **vLLM không chạy trên T4** |

`submission_formula_consensus_safe2_w010` là ablation thất bại: hai candidate
`3/3` (ID 24, 996) đều không tăng Execution. Không dùng consensus/confidence
làm verifier nếu chưa kiểm tra exact child-row và qualifier evidence.

**Kiểm tra nhanh trạng thái trước mỗi phiên làm việc:**

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
.venv\Scripts\activate
python -c "import json;d=[json.loads(l) for l in open('artifacts/retrieval.jsonl',encoding='utf-8')];print('retrieval:',len(d),'| co plan P1:', 'plan' in d[0]['route'])"
python -m pytest tests -q
```

`co plan P1: False` ⇒ retrieval là bản CŨ, phải chạy lại §2 trước khi làm gì khác.

---

## 0bis. ĐỔI GÌ THÌ PHẢI CHẠY LẠI GÌ (tra bảng này trước khi chạy)

| Sửa file ở... | 01 store | 02 retrieve | 03 rule | 04+upload payload | chạy Kaggle |
|---|:--:|:--:|:--:|:--:|:--:|
| `extraction/`, `utils/viet_num.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `router/`, `retrieval/` | – | ✅ | ✅ | ✅ | ✅ |
| `codegen/rule_*.py`, `units.py` | – | – | ✅ | ✅ | ✅ |
| `codegen/prompts.py`, `selection.py`, `generate.py`, `llm_client.py` | – | – | – | ✅ | ✅ |
| `submission/build.py`, `scripts/05` | – | – | – | – | – |
| chỉ `tests/`, `*.md` | – | – | – | – | – |

Payload luôn phải dựng lại khi **bất kỳ** file nào trong `vifinqa/` hoặc
`kaggle/kaggle_codegen.py` đổi, vì manifest băm SHA-256 toàn bộ code.

### Riêng bản P2.0 (`--llm-mode select`) — đang ở trạng thái này

Chỉ đụng `codegen/{selection,prompts,generate}.py` + `kaggle_codegen.py`.
Đã kiểm chứng: rule baseline trước/sau **giống hệt 150/150 bản ghi**.

```powershell
# BỎ QUA 01, 02, 03 — store, retrieval.jsonl, codegen_results.jsonl vẫn dùng được
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P2.0 selection mode"

# acc #2
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P2.0 selection mode"
Remove-Item Env:\KAGGLE_CONFIG_DIR
```

Rồi chạy notebook theo §7bis với `--llm-mode select`.

## 1. Cài đặt (một lần)

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest            # để chạy test
pip install kaggle            # để up payload bằng CLI
```

---

## 2. Pipeline local (CPU) — chạy theo đúng thứ tự này

### 2.1 Build store (chỉ chạy lại khi đổi code `extraction/`)

```powershell
python scripts/01_build_store.py
```

- ~30 phút cho 1.973 báo cáo. Ra `artifacts/store/`.
- Smoke: `python scripts/01_build_store.py --tickers VNM,VJC,ACB`
- **Bắt buộc chạy lại nếu** sửa `html_tables.py`, `report_parser.py`, `build_store.py`,
  hoặc `viet_num.py`.

### 2.2 Retrieval (chạy lại khi đổi `router/` hoặc `retrieval/`)

```powershell
python scripts/02_retrieve.py --out artifacts/retrieval.jsonl
```

- ~5 phút. **Đây là bước hay bị quên nhất** — mọi thay đổi router/metric/plan
  đều nằm trong file này.
- Smoke: thêm `--limit 150`.

### 2.3 Rule baseline (không cần GPU)

```powershell
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval.jsonl --out artifacts/codegen_results.jsonl
```

Kỳ vọng hiện tại: có cả `rule` (lookup) lẫn `rule_composite`
(growth/difference/ratio/ranking). Trên 150 câu đầu: `rule 123 / rule_composite 5 / none 22`.

Nếu thấy `rule_composite 0` trên toàn bộ 1.012 câu ⇒ retrieval là bản cũ
(thiếu `plan`), quay lại §2.2.

### 2.4 Build submission

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_results.jsonl --out-dir artifacts/submission
```

Nộp file: `artifacts\submission\submission.zip`

Log phải thấy: `expression-form check: all 1012 queries eval-compilable`.
Nếu thấy `[WARN] ... NOT single expressions` → xem §6.3.

**Biến thể (mỗi lần nộp chỉ đổi MỘT biến):**

```powershell
# ablation k=7
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_results.jsonl --out-dir artifacts/submission_k7 --sub-k 7

# chỉ nộp các id trong file câu hỏi chính thức (nếu BTC phát bộ riêng)
python scripts/05_build_submission.py ... --questions <duong_dan>\questions.jsonl
```

**KHÔNG dùng `--expand-docs`** — đã kiểm chứng làm sập DOCS_F2 (.84 → .61).

---

## 3. Bộ eval offline (đo trước khi tốn GPU / lượt nộp)

> ⛔ **KHÔNG BAO GIỜ NỘP BỘ EVAL.** Câu hỏi tự sinh, id riêng ⇒ leaderboard 0.0
> và mất một lượt nộp (đã dính lần #9). Luôn dùng `--offline-eval`.

```powershell
python scripts/09_gen_eval_suite.py --per-class 60
python scripts/02_retrieve.py --questions artifacts/eval/eval_questions.jsonl --out artifacts/eval/eval_retrieval.jsonl
python scripts/03_rule_baseline.py --retrieval artifacts/eval/eval_retrieval.jsonl --out artifacts/eval/eval_codegen.jsonl
python scripts/05_build_submission.py --retrieval artifacts/eval/eval_retrieval.jsonl --codegen artifacts/eval/eval_codegen.jsonl --out-dir artifacts/eval/eval_submission --offline-eval
python scripts/07_evaluate.py --submission artifacts/eval/eval_submission --gold artifacts/eval/eval_gold.json --by-class
```

**Baseline rule-only hiện tại (300 câu) — mốc so sánh cho mọi thay đổi sau này:**

| class | n | answer | exec | F2 |
|---|---:|---:|---:|---:|
| lookup | 60 | 0.783 | 0.783 | 0.556 |
| ratio_pct | 60 | 0.400 | 0.400 | 0.306 |
| growth_pct | 60 | 0.383 | 0.383 | 0.350 |
| difference | 60 | 0.283 | 0.283 | 0.329 |
| ranking | 60 | 0.133 | 0.133 | 0.332 |
| **tổng** | **300** | **0.3967** | **0.3967** | **0.374** |

(trước rule composite: 0.157 tổng, 4 lớp phức hợp đều 0.000)

`ranking` thấp là do phải đúng **cả 4** công ty trong một câu; suy ra độ chính xác
mỗi fact ≈ 0.6.

Nếu lỡ tạo `submission.zip` trong thư mục eval: xoá tay
`artifacts\eval\eval_submission\submission.zip`.

### 3.1 Formula eval v2 (bắt buộc trước khi sửa solver/retrieval phức hợp)

Bộ này dùng `2.707` fact được xác minh bằng mã VAS + canonical label + cột năm
tường minh. Với chỉ tiêu stock, gold bắt buộc lấy cột closing `31/12`, không lấy
opening/interim cùng năm. Có 288 câu, cân bằng 24 câu cho mỗi lớp công thức.

```powershell
python scripts/12_gen_formula_eval.py --per-class 24
python scripts/02_retrieve.py --questions artifacts/formula_eval/formula_questions.jsonl --out artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --depth 20 --row-rerank --row-score-weight 0.10
python scripts/03_rule_baseline.py --retrieval artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --out artifacts/formula_eval/formula_codegen_operand_cached_k15.jsonl --k 15
python scripts/05_build_submission.py --retrieval artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --codegen artifacts/formula_eval/formula_codegen_operand_cached_k15.jsonl --out-dir artifacts/formula_eval/formula_submission_operand_cached_sub5 --sub-k 5 --offline-eval
python scripts/07_evaluate.py --submission artifacts/formula_eval/formula_submission_operand_cached_sub5 --gold artifacts/formula_eval/formula_gold.json --by-class
python scripts/13_audit_formula_eval.py --retrieval artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --codegen artifacts/formula_eval/formula_codegen_operand_cached_k15.jsonl --out artifacts/formula_eval/formula_audit_operand_cached_k15.json --k 15
```

Mốc rule-only đầu tiên:

| class | Exec | evidence recall@15 | all evidence@15 |
|---|---:|---:|---:|
| growth_pct | 1.000 | 1.000 | 1.000 |
| gross_margin | 1.000 | 1.000 | 1.000 |
| average_margin_change | 1.000 | 1.000 | 1.000 |
| margin_change | 0.917 | 1.000 | 1.000 |
| debt_equity | 0.875 | 1.000 | 1.000 |
| margin_difference | 0.833 | 1.000 | 1.000 |
| ranking_ratio | 0.833 | 0.951 | 0.792 |
| count_years | 0.667 | 0.694 | 0.583 |
| count_threshold | 0.125 | 0.427 | 0.000 |
| temporal_count | 0.125 | 0.670 | 0.083 |
| count_multi_condition | 0.000 | 0.356 | 0.000 |
| nested_ranking | 0.000 | 0.604 | 0.000 |
| **tổng** | **0.6146** | | |

Mốc hiện tại sau operand requirements, exact row identity, greedy set-cover và
formula fallback guard:

| class | Exec |
|---|---:|
| average_margin_change | 1.000 |
| count_multi_condition | 0.125 |
| count_threshold | 0.917 |
| count_years | 1.000 |
| debt_equity | 0.875 |
| gross_margin | 1.000 |
| growth_pct | 0.958 |
| margin_change | 0.917 |
| margin_difference | 0.833 |
| nested_ranking | 0.292 |
| ranking_ratio | 1.000 |
| temporal_count | 0.333 |
| **tổng** | **0.7708** |

TABLES_F2 offline ở `sub-k=5` là `0.7732` (precision `0.6249`, recall
`0.9099`). So với mốc đầu, Execution tăng `+0.1562`. Đây vẫn là synthetic
benchmark; leaderboard chính thức vẫn `0.2866` cho đến khi chạy payload dưới đây.

`formula_audit_k15.json` tách riêng retrieval coverage, solver coverage,
answer accuracy khi solver đã trả lời và ba lý do từ chối phổ biến theo lớp.
Kết quả này chỉ là synthetic benchmark, tuyệt đối không upload file
`OFFLINE_EVAL_DO_NOT_UPLOAD.zip` lên leaderboard.

---

## 4. Kaggle — payload

### 4.1 Khi nào phải rebuild payload

Rebuild nếu **bất kỳ** thứ nào sau đây đổi: code trong `vifinqa/`,
`kaggle/kaggle_codegen.py`, `artifacts/store/`, `artifacts/retrieval.jsonl`.

### 4.2 Dựng payload

```powershell
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
python scripts/04_make_kaggle_payload.py --dry-run        # kiểm tra trước, không ghi đè
```

Ra `artifacts\kaggle_payload\` (~100 MB) + `payload-manifest.json`.

Payload formula-aware hiện tại:

```powershell
python scripts/02_retrieve.py --out artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl --depth 20 --row-rerank --row-score-weight 0.10
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl --out artifacts/codegen_rule_canonical_v2_formula_operand_exact_w010_k15.jsonl --k 15
python scripts/04_make_kaggle_payload.py --store-dir artifacts/store --retrieval artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl --out artifacts/kaggle-payload-formula-operand-exact-w010 --dataset-slug kaggle-payload-formula-operand-exact-w010 --dataset-id kien2005/kaggle-payload-formula-operand-exact-w010
```

Upload folder/ZIP `artifacts/kaggle-payload-formula-operand-exact-w010`, attach
dataset đó rồi import notebook
`r2ai-qwen2-5-coder-7b-formula-operand-exact-w010.ipynb`. Full run dùng
`--llm-mode select --llm-target all --k 15`; tải file
`codegen_formula_operand_sel7b_k15.jsonl` sau khi cell kiểm tra cuối cùng pass.

### 4.3 Up lên acc #1

```powershell
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P1 metric+shortlist+plan"
```

(Lần đầu tiên với một acc: đổi `version` → `create`.)

### 4.4 Up lên acc #2

```powershell
# một lần: tải kaggle.json của acc #2 vào D:\kaggle_acc2\kaggle.json
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload   # PHẢI chạy lại, id nằm trong metadata
kaggle datasets create -p artifacts\kaggle_payload --dir-mode zip               # lần đầu
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "refresh" # các lần sau
Remove-Item Env:\KAGGLE_CONFIG_DIR
```

**Bẫy:** `--dataset-id` được ghi vào `dataset-metadata.json`; nếu không chạy lại
`04_...` trước khi đổi acc, CLI sẽ đẩy nhầm acc. Kiểm tra:

```powershell
type artifacts\kaggle_payload\dataset-metadata.json
```

---

## 5. Kaggle — chạy notebook

### 5.1 Codegen (Qwen)

Notebook: `kaggle/vifinqa-codegen.ipynb` → File → Import Notebook.
Settings: **GPU T4 x2**, **Internet On**, Add Input = dataset payload (chỉ MỘT bản).

Log đúng phải có: `payload verified: schema=2` → `run signature: ...` →
`baseline written (...)` → `LLM queue: ...` → `[chunk 1/N] ...`

Cấu hình đang khuyến nghị (7B, 4-bit):

```
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --out /kaggle/working/codegen_results.jsonl \
    --n 1 --k 4 --max-tokens 256 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

Biến thể:

| Mục đích | Thêm/đổi cờ |
|---|---|
| Bật semantic matching | `--use-dense` (cần `pip install -q sentence-transformers` + `store/label_index/`) |
| Model 14B | `--model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit --batch-size 2` |
| Bỏ qua câu rule đã chắc | `--rule-first` |
| Chạy tiếp phiên trước | đặt file cũ vào `/kaggle/working/codegen_results.jsonl`, giữ NGUYÊN mọi cờ (khác cờ ⇒ `run_signature` khác ⇒ không resume) |
| OOM | `--batch-size 2`, `--k 3`, `--max-tokens 192` |

Tải `codegen_results.jsonl` từ tab Output → về local:

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <duong_dan>\codegen_results.jsonl --out-dir artifacts/submission_qwen
```

### 5.2 Embed label index (encode-only, ~15 phút)

Notebook: `kaggle/vifinqa-embed.ipynb`. GPU T4 x1 là đủ.
Tải `label_index/` → chép vào `artifacts\store\label_index\` → rebuild payload (§4).

---

## 6. Sửa lỗi thường gặp

### 6.1 Leaderboard trả 0.0 toàn bộ
Đã nộp nhầm bộ eval offline. Chỉ nộp file tên đúng `submission.zip` được sinh từ
`artifacts/retrieval.jsonl` (câu hỏi thật).

### 6.2 EXEC thấp hơn ANSWER
Có `pandas_query` nhiều dòng → grader `eval` báo SyntaxError. Sửa không cần GPU:

```powershell
python scripts/08_repair_expressions.py --submission artifacts/submission --out-dir artifacts/submission_fixed
python scripts/08_repair_expressions.py --codegen artifacts/codegen_results.jsonl   # sửa luôn ở gốc
```

### 6.3 `[WARN] ... NOT single expressions` khi build
Chạy `08_repair_expressions.py --codegen ...` rồi build lại.

### 6.4 Kaggle: `Engine core initialization failed`
vLLM V1 không chạy trên T4. Dùng `--backend hf` (mặc định notebook).

### 6.5 Kaggle: payload hash mismatch / schema
Payload cũ hoặc attach nhiều bản. Gỡ hết input, up lại theo §4, attach đúng 1 bản.

### 6.6 Windows in ra ký tự lỗi
Mọi script đã gọi `setup_stdout()` (UTF-8). Nếu vẫn lỗi: `chcp 65001`.

---

## 7. Nhật ký nộp bài (cập nhật sau mỗi lần nộp)

| # | Cấu hình | TABLES_F2 | DOCS_F2 | ANSWER | EXEC |
|---|---|---:|---:|---:|---:|
| 3 | pos=line, k=10, rule | .3641 | .8399 | .085 | .085 |
| 4 | + expand-docs ❌ | .3641 | .6066 | .085 | .085 |
| 5 | pos=line, k=5, rule | .4092 | .8093 | .085 | .085 |
| 6 | + Qwen 7B (query nhiều dòng ❌) | .4092 | .8093 | .1047 | .0613 |
| 8 | **P1 rule-only** | **.4241** | **.8628** | **.1285** | **.1285** |
| 9 | nhầm nộp bộ eval ❌ | 0 | 0 | 0 | 0 |
| 10 | **P1.5 rule composite** | **.4337** | **.8777** | **.1542** | **.1542** |
| 11 | P1.6 hợp nhất scoring | .4337 | .8777 | .1522 | .1522 |
| 12 | Qwen 7B `--llm-target empty` | .4334 | .8774 | **.1561** | .1561 |
| _ | _(điền lần nộp tiếp theo)_ | | | | |

**#12: 175 đáp án LLM mới → chỉ ~+2 câu đúng (≈2%).** Nguyên nhân đã audit:
35% query không lọc cột năm, 15% quên chia ANSWER_SCALE, 90% thiếu `regex=False`.
Chi tiết + hướng sửa: `CLAUDE.md` mục "CHẨN ĐOÁN LƯỢT QWEN #12".

**#11 trung tính** (−0.002 = 1 câu/506): bỏ 24 đáp án nhưng trong đó chỉ ~1 câu
đúng. Eval dự báo lookup +0.10 nhưng thực tế ≈ 0 ⇒ **lần thứ hai eval dự báo sai**.
Rule đã tới điểm lợi tức giảm dần: P1.5 +0.026, P1.6 −0.002.

## 7bis. Chạy Qwen song song 2 acc (giai đoạn hiện tại)

Điều kiện: đã chạy §2.2 + §2.3 với code P1.6 và rebuild payload (§4).

| | Acc #1 | Acc #2 |
|---|---|---|
| Notebook | `vifinqa-codegen.ipynb` | `vifinqa-codegen.ipynb` |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit` | `Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit --batch-size 2` |
| Out | `codegen_qwen7b.jsonl` | `codegen_qwen14b.jsonl` |
| Khác nhau đúng 1 biến | kích thước model | |

**`--llm-target` quyết định GPU tiêu vào đâu** (đo trên artifact hiện tại):

| target | số câu | ước tính 7B | ghi chú |
|---|---:|---:|---|
| `all` | 1012 | ~5.0h | phủ hết, nhưng phần lớn rule đã trả lời đúng |
| `weak` | 787 | ~3.9h | rỗng + rule confidence < 78 |
| **`empty`** | **569** | **~2.8h** | **ROI cao nhất**: chỉ câu rule bó tay |

Arbitration giữ đáp án rule khi bất đồng mà rule tự tin, nên chạy `all` phần lớn
là tiêu GPU vào việc xác nhận lại thứ đã đúng.

### `--llm-mode select` — DÙNG CÁI NÀY từ nay

Sau lượt #12 (model tự viết pandas): 35% query không lọc cột năm, 15% quên chia
`ANSWER_SCALE`, 90% thiếu `regex=False` → 175 đáp án mới chỉ ra ~2 câu đúng.

Ở `--llm-mode select`, model **chỉ xuất JSON** `{"op":..., "operands":[...]}`
chọn dòng trong shortlist; **ta** sinh biểu thức pandas với cột và đơn vị đúng.
Mô phỏng end-to-end: query thiếu `regex=False` **0%**, query không lọc cột **0%**.
Output ~30 token thay vì 256 → nhanh hơn nhiều, chạy được `--llm-target all`.

```
# ACC #1 — 7B, selection mode, TOÀN BỘ câu (arbitration mới có việc để làm)
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel7b.jsonl \
    --n 1 --k 4 --max-tokens 96 --batch-size 8 \
    --checkpoint-every 32 --time-budget-min 400

# ACC #2 — 14B, khác đúng MỘT biến (kích thước model)
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel14b.jsonl \
    --n 1 --k 4 --max-tokens 96 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

`--max-tokens 96` là đủ cho một object JSON; `--batch-size` tăng được vì output ngắn.

Chế độ cũ `--llm-mode code` vẫn giữ để đối chứng, nhưng không khuyến nghị.

Nếu phiên còn dư thời gian sau khi xong `empty`: chạy lại **cùng lệnh** đổi
`--llm-target weak` — resume sẽ giữ nguyên phần đã làm và chỉ bổ sung phần yếu.

Về local, mỗi file build một submission riêng rồi so:

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <path>\codegen_qwen7b.jsonl  --out-dir artifacts/submission_q7
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <path>\codegen_qwen14b.jsonl --out-dir artifacts/submission_q14
```

Kiểm nhanh Qwen có thực sự thêm giá trị (trước khi tốn lượt nộp):

```powershell
python -c "import json;from collections import Counter;r=[json.loads(l) for l in open(r'<path>\codegen_qwen7b.jsonl',encoding='utf-8')];print(Counter(x['source'] for x in r));print('arbitration:',Counter((x.get('arbitration') or {}).get('reason','n/a') for x in r))"
```

Đọc kết quả: `rule and llm agree` nhiều = tín hiệu tốt; `disagree; rule weak -> llm`
là phần LLM thực sự đóng góp; `source=none` còn nhiều = LLM cũng bó tay.

Ở lượt #12 toàn bộ 183 bản ghi đều là `rule produced nothing` — vì
`--llm-target empty` khiến LLM chỉ thấy câu rule bó tay nên **arbitration chưa
từng chạy**. Với `--llm-target all` sẽ thấy đủ 4 nhóm lý do.

## 8. Hiệu chuẩn eval offline ↔ leaderboard (QUAN TRỌNG)

| | eval offline | leaderboard |
|---|---:|---:|
| trước P1.5 | 0.157 | 0.1285 |
| sau P1.5 | 0.3967 | 0.1542 |
| mức tăng | **+153%** | **+20%** |

**Bộ eval phóng đại mức cải thiện khoảng 7 lần.** Hai lý do: (a) câu synthetic
dùng đúng 6 mã VAS mà rule engine biết; (b) phân bố lớp khác thực tế
(eval chia đều 20% mỗi lớp; bộ thi thật: lookup 46%, ranking 21%).

⇒ Dùng eval để biết **hướng** (tăng hay giảm), KHÔNG dùng để dự đoán **mức**.
Chỉ leaderboard mới là trọng tài về độ lớn.

## 9. Bản đồ khoảng trống hiện tại (đo trên artifact #10)

467/1012 câu có đáp án (46%); độ chính xác trên phần đã trả lời ≈ 33%.

| op | số câu | % rỗng | ghi chú |
|---|---:|---:|---|
| lookup | 468 | 25% | 118 câu — lỗ hổng dễ lấp nhất |
| **ranking** | **214** | **95%** | 203 câu — lớn nhất, nhưng là **composite lồng nhau** |
| difference | 147 | 66% | 97 câu |
| average | 70 | 91% | 64 câu |
| growth_pct | 50 | 68% | 34 câu |
| ratio | 38 | 37% | 14 câu |

**Vì sao `ranking` gần như rỗng hoàn toàn:** câu thật lồng 3 tầng, ví dụ
*"Trong nhóm HPG, HSG, MSR và NKG, doanh nghiệp có mức tăng lớn nhất từ 2023
sang 2024 của tỷ lệ X"* = rank(growth(ratio)). Rule engine hiện chỉ giải một
tầng. Bộ eval của tôi chỉ có ranking một tầng nên không lộ ra điều này.

## 10. Candidate schema-linking v3 w010 (2026-08-21)

Checkpoint chính thức vẫn là execution `0.2866`. Candidate này đã pass offline
nhưng chưa có kết quả leaderboard.

1. Upload file sau thành Kaggle Dataset, slug phải là
   `kaggle-payload-schema-linking-v3-w010`:

```text
artifacts/kaggle-payload-schema-linking-v3-w010.zip
```

2. Import và chạy notebook:

```text
r2ai-qwen2-5-coder-7b-schema-linking-v3-w010.ipynb
```

Notebook dùng input path:

```text
/kaggle/input/datasets/kien2005/kaggle-payload-schema-linking-v3-w010
```

Giữ GPU `T4 x2`, Internet On, và chạy toàn bộ cells. Cấu hình A/B được giữ
nguyên so với consensus trước: Qwen 2.5 Coder 7B, selection mode, empty-only,
`n=3`, consensus 2/3, `k=15`, temperature `0.35`, seed `13`.

3. Tải output sau về và đặt trong một folder kết quả mới:

```text
/kaggle/working/codegen_schema_linking_v3_consensus_empty_sel7b_k15_n3.jsonl
```

Output đã nhận tại
`kaggle/results-10/codegen_schema_linking_v3_consensus_empty_sel7b_k15_n3.jsonl`.
Không nộp file raw: nó làm 177 câu đang chạy được chuyển thành `failed`.

Hai override `safe2` cũ của `24` và `996` vẫn bị cấm vì chúng dùng sai
row cha. Schema-linking v3 đã sinh lại hai câu này bằng row con chính
xác; bản audited mới chỉ nhận `24,440,562,574,996` sau khi truy ngược
từng ô dữ liệu.

File cần nộp leaderboard:

```text
artifacts/submission_result10_schema_linking_v3_audited5_w010/submission.zip
```

SHA-256: `d229bf51a5d3e368ea0a33defbd7bdad9b6b15739e310833aacdbca0bcd47eea`.
Leaderboard của file này trả về `EXECUTION_ACCURACY=0.2866`, không tăng.
Với 506 câu được chấm, chỉ một câu đúng mới đã phải đẩy điểm
lên khoảng `0.2885`; do đó audited5 không tạo gain quan sát được. Không
nộp lại artifact này. Checkpoint chính thức tiếp tục là `0.2866`:

```text
artifacts/submission_formula_operand_retrieval_best2866_answers_w010/submission.zip
```

## 11. Checkpoint kết hợp chính thức 0.3004

Artifact từ nhánh `tranhuy` đã được leaderboard xác nhận:

```text
artifacts/tranhuy_02964_stockflow_exact_v11_oldretr_k5/submission.zip
```

Gói này đạt `EXECUTION_ACCURACY=0.2964`, cao hơn checkpoint cũ `0.2866`
đúng năm câu trên 506 câu được chấm. Tên thí nghiệm bên nguồn vẫn chứa
`02925`; dùng SHA-256 sau để nhận diện đúng file đã chấm:

```text
6773532be7dbe038310b9a0663090ebf17878111a8b27b53403cd24de5f23e9a
```

Không cần rerun Qwen. Bản kết hợp đã lấy nguyên answer, pandas query và
evidence của gói `0.2964`, rồi rebuild với formula-operand retrieval `w=0.10`:

```text
artifacts/submission_tranhuy_02964_answers_formula_operand_retrieval_w010/submission.zip
```

SHA-256:

```text
17750e0290d181ce13d35ef87b589ee6bdf9bcf9fa4831a6c749e8744e8e93b7
```

Offline validation: đủ 1.012 ID; answer/query/evidence giống gói `0.2964` ở
toàn bộ 1.012 câu; mọi query compile và replay khớp answer; ZIP gồm 1.553 CSV
và pass `unzip -t`. Leaderboard xác nhận kết quả kết hợp:

```text
TABLES_F2MACRO      0.4901
DOCS_F2MACRO        0.9079
EXECUTION_ACCURACY  0.2964
```

So với gói Trần Huy dùng old retrieval, TABLES_F2 tăng `0.0138`, DOCS_F2 tăng
`0.0149` và execution được giữ nguyên. So với checkpoint retrieval tốt nhất
trước đó, TABLES_F2 tăng thêm `0.0009`, DOCS_F2 tăng thêm `0.0011`, đồng thời
execution tăng năm câu đúng. Đây là checkpoint chính thức cần dùng làm nền
cho mọi thử nghiệm tiếp theo.

Batch canonical-v2 restore4, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_02964_plus_canonical_restore4_w010/submission.zip
```

SHA-256:

```text
cdeed91318c8d6eee2cc009ca4b42e5f67acf547c5072361dfb5cbae3107a7e4
```

Bản này phục hồi theo batch bốn canonical-v2 answer `589,591,635,838` mà
codegen Trần Huy đã chuyển từ `ok` về `failed`. So với checkpoint chính thức,
nó chỉ đổi bốn answer/query/evidence, không đổi relevant docs và chỉ đổi
relevant tables ở ID `838`. Toàn bộ 1.012 query replay thành công và ZIP pass
kiểm tra cấu trúc. Leaderboard xác nhận kết quả mới:

```text
TABLES_F2MACRO      0.4902
DOCS_F2MACRO        0.9079
EXECUTION_ACCURACY  0.3004
```

Batch bốn câu mang lại hai câu đúng ròng trên tập 506 câu chấm, tương đương
`+0.0040` execution. TABLES_F2 tăng nhẹ `0.0001`, DOCS_F2 giữ nguyên. Đây là
checkpoint chính thức dùng làm nền cho mọi thử nghiệm tiếp theo.

## 12. Checkpoint deterministic year-ranking exact10: 0.3083

Không cần rerun Qwen cho batch này. Solver deterministic đã xử lý các câu hỏi
trả về năm bằng phép `argmax/argmin` trên cùng một metric qua toàn bộ các năm.
Nó chỉ fill những câu đang `failed` trong checkpoint `0.3004`.

File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03004_plus_yearrank_exact10_w010/submission.zip
```

SHA-256:

```text
e4aa4fd2eb797e3274d4e5b9869b52c8bae09a7e4769e1f39393d21ad4898203
```

Allowlist gồm 10 ID:

```text
842,850,878,884,889,904,948,960,997,1000
```

Các guard trước khi build đều đã pass: đủ 1.012 ID, chỉ 10 answer/query thay
đổi so với checkpoint `0.3004`, mọi query compile và replay đúng answer trên
CSV đóng gói, `relevant_tables` dùng line number 1-based chính thức và ZIP pass
`unzip -t`. Không nộp artifact rule thô; chỉ nộp `submission.zip` ở trên.

Leaderboard xác nhận `ANSWER_ACCURACY=EXECUTION_ACCURACY=0.3083`, tăng bốn câu
đúng ròng trên 506 câu so với `0.3004`. `TABLES_F2MACRO=0.4902` và
`DOCS_F2MACRO=0.9079` giữ nguyên, nên gain đến hoàn toàn từ logic answer. Đây
là checkpoint chính thức mới và là base cho canonical note dictionary của 23
câu year-ranking còn lại.

## 13. Checkpoint canonical note v3 + year-ranking v2 exact13: 0.3202

Batch này không cần rerun Qwen. Canonical note dictionary v3 và direct
year-ranking resolver v2 đã xử lý 13/16 câu đơn giản còn lại bằng query
deterministic, rồi merge fill-only trên checkpoint `0.3083`.

File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03083_plus_note_v3_yearrank_v2_exact13_w010/submission.zip
```

SHA-256:

```text
a27e20365e3f4fa3a081014fe69e64eeb6b78dd4047863f119da64d3596f4a05
```

Allowlist gồm 13 ID:

```text
813,829,852,890,897,921,936,946,959,971,978,981,1008
```

Ba ID `907,910,986` vẫn fail closed vì thiếu ô dữ liệu năm hiện tại hoặc ô là
dấu `-`; không được thêm thủ công giá trị của năm trước. Submission mới chỉ
đổi đúng 13 ID trên, không ghi đè câu `ok` nào của checkpoint `0.3083`. Tất cả
1.012 query đã replay đúng answer, 229 test pass, line number bảng dùng chuẩn
1-based và ZIP pass `unzip -t`.

Leaderboard xác nhận:

```text
TABLES_F2MACRO      0.4913
DOCS_F2MACRO        0.9103
ANSWER_ACCURACY     0.3202
EXECUTION_ACCURACY  0.3202
```

So với `0.3083`, batch tăng sáu câu đúng trên tập chấm 506 câu, đồng thời tăng
table F2 `+0.0011` và docs F2 `+0.0024`. Đây là checkpoint chính thức mới và
là base bắt buộc cho mọi merge fill-only tiếp theo.

## 14. Checkpoint typed compositional ranking v3 exact12: 0.3300

Batch này không cần rerun Qwen. Typed planner tách rõ phép chọn theo công
thức/chỉ tiêu A và phép trả về công thức/chỉ tiêu B, sau đó solver deterministic
thực hiện `select-then-project` với exact metric, exact period và tie guard.

File cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010/submission.zip
```

SHA-256:

```text
4c674ca7e75db0e1b279ff2ea997799b7e7c1cdd96a0945fdcca6f308b92406d
```

Allowlist gồm 12 ID:

```text
415,440,459,475,479,491,505,507,543,556,558,575
```

Bốn ID `495,497,511,524` vẫn fail closed do thiếu exact row cho metric note,
EPS hoặc thuế; không nới fuzzy matching. Nhóm inventory-days dùng operand có
kiểu kỳ `inventory(t-1)`, `inventory(t)`, `COGS(t)` và đọc đầy đủ evidence của
mọi doanh nghiệp trước khi chọn công ty thắng. Column resolver cũng nhận năm bị
OCR tách thành hàng header riêng hoặc ngày bị ép thành số như `3092024`.

Submission chỉ đổi đúng 12 entry so với checkpoint chính thức `0.3202`, không
ghi đè câu `ok`. Tất cả 1.012 query compile và replay đúng answer, 240 test
pass, `relevant_tables` dùng line number 1-based và ZIP pass `unzip -t`.
Leaderboard xác nhận:

```text
TABLES_F2MACRO      0.4958
DOCS_F2MACRO        0.9139
ANSWER_ACCURACY     0.3300
EXECUTION_ACCURACY  0.3300
```

So với `0.3202`, batch tăng năm câu đúng ròng trên tập chấm 506 câu, table F2
tăng `0.0045` và docs F2 tăng `0.0036`. Đây là checkpoint chính thức mới và
là base bắt buộc cho median-filter planner v4.
