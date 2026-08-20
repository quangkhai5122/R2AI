# RUNBOOK — sổ tay lệnh ViFinQA

> **File này là nguồn sự thật duy nhất về LỆNH CHẠY.** Mỗi khi pipeline đổi,
> ghi đè trực tiếp vào đây (đừng tạo file mới) để không bị lạc phiên bản.
>
> Cập nhật lần cuối: **2026-08-20 (đã có leaderboard v5.3a/v5.3b)**.
> V5.3a: TABLES_F2 `.4453`, DOCS_F2 `.8975`, ANSWER/EXEC `.2490`.
> V5.3b: TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC `.2470`.
> So với frozen v5.2a `.2451`, hai ablation tương ứng khoảng +2 và +1 câu đúng ròng
> trên 506 câu chấm. Hai tập sửa rời nhau; candidate hợp nhất v5.3c là bước kế tiếp.
> Không chạy Kaggle: toàn bộ P2.4-silver/v5.3 là CPU local.

---

## 0. Trạng thái hiện tại (đọc trước khi chạy bất cứ thứ gì)

| Thành phần | Phiên bản/giá trị đang dùng |
|---|---|
| `vifinqa` package | 0.2.0 |
| Payload schema | **8** là provenance raw semantic-v5; staged payload stale so với source v5.3 và chỉ rebuild nếu có GPU run mới |
| `TABLE_POS_MODE` | `line` (BTC xác nhận: vị trí = **số dòng** của `<table>`) |
| `SUBMISSION_K` | 5 |
| Retrieval control chuẩn | `artifacts/retrieval.jsonl`, SHA-256 `96b71c5b…` |
| Control tốt nhất đã nộp | P2.2 v5.3a: TABLES_F2 .4453 / DOCS_F2 .8975 / ANSWER .2490 / EXEC .2490 |
| Leaderboard ablation mới nhất | v5.3b: TABLES_F2 .4443 / DOCS_F2 .8975 / ANSWER .2470 / EXEC .2470 |
| Thứ tự tiếp theo | tạo v5.3c union đúng 5 repair của v5.3a+v5.3b; không mở threshold trước khi nộp control này |
| Backend LLM Kaggle | `hf` (transformers). **vLLM không chạy trên T4** |
| Fuzzy scorer | `difflib.SequenceMatcher`, contract version `1` |

**Kiểm tra nhanh trạng thái trước mỗi phiên làm việc:**

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
# Dùng môi trường (base); không activate .venv.
python -c "import json;d=[json.loads(l) for l in open('artifacts/retrieval.jsonl',encoding='utf-8')];print('retrieval:',len(d),'| co plan P1:', 'plan' in d[0]['route'])"
python -m pytest -p no:cacheprovider --basetemp artifacts\pytest_tmp_runbook tests -q
```

`co plan P1: False` ⇒ retrieval là bản CŨ, phải chạy lại §2 trước khi làm gì khác.

---

## 0bis. ĐỔI GÌ THÌ PHẢI CHẠY LẠI GÌ (tra bảng này trước khi chạy)

| Sửa file ở... | 01 store | 02 retrieve | 03 rule | 04+upload payload | chạy Kaggle |
|---|:--:|:--:|:--:|:--:|:--:|
| `extraction/`, `utils/viet_num.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `router/`, `retrieval/{retrieve,bm25}.py` | – | ✅ | ✅ | ✅ | ✅ |
| `utils/viet_text.py` (chỉ scorer) | – | ✅ | ✅ | ✅ | ✅ |
| `retrieval/shortlist.py` | – | – | ✅ | ✅ | ✅ |
| `codegen/rule_*.py`, `units.py` | – | – | ✅ | ✅ | ✅ |
| `codegen/prompts.py`, `selection.py`, `generate.py`, `llm_client.py` | – | – | – | ✅ | ✅ |
| `submission/build.py`, `scripts/05` | – | – | – | – | – |
| `codegen/hybrid.py`, `scripts/11_merge_codegen_hybrid.py` | – | – | – | – | – |
| chỉ `tests/`, `*.md` | – | – | – | – | – |

Payload luôn phải dựng lại khi **bất kỳ** file nào trong `vifinqa/` hoặc
`kaggle/kaggle_codegen.py` đổi, vì manifest băm SHA-256 toàn bộ code.

### Riêng nhánh sau #17 — trạng thái hiện tại

- Giữ bất biến `artifacts/retrieval.jsonl` cho replay P2.1r và lần Qwen rescue đầu tiên.
- `artifacts/retrieval_rescue.jsonl` được dựng bằng router/retrieval hiện tại nhưng thay đổi
  cả route và thứ hạng table; đây là **ablation riêng**, không gọi là rescue-only.
- Mọi output mới dùng tên mới và run signature mới. Không resume trực tiếp artifact #17.
- Checkpoint Stage B dở được sinh từ payload schema 5. Source hiện hành là schema 6;
  phải tạo `p22b_oom_tail.json`, rebuild/upload rồi chỉ chạy tail bằng output mới. Không
  resume checkpoint schema 5 với code hiện tại.

```powershell
python -m pytest tests -q
python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --dataset-id <user>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip `
  -m "P2.2 schema6 Stage-B OOM tail"
```

Payload schema 5 ngày 2026-08-11 có **270 files** và stable manifest digest bắt đầu
`f174502ed8fe595e`; đây là provenance của checkpoint B đang dở. Recovery payload schema
6 chưa có hash cuối cho tới khi checkpoint được tải và tail mask được tạo. Khi chỉ thêm
`p22b_oom_tail.json`, manifest phải có 271 files. Retrieval vẫn là
`96b71c5b31a193dc…` và fuzzy scorer `difflib.SequenceMatcher/v1`.

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

#### Control bắt buộc cùng `k=4` với các lượt Qwen #12–#15

Control này không ghi đè baseline `k=6` và không cần GPU:

```powershell
python scripts/03_rule_baseline.py `
  --retrieval artifacts/retrieval.jsonl `
  --k 4 `
  --out artifacts/codegen_rule_k4_pre_factaware.jsonl

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_rule_k4_pre_factaware.jsonl `
  --out-dir artifacts/submission_rule_k4_pre_factaware `
  --sub-k 5
```

Artifact đã dựng ngày 2026-08-08 phải có đúng:
`rule=323`, `rule_composite=103`, `none=586`, tổng 1.012; build/replay phải thành công.
Đây là **pre-fact-aware control**. Không tái tạo file này sau khi đổi code rồi gọi nó là
cùng control.

Control CPU của code P2.1 với budget động (`--k 0`) dùng để tách phần rule/context khỏi
phần Qwen:

```powershell
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval.jsonl --k 0 --out artifacts/codegen_rule_dynamic_factaware.jsonl
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_rule_dynamic_factaware.jsonl --out-dir artifacts/submission_rule_dynamic_factaware --sub-k 5
```

Kết quả local đã kiểm chứng: `rule=324`, `rule_composite=129`, `none=559`; 1.012 query
eval-compilable, 1.116 CSV. Chỉ nộp control này nếu còn slot và cần attribution riêng.

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

### 2.5 Hybrid #15 (14B primary) + #14 (7B fallback) — không cần GPU

Policy chỉ thay một record 14B khi nó là placeholder cấu trúc đầy đủ:
`source=none`, `status=failed`, answer 0 và query hằng `0.0`. Một query hợp lệ tính ra
answer 0 **không** bị coi là rỗng. Script fail-fast nếu ID, question hoặc run signature
không nhất quán.

```powershell
python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/submission_sel14b/codegen_sel14b.jsonl `
  --fallback artifacts/submission_sel7b/codegen_sel7b.jsonl `
  --out artifacts/codegen_hybrid_sel14b_sel7b.jsonl `
  --audit artifacts/codegen_hybrid_sel14b_sel7b.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_hybrid_sel14b_sel7b.jsonl `
  --out-dir artifacts/submission_hybrid_sel14b_sel7b `
  --sub-k 5
```

Kết quả đã kiểm chứng trên artifact hiện có:

- giữ primary 14B: **702**;
- lấy fallback 7B: **35**;
- vẫn unresolved: **275**;
- hybrid signature bắt đầu bằng `43126dbdef3a2c51`;
- submission: 1.012 entries, 1.349 CSV, replay thành công.

Đây là artifact lịch sử của submission #16 (score không tăng so với #15), không nộp lại.
Không chạy hybrid output như checkpoint Kaggle; nó chỉ dùng để build submission.

### 2.6 P2.1r — replay Selection #17, không dùng GPU

Replay dùng chính `selection_trace` đã lưu trong #17, dựng lại shortlist trên retrieval
control và chạy compiler xác định. Policy mặc định chỉ được thay structural placeholder
`source=none/status=failed/answer=0/query=0.0`; 752 kết quả đã có của #17 không bị ghi đè.

```powershell
python scripts/12_replay_selection_p21r.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/submission_sel14b_factaware/codegen_sel14b_factaware.jsonl `
  --out artifacts/codegen_p21r_year_only_v3.jsonl `
  --audit artifacts/codegen_p21r_year_only_v3.audit.json `
  --k 0 --top-n 12 --replace-policy none_only --output-types year

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p21r_year_only_v3.jsonl `
  --out-dir artifacts/submission_p21r_year_only_v3 `
  --sub-k 5
```

Year-only v3 đã dựng ngày 2026-08-10: **752 kept / 207 skipped type / 25 replayed /
28 unresolved**, còn 235 structural-none; 1.012 expression compile/replay được, ZIP có
1.550 CSV. Run signature bắt đầu `43bd291ff6f8a58e`; SHA-256 codegen bắt đầu
`cec53f209909a806`; SHA-256 ZIP bắt đầu `6bceddd20709ae35`.

Đây là candidate CPU bảo thủ: answer `0` cũ không thể là một năm hợp lệ nên replay sai
không làm ANSWER/EXEC của 25 ID year thấp hơn placeholder, dù có thể vẫn chưa tăng và
evidence bổ sung có thể đổi TABLES/DOCS. Review cả 25 record ở trường `replayed_records`
trong audit trước khi nộp. Submission #18 đã xác nhận ANSWER/EXEC `.2253`.

Biến thể rộng hơn chỉ dùng sau review/P2.4 tune:

```powershell
python scripts/12_replay_selection_p21r.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/submission_sel14b_factaware/codegen_sel14b_factaware.jsonl `
  --out artifacts/codegen_p21r_all_v3.jsonl `
  --audit artifacts/codegen_p21r_all_v3.audit.json `
  --k 0 --top-n 12 --replace-policy none_only --output-types all
python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p21r_all_v3.jsonl `
  --out-dir artifacts/submission_p21r_all_v3 --sub-k 5
```

All-types v3: **752 kept / 40 replayed / 220 unresolved**; breakdown replay là
25 `year`, 5 `count`, 4 `number`, 3 `percent`, 3 `ratio`. Duplicate stable cells bị
fail-closed cho mọi operation. Review toàn bộ 5 `count` và 10 record số/percent/ratio,
vì các loại này có thể có gold answer bằng 0 hoặc predicate mà model chọn sai. ZIP có
1.569 CSV; run signature bắt đầu `36469da02106a4b7`, SHA-256 codegen bắt đầu
`24203de5782a5f14`, SHA-256 ZIP bắt đầu `727a1e29b2e2bb24`. Submission #19 đạt
ANSWER/EXEC `.2292` và hiện là control tốt nhất.

Hai thế hệ cũ `submission_p21r_none_only` / `codegen_sel14b_factaware_p21r.jsonl`
và `p21r_none_only_v2` / `codegen_p21r_none_only.jsonl` đều là
**pre-guard, DO NOT UPLOAD/USE**; bản 46-replay có 6 record chọn trùng cùng stable cell
qua index khác nhau. Không dùng
`--replace-policy trace_failures` cho submission chính; đó chỉ là ablation có thể ghi đè
đáp án đang thành công.

### 2.7 Rescue shortlist rỗng — audit CPU trước, Qwen sau

Rescue là opt-in và chỉ chạy khi strict shortlist không có candidate:

1. mở rộng table pool từ budget route lên tối đa 20, vẫn dùng strict scorer;
2. nếu vẫn rỗng, chấm 2D bằng `label + 0.9 × col_name`, threshold 28 sau bonus;
3. nếu vẫn rỗng, giữ `none`; không hạ threshold cho các prompt vốn đã có shortlist.

Audit không gọi LLM và không sửa artifact đầu vào:

```powershell
python scripts/13_audit_shortlist_rescue.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/submission_sel14b_factaware/codegen_sel14b_factaware.jsonl `
  --out artifacts/shortlist_rescue_audit.json `
  --k 0 --table-k 20 --min-score 28 --top-n 12
```

Kết quả trên đúng 142 trace `no_candidates` của #17: **24** câu được cứu chỉ nhờ
`widen_strict`, **83** câu nhờ `schema_2d`, **35** vẫn rỗng; tổng coverage shortlist
tiềm năng 107/142. Đây mới là coverage candidate, chưa phải 107 answer đúng.

Chỉ giữ cấu hình 20/28 nếu review P2.4 tune cho thấy precision không giảm. Không dùng số
candidate được cứu làm proxy cho ANSWER: coverage tăng nhưng row sai vẫn cho answer sai.

`artifacts/retrieval_rescue.jsonl` chứa cả fix entity alias và retrieval chạy lại. Muốn đo
nhánh đó, trước hết dựng **rule-only control riêng** rồi mới tốn GPU:

```powershell
python scripts/03_rule_baseline.py `
  --retrieval artifacts/retrieval_rescue.jsonl --k 0 `
  --out artifacts/codegen_rule_retrieval_rescue.jsonl
```

Kết quả control hiện tại: `rule=316 / rule_composite=110 / none=586`, chỉ **426** câu
có đáp án, thấp hơn schema-4 control frozen `324 / 128 / 560` = **452** câu. Có 30 câu old-ok→none
và chỉ 4 câu old-none→ok. Vì vậy **không dùng retrieval_rescue cho lượt GPU kế tiếp**;
giữ nó làm artifact chẩn đoán router/retrieval cho một ablation sau.

Không trộn output của retrieval này với `artifacts/retrieval.jsonl` khi build submission.

### 2.8 P2.4 — dev set người gán nhãn

Bundle cố định gồm tune=100 và locked=50, stratify theo operation × số fact. Fingerprint
hiện tại: `311f17edcc8540d52b407c7ab84637f3052108bcb997adaf0fcf8fc04cb436d1`.
Chi tiết evidence/AST/replay và protocol chống leakage ở `P2_4_LABELING_GUIDE.md`.

```powershell
python scripts/14_p24_devset.py validate-bundle
python -m pytest tests/test_p24_devset.py tests/test_p24_evaluate.py -q

$p24TuneDraft = "artifacts\devset_p24\p24_tune_gold.draft.jsonl"
$p24TuneGold = "artifacts\devset_p24\p24_tune_gold.jsonl"
if ((Test-Path -LiteralPath $p24TuneDraft) -or `
    (Test-Path -LiteralPath $p24TuneGold)) { throw "Tune working files already exist" }
Copy-Item -LiteralPath `
  "artifacts\devset_p24\p24_tune_gold.template.jsonl" `
  -Destination $p24TuneDraft

# Trong lúc gán nhãn: identity/schema; sau khi đủ 100 verified: strict cell/AST/replay.
python scripts/14_p24_devset.py validate-gold --split tune `
  --gold $p24TuneDraft --allow-template
python scripts/14_p24_devset.py fill-hashes --split tune `
  --input $p24TuneDraft --output $p24TuneGold
python scripts/14_p24_devset.py validate-gold --split tune `
  --gold $p24TuneGold
python scripts/14_p24_devset.py check-tune-input --input $p24TuneGold

# So sánh candidate bằng cùng gold; evaluator yêu cầu codegen đủ 1.012 ID.
python scripts/14_p24_devset.py evaluate --split tune `
  --gold $p24TuneGold `
  --codegen artifacts/submission_sel14b_factaware/codegen_sel14b_factaware.jsonl `
  --output artifacts/devset_p24/eval_p21_tune.json
python scripts/14_p24_devset.py evaluate --split tune `
  --gold $p24TuneGold `
  --codegen artifacts/codegen_p21r_year_only_v3.jsonl `
  --output artifacts/devset_p24/eval_p21r_tune.json
```

Không mở locked để chọn threshold/prompt. Freeze code, config và run signature trước;
sau đó strict-validate, seal, verify và chỉ đánh giá locked một lần. `fill-hashes` và
`evaluate` đều từ chối ghi đè output có sẵn; `build` và `seal-locked` cũng là one-shot,
không ghi đè bundle/seal đã khóa. Raw accuracy của 50
câu locked không phải ước lượng population không chệch do sampler giữ chỗ cho strata hiếm;
báo cả breakdown theo stratum và weighted aggregate. Tune hiện chỉ phủ 16/21 strata,
thiếu 5 strata = 12/1.012 câu (`represented_population_mass=0.988142`), nên weighted
tune là conditional trên strata được đại diện; luôn báo thêm `missing_strata`
và `complete_population_coverage`.

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

---

## 4. Kaggle — payload

### 4.1 Khi nào phải rebuild payload

Rebuild nếu **bất kỳ** thứ nào sau đây đổi: code trong `vifinqa/`,
`kaggle/kaggle_codegen.py`, `artifacts/store/`, `artifacts/retrieval.jsonl`.

### 4.2 Dựng payload

```powershell
python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --dataset-id <user1>/vifinqa-payload
python scripts/04_make_kaggle_payload.py --dry-run        # kiểm tra trước, không ghi đè
```

Ra `artifacts\kaggle_payload\` (~100 MB) + `payload-manifest.json`.

### 4.3 Up lên acc #1

```powershell
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip `
  -m "P2.1r frozen-scorer rescue-empty"
```

(Lần đầu tiên với một acc: đổi `version` → `create`.)

### 4.4 Up lên acc #2

```powershell
# một lần: tải kaggle.json của acc #2 vào D:\kaggle_acc2\kaggle.json
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --retrieval artifacts/retrieval.jsonl --dataset-id <user2>/vifinqa-payload   # PHẢI chạy lại, id nằm trong metadata
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

Notebook v1 rescue control: `kaggle/vifinqa-codegen.ipynb`. Lượt hiện hành P2.2 dùng
`kaggle/vifinqa-codegen-p22.ipynb` và chạy B rồi C theo §7quater.
Settings: **GPU T4 x2**, **Internet On**, Add Input = dataset payload (chỉ MỘT bản).

Log mới đúng phải có: `payload verified: schema=6` →
`fuzzy scorer: backend=difflib.SequenceMatcher version=1` → `run signature: ...` →
`baseline written (...)` → `LLM queue: ...` → `[chunk 1/N] ...`

Cấu hình đang khuyến nghị: **Qwen 14B fallback cho P2.1r với rescue strict-empty**.
Payload phải chứa retrieval control; `--llm-target empty` tiết kiệm GPU và hybrid local
giữ nguyên mọi output P2.1r đã thành công. `--k 0` là budget động, không phải 0 table.

```
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target empty \
    --out /kaggle/working/codegen_sel14b_rescue.jsonl \
    --n 1 --temperature 0.7 --k 0 \
    --rescue-no-candidates --rescue-table-k 20 --rescue-min-score 28 \
    --max-tokens 96 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400 --seed 13
```

Biến thể:

| Mục đích | Thêm/đổi cờ |
|---|---|
| Bật semantic matching | `--use-dense` (cần `pip install -q sentence-transformers` + `store/label_index/`) |
| Smoke bắt buộc | dùng notebook: output riêng `codegen_sel14b_rescue_smoke.jsonl`, `--llm-target all --limit 12 --no-resume` |
| Bỏ qua câu rule đã chắc | `--rule-first` |
| Chạy tiếp phiên trước | đặt checkpoint vào đúng `/kaggle/working/codegen_sel14b_rescue.jsonl`, giữ NGUYÊN mọi cờ |
| OOM | schema 6 tự hạ batch và tách `n=2` thành hai lượt `n=1`; Stage B dở phải dùng tail mask + output mới theo P2.2 runbook mục 9 |

Tải `codegen_sel14b_rescue.jsonl` từ tab Output → về local rồi hybrid, không build/nộp
fallback trực tiếp:

```powershell
python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p21r_year_only_v3.jsonl `
  --fallback <duong_dan>\codegen_sel14b_rescue.jsonl `
  --out artifacts/codegen_hybrid_p21r_rescue.jsonl `
  --audit artifacts/codegen_hybrid_p21r_rescue.audit.json
python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_hybrid_p21r_rescue.jsonl `
  --out-dir artifacts/submission_hybrid_p21r_rescue --sub-k 5
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
| 13 | Qwen 7B `--llm-target weak` | .4334 | .8774 | **.1443** | .1443 |
| 14 | Qwen 7B Selection structure | .4334 | .8774 | **.1838** | .1838 |
| 15 | Qwen 14B Selection structure | .4334 | .8774 | **.1957** | .1957 |
| 16 | Hybrid #15 + fallback #14 (35 câu) | .4334 | .8774 | **.1957** | .1957 |
| 16.1 | Rule dynamic fact-aware | .4352 | .8805 | **.1522** | .1522 |
| 16.2 | Rule k=4 pre-fact-aware | .4334 | .8774 | **.1482** | .1482 |
| 17 | P2.1 Qwen 14B fact-aware, dynamic k | .4406 | .8937 | **.2115** | .2115 |
| 18 | P2.1r year-only v3, 25 replay | .4426 | .8961 | **.2253** | .2253 |
| 19 | P2.1r all-types v3, 40 replay | **.4439** | **.8969** | **.2292** | **.2292** |
| 20 | P2.2 B+C semantic v5.1, 6 fill-only | **.4443** | **.8975** | **.2312** | **.2312** |
| 21 | P2.2 v5.2a lookup repair, 13 signed-silver | **.4443** | **.8975** | **.2451** | **.2451** |
| 22 | P2.2 v5.2b multi-operand signed-silver, 6 repair | **.4443** | **.8975** | **.2451** | **.2451** |

| 23 | P2.2 v5.3a single-cell consensus, 3 repair | **.4453** | **.8975** | **.2490** | **.2490** |
| 24 | P2.2 v5.3b structural-none lookup, 2 rescue | **.4443** | **.8975** | **.2470** | **.2470** |
Submission #20 tăng ròng `.0020 = 1/506` câu đúng so với #19. Submission #21 tăng tiếp
`.0139`, tương ứng khoảng **+7/506 câu đúng ròng**, trong khi toàn bộ retrieval metric giữ
nguyên. Kết quả này xác nhận hướng sửa semantic deterministic có giá trị; không chứng minh
mọi ID v5.2a đều đúng riêng lẻ. V5.2b mở rộng có kiểm soát sang đúng 6 multi-operand ID
nhưng leaderboard hoàn toàn không đổi; không tiếp tục mở rộng generic multi-operand ở
nhánh này.

Submission #23 tăng ANSWER/EXEC từ `.2451` lên `.2490`. Với 506 câu chấm và cách làm tròn
4 chữ số, kết quả khớp **126/506 so với 124/506**, tức khoảng **+2 câu đúng ròng** trong
3 ID sửa. TABLES_F2 tăng `.0010` nhờ recall tăng `.0019`; DOCS_F2 giữ nguyên. Submission
#24 tăng lên `.2470`, khớp **125/506**, tức khoảng **+1 câu đúng ròng** trong 2 ID rescue;
toàn bộ retrieval metric của #24 giữ nguyên so với v5.2a.

Vì #23 và #24 đều dùng cùng frozen v5.2a và thay hai tập ID rời nhau
`[245,329,730]` và `[158,213]`, phép cộng row-level dự báo union năm repair đạt khoảng
`127/506 = .2510`. Đây là dự báo, chưa phải score xác nhận; bước kế tiếp là build/nộp một
v5.3c union exact-allowlist trước khi sang fact-slot table reranker. P2.4-silver và hai
ablation runnable ở mục 10 bên dưới và mục 15 của
`RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md`.


**#18/#19 xác nhận P2.1r có giá trị thật trên leaderboard.** Với độ phân giải score phù
hợp public split 506 câu, #17 → #18 tương ứng khoảng **+7 câu đúng ròng**; #18 → #19
thêm khoảng **+2 câu**, tổng #17 → #19 khoảng **+9 câu**. Đây là suy luận từ score
aggregate, không tiết lộ ID nào đúng. All-types đồng thời tăng TABLES/DOCS, nên chưa thấy
trade-off evidence do 15 replay ngoài `year` gây ra.

| Metric đầy đủ | #18 year-only | #19 all-types | #19 − #18 |
|---|---:|---:|---:|
| TABLES_F2MACRO | .4426 | **.4439** | +.0013 |
| TABLES_PRECISION | .2759 | **.2764** | +.0005 |
| TABLES_RECALL | .6300 | **.6316** | +.0016 |
| TABLES_MRR5 | .5875 | .5875 | 0 |
| DOCS_F2MACRO | .8961 | **.8969** | +.0008 |
| DOCS_PRECISION | .9488 | .9488 | 0 |
| DOCS_RECALL | .8933 | **.8943** | +.0010 |
| DOCS_MRR5 | .9654 | .9654 | 0 |
| ANSWER_ACCURACY | .2253 | **.2292** | +.0039 |
| EXECUTION_ACCURACY | .2253 | **.2292** | +.0039 |

**#12: 183 final LLM / khoảng 166 placeholder được lấp → net chỉ ~+2 câu đúng.**
Tỷ lệ marginal thô khoảng 1.1–1.2%, không phải 2%. Nguyên nhân đã audit:
35% query không lọc cột năm, 15% quên chia ANSWER_SCALE, 90% thiếu `regex=False`.
Chi tiết + hướng sửa: `CLAUDE.md` mục "CHẨN ĐOÁN LƯỢT QWEN #12".

**#11 trung tính** (−0.002 = 1 câu/506): bỏ 24 đáp án nhưng trong đó chỉ ~1 câu
đúng. Eval dự báo lookup +0.10 nhưng thực tế ≈ 0 ⇒ **lần thứ hai eval dự báo sai**.
Rule đã tới điểm lợi tức giảm dần: P1.5 +0.026, P1.6 −0.002.

## 7bis. P2.1 — cấu hình đã tạo submission #17 (lịch sử tái lập)

Phần này giữ nguyên để tái lập #17; **không phải lệnh khuyến nghị cho lượt chạy tiếp**.
#17 đã dùng budget động, shortlist theo F-slot, ticker/report/year và rejection taxonomy.

| Cờ | #15 control | P2.1 mới |
|---|---|---|
| model | Qwen2.5-Coder-14B-Instruct NF4 | giữ nguyên |
| mode/target | `select` / `all` | giữ nguyên |
| n / temperature / seed | 1 / 0.7 / 13 | giữ nguyên |
| table context | fixed `--k 4` | **`--k 0` = route budget 4–12** |
| shortlist | global, một cột/label | fact-aware, multi-year, có provenance |
| output | `codegen_sel14b.jsonl` | `codegen_sel14b_factaware.jsonl` |

### Bước A — smoke bắt buộc

```text
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel14b_factaware_smoke.jsonl --limit 12 \
    --n 1 --temperature 0.7 --k 0 --max-tokens 96 --batch-size 4 \
    --checkpoint-every 4 --time-budget-min 30 --seed 13 --no-resume
```

Log phải có `payload verified`, `run signature`, `baseline written`, `LLM queue` và
`[chunk ...]`. QA smoke phải có 12 ID duy nhất, finite answer và `selection_trace`.

### Bước B — full run 14B

```text
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel14b_factaware.jsonl \
    --n 1 --temperature 0.7 --k 0 --max-tokens 96 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400 --seed 13
```

Không đổi tên output/cờ khi resume. Không copy artifact #15 cũ vào output này vì signature
và prompt khác. Nếu OOM, runner tự hạ batch; mọi thay đổi tay về batch/k/token phải dùng
output mới vì code hiện tại fingerprint các cờ này.

### Bước C — QA artifact sau khi download

```powershell
python -c "import json,math;from collections import Counter;p=r'<path>\codegen_sel14b_factaware.jsonl';r=[json.loads(x) for x in open(p,encoding='utf-8')];ids=[x['id'] for x in r];print('rows/unique',len(r),len(set(ids)));print('source',Counter(x['source'] for x in r));print('outcome',Counter((x.get('selection_trace') or {}).get('outcome','not_run') for x in r));print('reject',Counter(k for x in r for k,n in ((x.get('selection_trace') or {}).get('rejection_counts') or {}).items() for _ in range(n)));assert len(r)==1012 and len(set(ids))==1012 and all(math.isfinite(float(x['answer'])) for x in r)"

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen <path>\codegen_sel14b_factaware.jsonl `
  --out-dir artifacts/submission_sel14b_factaware `
  --sub-k 5
```

Nộp `artifacts\submission_sel14b_factaware\submission.zip`. Sau khi có điểm, thêm một
dòng mới vào bảng §7; không ghi đè điểm #15 hoặc artifact hybrid.

`selection_trace` schema v1 lưu raw response đã sanitize/cắt tối đa 2.000 ký tự kèm
SHA-256. Các reason chính: `parse_error`, `model_none`, `invalid_selection`,
`synthesis_error`, `execution_failed`, `semantic_validation_failed`, `answer_mismatch`.
Trace chỉ là metadata audit, không làm thay đổi answer/query/arbitration.

## 7ter. Lệnh runnable hiện có — frozen #19 → rescue-v1 control → hybrid

### Bước 1 — frozen control zero-GPU

Giữ `artifacts/codegen_p21r_all_v3.jsonl` và submission #19 làm control bất biến:
`752 kept / 40 replayed / 220 unresolved`, ANSWER/EXEC `.2292`. Year-only #18 chỉ còn là
ablation bảo thủ để phân rã 25 year so với 15 replay ngoài year.

### Bước 2 — P2.4 nếu có thời gian, không còn block implementation

Làm theo §2.8 và `P2_4_LABELING_GUIDE.md`. Không dùng locked để chọn
`rescue-table-k`, `rescue-min-score`, prompt hoặc checkpoint. Nếu chưa gán đủ tune, có
thể chạy smoke/control và triển khai V2 theo §7quater, nhưng không tune threshold/policy
theo public leaderboard nhiều lần.

### Bước 3 — dựng/upload payload control

```powershell
python scripts/04_make_kaggle_payload.py `
  --retrieval artifacts/retrieval.jsonl `
  --target-dir artifacts/p22_targets `
  --dataset-id lequangkhai5122005/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip `
  -m "P2.2 schema6 typed IR masks B-C"
```

Manifest phải là schema 6, fuzzy scorer `difflib.SequenceMatcher` version 1. Import
`kaggle/vifinqa-codegen.ipynb`; smoke dùng `all` trên 12 câu để test đường LLM, còn full
run dùng đúng:

```text
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target empty \
    --out /kaggle/working/codegen_sel14b_rescue.jsonl \
    --n 1 --temperature 0.7 --k 0 \
    --rescue-no-candidates --rescue-table-k 20 --rescue-min-score 28 \
    --max-tokens 96 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400 --seed 13
```

Không dùng smoke làm checkpoint cho full vì target khác nên run signature khác. Resume
full chỉ dùng đúng output full và giữ nguyên cả `batch-size`, `checkpoint-every`, `limit`;
ba cờ này cũng được fingerprint vì chúng thay đổi thứ tự RNG khi sampling.

### Bước 4 — hybrid bảo thủ và build submission

```powershell
python scripts/11_merge_codegen_hybrid.py `
  --primary artifacts/codegen_p21r_all_v3.jsonl `
  --fallback <path>\codegen_sel14b_rescue.jsonl `
  --out artifacts/codegen_hybrid_p21r_rescue.jsonl `
  --audit artifacts/codegen_hybrid_p21r_rescue.audit.json

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_hybrid_p21r_rescue.jsonl `
  --out-dir artifacts/submission_hybrid_p21r_rescue --sub-k 5
```

QA bắt buộc: 1.012 ID duy nhất, finite answer, mọi expression compile/replay, audit hybrid
chỉ thay structural-none, và retrieval hash trùng control. Lượt này là **Selection v1
rescue control**; Structured Selection v2 được phát triển/đo riêng theo §7quater.

## 7quater. Lịch sử P2.2 grounded schema 7 — RETIRED

> Phần 7quater dưới đây chỉ là provenance schema 7. Không dùng các câu “hiện hành”
> trong lịch sử này; quy trình hiện hành bắt đầu ở 7quinquies.
Kết quả #18/#19 đã thỏa evidence gate. P2.2 hiện có atomic metric-slot planner, named
fact/binding, typed nested IR và grounded compiler bắt đủ F-slot/ticker/year. Payload
hiện hành là schema 7. Quy trình tail schema 6 đã retire; lệnh duy nhất được dùng nằm ở
mục 10 của `RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md`.

Lý do:

- flat typed replay #17 → #19 tăng khoảng 9/506 câu đúng và không làm metric nào giảm;
- 15 replay ngoài `year` của #19 tăng thêm khoảng 2 câu so với #18;
- #19 còn đúng **220 structural-none**: 142 `no_candidates` và 78 `rejected` đã có
  candidate. Trong 78 rejected có 55 output non-year phù hợp để thử nested IR; 23 year
  còn lại chủ yếu là period/alias linking, không nên ép qua AST;
- rescue tìm candidate cho 107/142 `no_candidates`, nhưng chỉ 48/142 phủ đủ mọi fact
  slot hiện tại; 59 câu còn thiếu slot và 35 câu vẫn rỗng.

Phạm vi P2.2 đã triển khai:

1. **Atomic metric-slot planner:** tách role `filter`, `rank`, `project`, `numerator`,
   `denominator`, `base`, `end` theo ticker/year trước khi shortlist. Hiện toàn bộ 1.037
   fact record của 220 câu đều chỉ có role `value`, nên generic AST một mình chưa đủ.
2. **P2.2b atomic bindings:** model bind candidate ổn định vào fact có tên/kiểu; leaf
   chứa candidate index/constant ID và provenance, không chứa pandas.
3. **P2.2c nested tree:** compiler xác định cho `filter/compare`, arithmetic/ratio/growth,
   `argmax/argmin` + projection, `count/sum/average`; giới hạn depth/node và exact arity.
4. Mọi node phải qua type, unit, entity/year alignment, stable-cell uniqueness, evidence
   grounding, finite-answer và execution replay. Fail thì giữ nguyên #19.

Thiết kế ablation bắt buộc:

- **A — frozen control:** #19 all-types v3;
- **A1 — v1 rescue control:** chạy cấu hình Selection v1 đã có, lưu trace theo đúng nhóm
  failure; không trộn score này với V2;
- **B — V2 rejected non-year:** chỉ cho phép thay 55 structural-none có candidate và
  được thay bằng B-groundable 15 atomic-complete, không truncated;
- **C — grounded rescue:** sau audit B mới mở 31/48 câu rescue atomic-complete.

Không chạy V2 trên toàn bộ 1.012 câu và không ghi đè rule/Selection đang thành công.
Checkpoint schema 5 đã được CPU replay: 48 attempted, chỉ ID 855 còn accepted; 9/9
accepted sai đã biết đều bị reject. Không chạy tail 7 câu. Lượt tiếp theo là schema 7,
B-groundable 15 câu trước.

### Lệnh triển khai P2.2

Dùng nguyên block ở **mục 10** của `RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md`.
Notebook hiện hành phải in schema 7 và `LLM queue: 15`. Run All tự dừng trước C bằng
`APPROVE_STAGE_C=False`. Sau khi tải B, bắt buộc chạy
`48_audit → 50_replay → 48_audit → hybrid → build` rồi gửi artifact về review.

Toàn bộ command Local/Kaggle, resume/OOM contract, hash đầy đủ, audit C và build BC nằm
trong `RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md`. Nhật ký thay đổi ở
`P2_2_STRUCTURED_SELECTION_V2_IMPLEMENTATION.md`. Chỉ dùng leaderboard cho ablation đã
freeze; không lặp threshold/prompt theo public score.

## 7quinquies. P2.2 semantic-grounded v5 — HISTORICAL GPU STAGES

Schema 8 thay thế schema 7. Planner mở slot `(entity, period, metric, role)` và chỉ
đưa candidate vào prompt khi label/code/column/date chứng minh trực tiếp slot đó. Route
phải qua cả entity guard lẫn planner guard; candidate chỉ khớp nhờ context, sai năm,
sai VAS code, sai entity hoặc sai metric sẽ bị loại fail-closed.

Các thay đổi vận hành:

- prompt JSON rút gọn, fact chỉ dùng tên `F1..Fn`, giảm lặp envelope;
- trace ghi `finish_reason`, token count và `hit_max_tokens`; response chạm giới hạn
  được phân loại `generation_truncated`, không bị nhầm với parse error;
- compiler hỗ trợ typed money literal (`nghìn/tỷ/triệu đồng`) nhưng không cho number
  literal giả làm money;
- replay mặc định yêu cầu checkpoint và mask trùng exact; tùy chọn
  `--allow-checkpoint-superset` chỉ dùng cho audit checkpoint cũ, tuyệt đối không tái
  sử dụng response ngoài mask;
- mask hiện hành: B `p22b_semantic_groundable_v5.json` có IDs `[855,966]`; C
  `p22c_semantic_groundable_v5.json` có IDs `[102,183,355,591]`.

Gate đã đo ngày 2026-08-14:

- focused P2.2: **80 passed**; full suite: **276 passed**; compileall pass;
- payload schema 8: 288 manifest files, source/runtime **62/62 exact**;
- stable manifest SHA-256
  `b61cf8206c5802863ba36d0c7e41976d81ce2e97c083de49f5150d27b221dc67`;
- raw manifest SHA-256
  `0b31d13d6e120c0819a70cf678e52c68b3ed2948587ce8d7c9fece6cdfd56f50`;
- B mask SHA-256
  `a12ea224b2a38f19f768e0d81be27f73f2a9281c26b72b1fe2c378ad2f12bf60`;
- C mask SHA-256
  `32d8e21de24b8613cec04ec9903b0dd5248dd77f8e2db939e36ba94e89abda5d`.

Checkpoint schema-7 `codegen_p22b_groundable_sel14b.jsonl` được replay dưới guard mới
chỉ trên giao của mask: target 2, attempted 2, **accepted 0**, rejected 2; 13 response
ngoài mask bị bỏ. Artifact này là đối chứng, không được hybrid/nộp.

Lệnh Local/Kaggle authoritative nằm ở **mục 11** của
`RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md`. Notebook phải in `schema=8` và B
`LLM queue: 2`; nếu in 15/31 hoặc schema 7 thì dừng ngay.

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

## 10. P2.4-silver tự động và hai submission v5.3 (CURRENT)

### 10.1 Nguyên tắc và control

- Môi trường local là `(base)`, không activate `.venv`; mọi lệnh dưới đây CPU-only.
- Frozen primary cho **cả hai** ablation là
  `artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl`, SHA-256
  `e339da82b8a49a3160427946d1f05ba59269c6f730e2ec4bf5d4e22864351ab4`,
  run signature `dc34176abba043ff3a0b42f1e8c5861067c82ba165bf36c29f0a641eb33b69d0`.
- Frozen retrieval SHA-256:
  `96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a`.
- Không xếp v5.3b lên v5.3a hoặc v5.2b. Hai ZIP phải độc lập để leaderboard attribution
  chỉ gồm đúng 3 consensus repairs hoặc 2 structural-none rescues.

### 10.2 P2.4-silver tự động

Bundle canonical lấy một fact signed-value đại diện trên mỗi cặp report liền kề và tách
ticker 70/15/15; expected value không được truyền vào resolver. Build/output là
exclusive-create, không chạy lại lên cùng đường dẫn:

```powershell
python scripts/54_p24_auto_silver.py build `
  --store-dir artifacts/store `
  --out-dir artifacts/p24_silver_auto_v53 `
  --seed 2453 --max-per-report-pair 1 --max-tickers-per-split 8

python scripts/54_p24_auto_silver.py evaluate `
  --split artifacts/p24_silver_auto_v53/p24_silver_tune.jsonl `
  --store-dir artifacts/store `
  --out artifacts/p24_silver_auto_v53/eval_tune_v53.json `
  --expect-split-sha256 8218ee4cda90f026f8a7854cf1ba441165fed5ae4d41879a9b69b5dff6df1ec4

# Chỉ mở locked sau khi tune accepted_answer_precision >= .95 và code/threshold freeze.
python scripts/54_p24_auto_silver.py evaluate `
  --split artifacts/p24_silver_auto_v53/p24_silver_locked.jsonl `
  --store-dir artifacts/store `
  --out artifacts/p24_silver_auto_v53/eval_locked_v53.json `
  --expect-split-sha256 98c489ce5779c42b5b290949522367fa2f377d65585e777e5ad25f44672006cb
```

Gate đã đo:

- fingerprint bundle `15be1d901009ee769883552f4e4132af2d4c13da55dcc8b2e6610715923eabb5`;
- 377 facts / 377 report-pairs: train 118, tune 123, locked 136; ticker disjoint;
- tune: coverage/cell/answer `.8617886`, accepted-answer precision `1.0`;
- locked: coverage/cell/answer `.9191176`, accepted-answer precision `1.0`;
- `artifacts/p24_silver_auto/` là stress bundle 4.482 facts chưa evaluate hoàn tất, không
  phải canonical gate và không được trộn metric với bundle report-diverse ở trên.

P2.4-silver chỉ kiểm cell/period/unit cho lookup một ô. Nó không phải gold của 1.012 câu
và không ước lượng trực tiếp leaderboard accuracy.

### 10.3 V5.3a — single-cell consensus repair (submission thứ nhất)

Preflight phải trả đúng `[245,329,730]`; ID 91 là false positive đã bị metric-token gate
loại vì `nguyên giá` không đồng nhất với `giá vốn`.

```powershell
python scripts/55_build_v53a_single_cell_consensus.py `
  --primary artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl `
  --retrieval artifacts/retrieval.jsonl --store-dir artifacts/store --preflight

python scripts/55_build_v53a_single_cell_consensus.py `
  --primary artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl `
  --retrieval artifacts/retrieval.jsonl --store-dir artifacts/store `
  --out artifacts/codegen_p22bc_semantic_v53a_overlay.jsonl `
  --audit artifacts/codegen_p22bc_semantic_v53a_overlay.audit.json `
  --expect-selected-ids 245,329,730 `
  --expect-primary-signature dc34176abba043ff3a0b42f1e8c5861067c82ba165bf36c29f0a641eb33b69d0 `
  --expect-primary-sha256 e339da82b8a49a3160427946d1f05ba59269c6f730e2ec4bf5d4e22864351ab4 `
  --expect-retrieval-sha256 96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22bc_semantic_v53a_overlay.jsonl `
  --store-dir artifacts/store `
  --out-dir artifacts/submission_p22bc_semantic_v53a --sub-k 5 --pos-mode line
```

Artifact đã khóa:

- codegen SHA-256 `77906d6c4dfd3adf88e7d882d45f34d6a2040da934f275d4b3ae3c2c5c44cee1`;
- run signature `439782fd55542e2269a4415a2a6c970accffd1f3a06bebaee5c0742adbe9c5b7`;
- ZIP SHA-256 `c538f805411a7cc540f3e38d3712b8cdb0c83d315748e583f7a37690de953e88`.

### 10.4 V5.3b — structural-none lookup rescue (submission thứ hai)

Universe là đúng 30 route-lookup structural-none. Chỉ ID 158 và 213 qua guard; 17/30
thực chất đa-fact, 10 thiếu evidence đủ mạnh, 1 thiếu exact report. Không ép đủ 30.

```powershell
python scripts/56_build_v53b_lookup_rescue.py `
  --primary artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl `
  --retrieval artifacts/retrieval.jsonl --store-dir artifacts/store --preflight

python scripts/56_build_v53b_lookup_rescue.py `
  --primary artifacts/codegen_p22bc_semantic_v52a_overlay.jsonl `
  --retrieval artifacts/retrieval.jsonl --store-dir artifacts/store `
  --out artifacts/codegen_p22bc_semantic_v53b_lookup_rescue.jsonl `
  --audit artifacts/codegen_p22bc_semantic_v53b_lookup_rescue.audit.json `
  --expect-selected-ids 158,213 `
  --expect-target-ids 37,125,128,156,158,213,233,285,336,362,365,421,424,427,430,431,435,456,583,599,625,641,657,663,750,783,863,887,909,924 `
  --expect-primary-signature dc34176abba043ff3a0b42f1e8c5861067c82ba165bf36c29f0a641eb33b69d0 `
  --expect-primary-sha256 e339da82b8a49a3160427946d1f05ba59269c6f730e2ec4bf5d4e22864351ab4 `
  --expect-retrieval-sha256 96b71c5b31a193dcad969de6b1e5ac64ff38c36bfcd44c15e491c240f09d685a

python scripts/05_build_submission.py `
  --retrieval artifacts/retrieval.jsonl `
  --codegen artifacts/codegen_p22bc_semantic_v53b_lookup_rescue.jsonl `
  --store-dir artifacts/store `
  --out-dir artifacts/submission_p22bc_semantic_v53b_lookup_rescue `
  --sub-k 5 --pos-mode line
```

Artifact đã khóa:

- codegen SHA-256 `18a8adebd87c8e5b947f198cf27073aa05f5ee4cc36d3c02b37a7b29c872cf00`;
- run signature `4e794f7e4accae4e48ce8ff44e7ec5abdc700b2f1df00e748166bb58228c85e3`;
- ZIP SHA-256 `dbca0c56f825ed4806389f19326d7e986f0d88ffb6e3e8e7c2efc875e2317cc8`.

### 10.5 Gate cuối và thứ tự nộp

- Full suite: `301 passed`; focused P2.4/v5.3/v5.2: `18 passed`.
- Mỗi codegen có 1.012 ID unique/finite, đúng một run signature; v5.3a chỉ đổi
  `[245,329,730]`, v5.3b chỉ đổi `[158,213]`.
- Mỗi ZIP có một `results.json` + 1.575 CSV; mọi query compile/replay; question/ID khớp
  frozen retrieval. Structural-none: v5.3a giữ 214, v5.3b giảm 214 → 212.
- Trình tự frozen đã hoàn tất đúng thiết kế, không tune giữa hai lượt.
- V5.3a đạt TABLES_F2 `.4453`, DOCS_F2 `.8975`, ANSWER/EXEC `.2490`: xấp xỉ +2/506
  câu đúng ròng; ba repair không thể được định danh đúng/sai riêng lẻ từ score tổng.
- V5.3b đạt TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC `.2470`: xấp xỉ +1/506
  câu đúng ròng; score tổng không cho biết ID 158 hay 213 là câu được cứu.
- Vì hai tập ID rời nhau trên cùng primary, union exact năm repair có expected
  ANSWER/EXEC `127/506 = .2510`. Đây là suy luận cộng tính theo row, cần leaderboard xác nhận.
- Bước kế tiếp: tạo v5.3c union, audit no-drift và nộp một lần. Chỉ sau control đó mới mở
  fact-slot table reranker; không thay threshold/allowlist của hai repair đã xác nhận.
