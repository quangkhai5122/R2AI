# ViFinQA Competition — Project Memory

Cuộc thi: Financial Table Retrieval & Text-to-Pandas trên BCTC 100 công ty niêm yết VN
(1.973 báo cáo OCR, 2015–2025). Chiến lược gốc: `ViFinQA_Claude_Strategy.md`.
Luật thi: `instructions/*.md`.

**`RUNBOOK.md` là nguồn sự thật duy nhất về lệnh chạy** — mọi thay đổi pipeline
phải ghi đè vào đó (không tạo file hướng dẫn mới). `README.md` chỉ là tổng quan.
Thay đổi mới nhất: `P2_2_STRUCTURED_SELECTION_V2_IMPLEMENTATION.md`; command runnable ở
`RUNBOOK_P2_2_STRUCTURED_SELECTION_V2.md` và `kaggle/vifinqa-codegen-p22.ipynb`.
## SNAPSHOT HIỆN TẠI — 2026-08-20

- Best submitted hiện là v5.3a: TABLES_F2 `.4453`, DOCS_F2 `.8975`,
  ANSWER/EXEC `.2490`, tương ứng khoảng +2/506 câu đúng ròng so với v5.2a.
- V5.3b đạt TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC `.2470`, tương ứng
  khoảng +1/506; retrieval metric giữ nguyên so với v5.2a.
- Hai tập ID rời nhau trên cùng frozen primary; union exact năm repair dự báo `.2510`.
- P2.4-silver tự động đã được dựng từ các ô lặp ở hai báo cáo kề nhau, tách ticker
  train/tune/locked. Bundle canonical có fingerprint
  `15be1d901009ee769883552f4e4132af2d4c13da55dcc8b2e6610715923eabb5`;
  tune 123 fact đạt coverage/accuracy `.8618`, locked 136 fact đạt `.9191`, accepted
  precision `1.0`. Đây là verifier silver cho cell/period/unit, không phải gold ẩn.
- V5.3a single-cell consensus sửa đúng ba status-ok lookup ID `[245,329,730]`.
  Codegen SHA `77906d6c4dfd3adf88e7d882d45f34d6a2040da934f275d4b3ae3c2c5c44cee1`,
  run signature `439782fd55542e2269a4415a2a6c970accffd1f3a06bebaee5c0742adbe9c5b7`,
  submission ZIP SHA `c538f805411a7cc540f3e38d3712b8cdb0c83d315748e583f7a37690de953e88`.
- V5.3b khóa target ở đúng 30 structural-none lookup nhưng chỉ hai ID `[158,213]`
  qua guard; 214 structural-none giảm còn 212. Codegen SHA
  `18a8adebd87c8e5b947f198cf27073aa05f5ee4cc36d3c02b37a7b29c872cf00`,
  run signature `4e794f7e4accae4e48ce8ff44e7ec5abdc700b2f1df00e748166bb58228c85e3`,
  submission ZIP SHA `dbca0c56f825ed4806389f19326d7e986f0d88ffb6e3e8e7c2efc875e2317cc8`.
- Cả hai candidate có 1,012 record, một signature, finite answers, compile/replay đủ;
  mỗi ZIP có 1 `results.json` + 1,575 CSV. Full suite cuối: **301 passed**.
- Thứ tự ablation đã hoàn tất không tune giữa hai lượt. Bước kế tiếp là v5.3c union
  exact IDs `[158,213,245,329,730]`; chỉ sau đó mới mở fact-slot table reranker.
  Lệnh canonical và leaderboard history nằm trong `RUNBOOK.md`.

## SNAPSHOT LỊCH SỬ — 2026-08-16 (v5.2b)

- Best submitted v5.2a: TABLES_F2 `.4443`, DOCS_F2 `.8975`, ANSWER/EXEC `.2451`;
  tăng `.0139`, khoảng +7/506 câu đúng ròng so với v5.1; còn 214 structural-none.
- V5.2b CPU overlay chỉ sửa 6 multi-operand ID có signed silver support độc lập:
  `[605,718,721,771,827,927]`. Ops được khóa ở difference/growth_pct/ratio/average;
  exact fact-leaf assignment, target period, metric, unit và stable cell phải qua guard.
- Answer/support-count theo operand: `-20.73/4,3`, `-39.92/1,2`, `1.40/2,3`,
  `30436754/1,1`, `602.38/1,1,2,1`, `37.25/2,1,1,2,2`.
- IDs 665/667/762 bị loại fail-closed vì ít nhất một operand thiếu signed support.
- Codegen candidate: `artifacts/codegen_p22bc_semantic_v52b_overlay.jsonl`,
  SHA-256 `51287d094488edac7b376bf6648dac289218fd1ffd69d28f9e097a1290580f4b`;
  run signature `98a638a1d0b5b58f763195578799fee2199e5c54149c28c740a21353faff242f`.
- File cần nộp: `artifacts/submission_p22bc_semantic_v52b/submission.zip`,
  SHA-256 `90a766fa5860d6efc1a07afaf5967de4ca28a2ec963d993e4b018265b5401209`.
- Audit: 1,012 records; đúng 6 thay đổi, 1,006 semantic-unchanged, 214 structural-none;
  ZIP có 1 results + 1,575 CSV; full suite 294/294.
- Payload schema 8 là provenance raw run và stale so với source v5.2b; không rebuild
  vì lượt này không cần GPU. Tiếp theo chỉ nộp candidate v5.2b một lần và báo đủ score.


## SNAPSHOT LỊCH SỬ — 2026-08-13 (schema 5/6, RETIRED)

- Frozen control #19: TABLES_F2 `.4439`, DOCS_F2 `.8969`, ANSWER/EXEC `.2292`;
  `artifacts/codegen_p21r_all_v3.jsonl`, 220 structural-none.
- Structured Selection v2 đã triển khai: atomic metric slots, named facts/bindings,
  typed nested IR, deterministic compiler/semantic replay và trace schema 2.
- Checkpoint Stage B schema 5 SHA `4f6c0bee...` đã audit: 1.012 rows, target 55,
  attempted 48, pending 7, accepted 10, rejected 38. Tail mask đã freeze IDs
  `909,926,939,966,992,1001,1012` (SHA `2ab36ca7...`).
- Semantic review dựng lại shortlist cho 10 accepted: chỉ ID 855 plausible (chưa gold);
  9 output còn lại sai/thiếu metric, entity, year hoặc tái dùng cùng ref nhiều năm.
- **HOLD:** chưa chạy tail/C, chưa hybrid. Bước kế tiếp là thêm guard provenance/coverage
  deterministic và replay raw responses trong checkpoint trước khi dùng thêm GPU.
- Schema 6 thêm fallback khi `n=2` OOM ở batch 1: sinh tuần tự `n=1+n=1`. Mọi tail dùng
  output/signature mới; không resume checkpoint schema 5 bằng runtime mới.
- Payload schema 5/stable manifest `f174502ed8fe595e...` là provenance của checkpoint
  gốc. Recovery payload schema 6 chỉ đóng sau khi tail mask được tạo/fingerprint.
- Gate local: **243 tests pass**. P2.2 chưa có leaderboard score.

## SỰ THẬT ĐÃ XÁC NHẬN (không cần kiểm chứng lại)

1. **Vị trí bảng trong `relevant_tables` = SỐ DÒNG (1-based) nơi `<table>` bắt đầu
   trong file OCR .txt** — BTC xác nhận trực tiếp qua trao đổi với đội, và đã
   verify trên leaderboard (TABLES_F2: 0.0 với thứ-tự-bảng → 0.364 với line-number).
   Code: store có cột `line_no`; nội bộ vẫn key bằng `table_pos` (thứ tự 0-based);
   map sang line khi build submission (`config.TABLE_POS_MODE = "line"` là mặc định).
2. **`report_id` format đúng** = tên thư mục báo cáo, vd
   `VNM_financial_statements_2023_consolidated` (DOCS_F2 0.84, MRR5 0.95).
3. **Test set = 506 câu** trong số 1.012 câu public (HF). Nộp đủ 1.012 vẫn được
   chấm bình thường; warning `gold=506 pred=1012` vô hại. BTC KHÔNG phát dữ liệu
   riêng — bản HuggingFace trong `data/ViFinQA/` là nguồn chuẩn.
4. **Suy ra từ leaderboard:** gold trung bình ~2.6 bảng/câu, ~1.15 doc/câu, chủ
   yếu trong cùng 1 báo cáo.
5. **`--expand-docs` ĐÃ THỬ VÀ THẤT BẠI** (DOCS_F2 0.84→0.61; precision sập
   0.91→0.28 đổi lấy +5.5% recall). Đừng thử lại.
6. **`pandas_query` PHẢI là MỘT BIỂU THỨC (grader dùng `eval`, không phải `exec`).**
   Bằng chứng leaderboard: #5 toàn expression → EXEC == ANSWER == 0.085. #6 có
   233/1.012 query là script nhiều dòng → mọi script là `SyntaxError` khi eval →
   bị tính crash: EXEC tụt 0.085→0.0613 DÙ ANSWER tăng 0.085→0.1047. Code đã
   chống tái phát ở 3 tầng: prompt yêu cầu 1 dòng; `codegen/to_expression.py`
   inline script thẳng thành 1 biểu thức (AST, có verify giá trị) ngay trong
   `_final()`; `submission/build.py` in cảnh báo nếu còn query không eval được.
   Sửa submission cũ không cần GPU: `scripts/08_repair_expressions.py`.
7. **vLLM KHÔNG CHẠY ĐƯỢC trên Kaggle T4** (Turing/SM75). Đã thử 2 lần:
   `swap_space` bị bỏ khỏi EngineArgs (đã fix bằng retry-loop drop kwarg), rồi
   `RuntimeError: Engine core initialization failed` — V1 engine không hỗ trợ
   SM75, mà V0 engine (chạy được T4) đã bị xoá khỏi vLLM nên `VLLM_USE_V1=0`
   vô hiệu. **Giải pháp: `HfBatchClient` (transformers batched generate),
   `--backend hf` — mặc định của notebook.** Chỉ thử lại vLLM khi pin
   `vllm==0.7.3` và còn dư thời gian.
8. **`SUBMISSION_K=5` ĐÃ ĐƯỢC LEADERBOARD KIỂM CHỨNG** so với k=10:
   TABLES_F2 0.3641→0.4092 (+0.0451; +12,4% tương đối), precision
   ~0.176→0.2621, recall ~0.684→0.5852 và MRR5 gần như không đổi
   (0.588→0.5882). DOCS_F2 giảm nhẹ 0.8399→0.8093. Kết quả này ủng hộ
   k=5 thay k=10, nhưng CHƯA chứng minh k=5 tối ưu so với k=7. Không suy
   F2 macro bằng cách cắm macro precision/recall vào công thức F2.

## LỊCH SỬ NỘP BÀI

| # | Cấu hình | TABLES_F2 | DOCS_F2 | ANSWER | Ghi chú |
|---|---|---|---|---|---|
| 1 | rule-only, pos=order base0, k=10 | 0.0 | 0.8399 | 0.085 | |
| 2 | như #1, base1 | 0.0 | 0.8399 | 0.085 | giống hệt #1 → order sai |
| 3 | pos=line, k=10 | **0.3641** | 0.8399 | 0.085 | line ĐÚNG (P .176/R .684/MRR5 .588) |
| 4 | #3 + expand-docs | 0.3641 | 0.6066 | 0.085 | expand-docs FAIL |
| 5 | pos=line, k=5, rule-only | **0.4092** | 0.8093 | 0.085 | EXEC .085; TABLE P .2621/R .5852/MRR5 .5882; DOC P .9168/R .7999/MRR5 .9457 |
| 6 | #5 + Qwen 7B (Kaggle) | 0.4092 | 0.8093 | **0.1047** | EXEC **0.0613** — LLM cải thiện ANSWER nhưng 233 query nhiều dòng bị crash. TABLES/DOCS y hệt #5 (đúng kỳ vọng: chỉ đổi trục codegen) |
| 7 | #6 + repair expression | (chờ) | (chờ) | (chờ) | dự kiến EXEC ≈ ANSWER ≈ 0.10; xây bằng `scripts/08_repair_expressions.py`, KHÔNG chạy lại GPU |
| 8 | **P1 rule-only** (không LLM) | **0.4241** | **0.8628** | **0.1285** | EXEC 0.1285. TABLE P .2702/R .6049/MRR5 .5752; DOC P .9212/**R .8635**/MRR5 .9572. Vượt cả lượt Qwen #6 trên MỌI trục. Targeted doc expansion kéo DOCS_RECALL .7999→.8635 (khác hẳn `--expand-docs` đại trà đã fail) |
| 9 | (nhầm) nộp bộ eval offline | 0.0 | 0.0 | 0.0 | Bộ eval là câu hỏi TỰ SINH id 1..N → grader khớp id nên sai hết. Đã thêm rào chắn trong `submission/build.py` |
| 10 | **P1.5 rule composite** | **0.4337** | **0.8777** | **0.1542** | EXEC 0.1542. TABLE P .2785/R .6161; DOC P .9483/R .8722/MRR5 .9654. Tất cả trục đều tăng, không dùng GPU |
| 11 | P1.6 hợp nhất scoring | 0.4337 | 0.8777 | 0.1522 | EXEC 0.1522. Trung tính (−1 câu/506): bỏ 24 đáp án nhưng chỉ ~1 câu đúng. **Rule đã tới lợi tức giảm dần** |
| 12 | Qwen 7B `--llm-target empty` | 0.4334 | 0.8774 | 0.1561 | EXEC 0.1561. LLM thêm 175 đáp án mới nhưng chỉ ra **~+2 câu đúng** (≈2% chính xác). Xem chẩn đoán bên dưới |

## CHẨN ĐOÁN LƯỢT QWEN #12 (đã audit artifact — đừng đo lại)

183 đáp án LLM, lỗi phân bố như sau:

| Lỗi | Tỷ lệ |
|---|---:|
| **KHÔNG lọc cột năm** → `.iloc[0]` lấy dòng đầu tiên, sai kỳ | **35%** |
| Quên chia `ANSWER_SCALE` → trả VND thô khi hỏi triệu/tỷ | 15% |
| `str.contains` thiếu `regex=False` → ngoặc thành nhóm regex | 90% |
| Dùng đúng `['col'] == N` mà shortlist đã đưa sẵn | chỉ 35% |

**Nguyên nhân gốc:** shortlist đã tính sẵn `var/label/code/col/col_name/value/
unit_scale` — tức là ĐÃ định vị đúng ô — nhưng ta vẫn bắt model tự viết lại phần
địa chỉ hoá bằng pandas, chỗ nó mắc lỗi đơn vị và cột. Giá trị thật của LLM
(chọn ô nào, dùng công thức gì) bị chôn dưới phần cơ khí ta bắt nó làm lại.

Ba warning trên Kaggle là triệu chứng của đúng các lỗi này, vô hại với tiến trình:
`match groups` = regex=True + ngoặc; `invalid escape sequence '\('` = model viết
`'\('` trong chuỗi thường → pattern không khớp → IndexError → câu vẫn rỗng;
`invalid value encountered in scalar divide` = 0/0 → NaN → executor loại.

**Arbitration chưa từng được kích hoạt:** `--llm-target empty` nghĩa là LLM chỉ
thấy câu rule bó tay, nên luôn `rule produced nothing` (183/183). Muốn dùng
ensemble thật phải chạy `--llm-target weak` hoặc `all`.

## P2.0 — STRUCTURED SELECTION (2026-08-04)

21. **`--llm-mode select`: model CHỌN ô, TA viết pandas.** File mới
    `codegen/selection.py` (schema JSON, parser chịu lỗi, synthesizer) +
    `SELECT_SYSTEM`/`build_select_user` trong `prompts.py` +
    `QuestionBundle.select_messages()` + nhánh `_selection_result` trong
    `generate.py`. Model xuất `{"op": ..., "operands": [chỉ số shortlist]}`;
    synthesizer sinh biểu thức với `regex=False`, điều kiện `['col'] == N` và
    phép chia `ANSWER_SCALE` — nên **ba lớp lỗi của #12 là bất khả thi về mặt
    cấu trúc**, không phải "hy vọng model làm đúng".
22. Mô phỏng end-to-end (120 câu, oracle selector): 22 câu rỗng → 19 có đáp án;
    query thiếu `regex=False` **0%** (trước 90%), query không lọc cột **0%**
    (trước 35%). Output ~30 token thay vì 256 → chạy được `--llm-target all`
    trong một phiên, nhờ đó arbitration mới có việc để làm.
23. Chế độ cũ vẫn giữ ở `--llm-mode code` để đối chứng.

16. **Eval offline phóng đại mức cải thiện ~7 lần.** P1.5: eval 0.157→0.3967
    (+153%) nhưng leaderboard 0.1285→0.1542 (+20%). Nguyên nhân: câu synthetic
    dùng đúng 6 mã VAS rule đã biết, và phân bố lớp khác thực tế (eval chia đều
    20%/lớp; thật: lookup 46%, ranking 21%). **Dùng eval để biết HƯỚNG, không
    dùng để dự đoán MỨC.**
17. **Bản đồ khoảng trống (artifact #10): 467/1012 câu có đáp án (46%),**
    độ chính xác trên phần đã trả lời ≈33%. Rỗng theo op: ranking 203/214 (95%),
    lookup 118/468 (25%), difference 97/147, average 64/70, growth 34/50.
19. **P1.6 — hợp nhất scoring của rule lookup vào `build_shortlist`.**
    `try_rule_answer` từng TỰ chấm điểm riêng, lệch với shortlist: dòng được
    shortlist cho 78 điểm (nhờ bằng chứng năm + mã VAS) chỉ đạt ~56 ở đường
    riêng → bị từ chối, câu thành rỗng. Sau khi hợp nhất: eval lớp `lookup`
    **0.783 → 0.883**. Trên 150 câu thật coverage giảm nhẹ (123→118) nhưng soi
    tay thì 4–5/6 câu "mất" vốn là đáp án RÁC (vd trả 678 nghìn tỷ tỷ từ nhãn
    tên công ty con) — đổi đáp án sai lấy im lặng không mất điểm.
20. **`ranking` thật là composite LỒNG NHAU** — vd "doanh nghiệp có mức tăng lớn
    nhất từ 2023 sang 2024 của tỷ lệ X" = rank(growth(ratio)). Rule engine mới
    giải một tầng. Bộ eval chỉ có ranking một tầng nên không lộ ra hạn chế này.
    Đây là chỗ LLM có cơ hội thắng rule — nhưng cần facts đã được định vị sẵn.

## PHÂN TÍCH SUBMISSION #6 (Qwen 7B, đã audit artifact)

- 1.012 entries: 648 vẫn là placeholder `0.0` (LLM/rule không ra gì) → **đây là
  khoảng trống điểm lớn nhất**, không phải chất lượng code của LLM.
- 364 câu có đáp án thực; 175 câu answer khác baseline #5.
- 233 query là script nhiều dòng → crash khi eval. Trong đó **56 câu vốn đã có
  rule one-liner chạy được** bị LLM ghi đè → mất EXEC so với baseline.
- Đã inline được 230/233 script thành biểu thức, giá trị khớp `answer` 100%;
  3 câu còn lại (id 682/716/890) dùng if-else, gán lại biến, tuple-unpack.
- Không có query nào dùng `np.` hay `print()` → namespace không phải vấn đề.

## CHẨN ĐOÁN 648 CÂU RỖNG (đo trên artifact #6 — đừng đo lại)

Phân lớp 1.012 câu (regex ticker/năm/từ-khoá-aggregate):

| Lớp | Số câu | % rỗng |
|---|---:|---:|
| 1 ticker, 1 năm, không aggregate | 505 | 43% |
| ≥2 ticker trong câu | 176 | 95% |
| ≥2 năm trong câu | 356 | 84% |
| có aggregate/ranking/"giả sử" | 367 | 92% |
| **PHỨC HỢP (hợp của 3 dòng trên)** | **507** | **85%** |

- **100% câu rỗng ĐỀU CÓ bảng trong context** → không phải lỗi "không có dữ liệu".
- LLM chỉ chuyển được 177/824 câu `none` của #5 thành có đáp án (~26% success).
- Router bắt ticker ĐÚNG (chỉ 2/176 câu multi-ticker bị thiếu) → entity extraction
  không phải nút thắt.
- **Nút thắt thật với câu đơn giản = TẦNG KHỚP NHÃN, không phải retrieval.**
  Đo 45 câu đơn-giản-rỗng: điểm khớp nhãn tốt nhất median 54 ở k=4, 58 ở top-20,
  **59 khi quét TOÀN BỘ báo cáo** → tăng k/rerank bảng KHÔNG cứu được.
  Soi tay: dòng đúng CÓ trong báo cáo nhưng chỉ đạt 62–64 điểm
  (vd metric "so du tra truoc cho nguoi ban" ↔ nhãn "Trả trước cho người bán"),
  dưới ngưỡng rule 78. Hai nguyên nhân: (a) `metric_norm` bị nhiễu vì tạo bằng
  cách TRỪ stopword khỏi câu (còn "so du", "tinh dong", mảnh tên công ty);
  (b) khớp thuần từ vựng không bắt được diễn đạt khác.
- Một phần câu "trông có vẻ đơn giản" thực ra là so sánh 2 công ty viết bằng
  tên đầy đủ (id 759, 780) → tỷ lệ phức hợp thực tế >50%.

## P1 ĐÃ TRIỂN KHAI (2026-08-04) — chi tiết ở `P1_IMPLEMENTATION.md`

8. **`answer` phải đúng đơn vị câu hỏi** (BTC xác nhận): hỏi "bao nhiêu phần trăm"
   → trả **90**, không phải 0.9. Code: `codegen/units.py`. Bẫy đã mắc: nhãn
   "Tỷ lệ ..." KHÔNG phải bằng chứng về thang đo (ô có thể chứa 0.9 hoặc 90);
   chỉ ký tự `%` hoặc |value| > 1.5 mới là bằng chứng.
9. **Metric extraction phải EXTRACTIVE, không subtractive.** `router/metric_phrase.py`
   cắt câu xuống cụm chỉ tiêu → điểm khớp nhãn 54→72 (median), rule tự trả lời
   thêm 11/45 câu từng rỗng.
10. **Ngưỡng rule = 62, KHÔNG dùng biên chống mơ hồ.** Hiệu chuẩn trên eval suite
    (coverage×accuracy lớp lookup): 78-không-biên 0.680 | **62-không-biên 0.720**
    | 62+biên8 0.440 | 62+biên15 0.320. Từ chối trả lời tốn điểm hơn chọn nhầm.
    Câu near-tie vẫn trả lời nhưng `confidence` hạ xuống 60 + gắn `AMBIGUOUS`
    để `--rule-first` không bỏ qua LLM.
11. Rule-only trên 150 câu đầu: `rule 79/none 71` → **`rule 123/none 27`**.
    Eval suite 125 câu: answer_acc 0.088 → **0.136** (lookup 0.44 → 0.68).
12. **PHẢI chạy lại `02_retrieve.py`** sau P1 — retrieval.jsonl cũ không có
    `plan`/`metric_variants`/`evidence_budget`.

## P1.5 — RULE ENGINE COMPOSITE (2026-08-04)

13. **Rule tự giải được câu phức hợp**, không chỉ lookup. File mới:
    `codegen/fact_resolver.py` (mỗi Fact → 1 ô cụ thể, có provenance + confidence),
    `codegen/rule_composite.py` (growth/difference/ratio/margin/ranking/sum/average
    → biểu thức pandas MỘT dòng), `codegen/arbitrate.py` (trọng tài rule↔LLM).
    Eval 300 câu: answer_acc **0.157 → 0.3967**; ratio 0→.400, growth 0→.383,
    difference 0→.283, ranking 0→.133, lookup giữ .783 (không hồi quy).
14. **Bốn bug đã sửa khi làm phần này (đừng lặp lại):**
    - `router`: chỉ thêm năm trước khi có ĐÚNG 1 ticker (trước đó câu so sánh
      2 công ty bị nở thành 4 fact → không giải được).
    - `metric_phrase`: phải cắt mệnh đề thời gian ĐẦU câu ("Năm 2015, ...") và
      xoá từ chỉ phép toán ("chênh lệch", "so với") — chúng kéo điểm khớp xuống <62.
    - `decompose.split_ratio_metric`: "A **trên** B" phải tách thành 2 metric khác
      nhau, nếu không `ratio` không bao giờ đủ 2 fact.
    - `shortlist`: cột có năm KHÁC năm hỏi bị phạt nặng (-30, không phải -6);
      lọc ngưỡng phải chạy LẠI sau khi cộng điểm cột; thêm tín hiệu **mã VAS**
      (`VAS_CODE_HINTS`) để tách "Lợi nhuận sau thuế" (60) khỏi
      "Lợi nhuận sau thuế chưa phân phối" (421).
15. **Trọng tài rule↔LLM** (`arbitrate.py`): đồng thuận → tin (conf≥85); lệch mà
    rule conf≥78 → giữ rule; lệch mà rule yếu → lấy LLM. Ghi lại lý do vào field
    `arbitration` của từng record để audit. Lý do mặc định thiên rule: rule-only
    (#8) đang thắng lượt Qwen (#6) trên mọi trục.

## TRẠNG THÁI & VIỆC TIẾP THEO

- ANSWER/EXEC 0.085 là trần của rule-baseline. Submission #5 vẫn là rule-only:
  artifact có 188 `source=rule`, 824 `source=none`, 0 `source=llm`; baseline
  Qwen trên Kaggle CHƯA có điểm leaderboard.
- **Việc giá trị nhất: chạy lại Kaggle codegen bằng payload schema v2 hiện hành.**
  Notebook `kaggle/vifinqa-codegen.ipynb` dùng backend `hf`, mặc định
  `Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit` (bitsandbytes NF4). Payload cũ
  từng gây log `LLM round 1: 1012 prompts`, không có baseline/chunk và CUDA OOM;
  nguyên nhân là notebook cũ không cập nhật `generate.py`. Notebook hiện tại
  KHÔNG hot-patch: nó kiểm SHA-256 rồi copy đúng nguyên trạng code trong payload.
- `SUBMISSION_K=5` đã được kiểm chứng: TABLES_F2=0.4092, tốt hơn k=10
  (0.3641), dù DOCS_F2 giảm 0.8399→0.8093. Có thể build ablation k=7 từ
  cùng retrieval/codegen, không cần chạy lại LLM. Lượt Qwen đầu tiên phải giữ
  `sub-k=5`, `expand-docs=False` để so trực tiếp với #5.
- Trong-doc table recall ≈ 0.82 → ~18% bảng gold (đa phần thuyết minh) chưa vào
  top-k. Giai đoạn 1 theo strategy: BGE-M3 + Qwen3-Reranker trong scope đã khoá.
- Payload Kaggle làm việc trong không gian `table_pos` (order) — map sang line
  diễn ra ở local lúc build submission → codegen_results.jsonl cũ/mới đều tương thích.
- P0 data snapshot đã dựng riêng ở `artifacts/store_p0` +
  `artifacts/retrieval_p0.jsonl` và full replay sạch. Nó đổi top-5 ở 181/1.012
  câu so với control #5, nên CHƯA trộn vào lượt Qwen control; xem session log.

## KIẾN TRÚC (tóm tắt)

```
data/ViFinQA/                nguồn HF: financial_statements/, questions/, code_stock.csv
vifinqa/                     package chính (xem README §5 cho map chi tiết)
  config.py                  TABLE_POS_MODE="line", SUBMISSION_K=5, đường dẫn
  extraction/                parse <table> + số VN + đơn vị + store parquet (line_no ở đây)
  router/ retrieval/         structured lookup (ticker/năm/doc_type) + BM25 in-scope
  codegen/                   HF batch + semantic guard/self-debug + rule fallback
  submission/build.py        map table_pos→line_no, build zip
  validation/                synthetic val (gold cũng dùng line-number)
scripts/01..07               pipeline CLI theo thứ tự
kaggle/                      notebook verify đúng một payload + kaggle_codegen.py
artifacts/                   store/, retrieval.jsonl, submission/ (gitignored)
```

## HIỆU NĂNG KAGGLE (đo trên dữ liệu thật, 2026-08)

- Prompt dài: k=5 → ~3.070 token, k=4 → ~2.630, k=3 → ~2.040 (median).
- `--rule-first` chỉ bỏ qua **~10%** câu (rule conf≥90) → vẫn còn ~900 câu cho LLM.
- 7B NF4 + batch 4 + n=2 + max_tokens 384 + k=5 = **quá nặng cho 1 phiên 9h**
  (user chạy 9h chưa xong). Cấu hình khuyến nghị: `--n 1 --k 4 --max-tokens 256
  --batch-size 4 --temperature 0`. Prefill chiếm phần lớn → giảm `k` là đòn
  bẩy mạnh nhất; greedy giúp resume ổn định hơn khi chỉ lấy một sample.
- `run_codegen` giờ **crash-safe**: ghi rule-baseline đầy đủ ngay phút thứ ~2,
  rồi LLM ghi đè dần theo chunk (`--checkpoint-every 32`), `--time-budget-min`
  dừng sạch trước giới hạn phiên. Resume chỉ bỏ qua record có
  `llm_attempt_status=completed` và exact `run_signature`, kể cả attempt bị reject hoặc
  arbitration giữ rule. File out luôn đủ 1.012 dòng và nộp được ở mọi thời điểm.
- HF runner tự giảm batch khi CUDA OOM (`4→2→1`) và checkpoint round 1 trước
  self-debug. Resume chỉ nhận kết quả có cùng `run_signature`; phiên Kaggle mới
  phải đưa checkpoint cũ về đúng đường dẫn `/kaggle/working/codegen_results.jsonl`.

## GOTCHAS

- **KHÔNG BAO GIỜ nộp `artifacts/eval/eval_submission`** — câu hỏi tự sinh, id
  không khớp bộ thi ⇒ 0.0 toàn bộ + mất lượt nộp (đã dính lần #9). `build.py` giờ
  tự phát hiện và đổi tên thành `OFFLINE_EVAL_DO_NOT_UPLOAD.zip`; chỉ nộp file
  tên đúng `submission.zip`.
- **Rebuild payload sau P1**: cả `vifinqa/` lẫn `retrieval.jsonl` đều đổi. Phải
  chạy `04_make_kaggle_payload.py` với `--dataset-id` của TỪNG acc trước mỗi lần
  up (id nằm trong `dataset-metadata.json`), và đặt `KAGGLE_CONFIG_DIR` khi đẩy
  sang acc #2. Chi tiết: `P1_IMPLEMENTATION.md` §6.1.

- Đổi code extraction ⇒ PHẢI rebuild store (`scripts/01_build_store.py`, ~30 phút full).
- `artifacts/validation/val_gold.json` sinh trước 2026-08-02 dùng vị trí order —
  chạy lại `scripts/06_gen_validation.py` nếu cần đánh giá offline.
- vLLM API trôi giữa version: `VllmBatchClient` tự drop kwarg không hỗ trợ.
  T4 lỗi V1/compute-capability → pin `vllm==0.7.3`.
- Notebook P2.2 hiện hành yêu cầu đúng một payload schema 6 và kiểm SHA-256 mọi runtime/mask;
  attach nhiều phiên bản payload hoặc payload chưa upload lại sẽ dừng ngay.
- Windows console: mọi script đã gọi `setup_stdout()` (UTF-8).
- Corpus TiniX license CC BY-NC 4.0 — ghi nguồn, phi thương mại.
- Mỗi lần nộp chỉ đổi MỘT biến (trừ khi các metric nhóm độc lập: k ảnh hưởng
  TABLES, codegen ảnh hưởng ANSWER/EXEC — có thể gộp).
