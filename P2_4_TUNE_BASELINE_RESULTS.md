# P2.4 tune baseline — kết quả trước Structured Selection v2

Gold: `artifacts/devset_p24/p24_tune_gold.final.jsonl`.

Canonical gold SHA256:
`8a3725276c4baafbb3ecbeaa3ea8f3bfcc488f16406e3994bd09b3a1a255d331`.

## Hai candidate v3

| candidate | correct | ANSWER | EXEC | executable / coverage | weighted ANSWER* |
|---|---:|---:|---:|---:|---:|
| year-only v3 | 32/100 | 0.320 | 0.320 | 0.830 | 0.321179 |
| all-types v3 | 32/100 | 0.320 | 0.320 | 0.830 | 0.321179 |

\* Post-stratified conditional trên 16/21 represented strata;
`represented_population_mass=0.988142`, không phải population-wide unbiased estimate.

Reports:

- `artifacts/devset_p24/eval_year_only_v3_tune_final.json`
- `artifacts/devset_p24/eval_all_v3_tune_final.json`

Hai candidate tạo dự đoán giống hệt nhau trên cả 100 tune IDs. Vì vậy P2.4 hiện chưa đo được
increment all-types so với year-only; 40 replay all-types nằm ngoài mẫu tune cố định. Không suy
ra leaderboard gain là giả — chỉ kết luận tune này không chứa các ID tạo khác biệt đó.

## Coverage không còn là nút thắt chính

- Coverage/executable: 83/100.
- Correct: 32/100.
- Precision có điều kiện trên câu covered: 32/83 = **38.6%**.
- Sai dù query chạy được: **51 câu**.
- Structural none: **17 câu**.

Do đó một rescue chỉ biến 17 `none` thành query không đủ. P2.2 phải sửa chọn fact, typing,
operator và nested semantics của 51 câu đã executable nhưng sai.

Nếu loại 39 câu `lookup|single`:

- còn 61 câu;
- đúng 10/61 = **16.4%**;
- covered 46/61 = **75.4%**;
- đúng trên covered 10/46 = **21.7%**.

Đây là khoảng trống chính cho Structured Selection v2.

## Theo source

| source | n | correct | accuracy |
|---|---:|---:|---:|
| rule | 24 | 16 | **0.667** |
| rule_composite | 10 | 3 | 0.300 |
| llm_select | 46 | 11 | **0.239** |
| llm_select_p21r | 3 | 2 | 0.667 |
| none | 17 | 0 | 0.000 |

Rule-first phải được giữ. P2.2 chỉ thay record chưa giải hoặc record LLM có verifier chứng minh
plan tốt hơn; không được ghi đè rule đúng chỉ để tăng số câu có LLM output.

## Theo output type

| output | n | correct | accuracy | coverage |
|---|---:|---:|---:|---:|
| number | 70 | 29 | 0.414 | 0.900 |
| year | 3 | 2 | 0.667 | 1.000 |
| percent | 16 | 1 | **0.063** | 0.750 |
| percentage_point | 3 | 0 | **0.000** | 1.000 |
| ratio | 6 | 0 | **0.000** | 0.167 |
| count | 2 | 0 | **0.000** | 0.500 |

Count + percent + percentage-point + ratio chỉ đúng **1/27 = 3.7%**. Đây là bằng chứng
trực tiếp rằng typed output contract, scale/unit algebra và operator semantics phải đứng trước
self-consistency hoặc sampling nhiều lần.

## Theo cấu trúc đáng chú ý

- `lookup|single`: 22/39 = 56.4% — vẫn còn lỗi exact-cell nhưng là nhóm mạnh nhất.
- `lookup|multi_*`: 0/8 — multi-fact lookup thực chất là filter/conditional sum, không phải lookup.
- `ranking|multi_2_4`: 4/13 = 30.8%.
- `ranking|multi_5_plus`: 1/8 = 12.5%.
- toàn bộ ranking multi: 5/21 = 23.8%.
- difference: 3/15 = 20.0%.
- average: 1/6 = 16.7%.

## Kiến trúc P2.2 nên đóng băng để triển khai

1. **Atomic fact slots có provenance**
   - entity/ticker, report scope, year/period role, metric code/label, statement type;
   - exact value column classifier loại `CHỈ TIÊU/MS/TM`, beginning/prior/current;
   - mỗi slot trả `(report, table, row, col, value, scale)` và confidence/rejection reason.

2. **Typed plan IR**
   - scalar types: money, raw_number, ratio, percent, percentage_point, count, year;
   - unit-scale propagation bắt buộc;
   - operators: lookup, arithmetic, growth, margin, average/sum, compare/filter,
     median partition, count, argmin/argmax-project, conditional aggregate;
   - không cho LLM đưa raw pandas hoặc raw numeric constant thay evidence.

3. **Deterministic compiler + verifier**
   - compile plan sang pandas expression;
   - exact evidence refs bằng plan refs;
   - reject metadata cells, duplicate stable cells, divide-by-zero, sentinel output,
     output-type mismatch và inconsistent period roles;
   - rule-first arbitration; P2.2 chỉ fill `none` hoặc LLM failure trong ablation đầu.

4. **Stage rollout trên tune**
   - P2.2a: percent/pp/ratio/count typing và simple two-fact plans;
   - P2.2b: filter + argmin/argmax-project cho ranking multi;
   - P2.2c: median, conditional sum/average, nested growth(ratio) và scenario transforms;
   - mỗi stage báo raw + weighted ANSWER/EXEC, coverage, non-single accuracy,
     per-output/per-stratum và rule regression.

## Gate khoa học

- Không mở locked trong development.
- Không chọn architecture bằng leaderboard rồi báo tune như confirmatory.
- Một thay đổi chỉ được giữ khi tune gold final cho thấy gain ở target stratum/type và không
  làm giảm rule correctness ngoài tolerance đã định trước.
- Tune chỉ có 100 câu; báo đúng số câu thay đổi/correct, không chỉ báo phần trăm.
