# Session 2026-08-11 — hoàn thành P2.4 tune gold

## Kết quả

Đã tạo tune gold độc lập từ dữ liệu báo cáo ViFinQA cho đủ 100 câu, không dùng Qwen output
làm nhãn. Mỗi record chứa exact `(report_id, table_pos, row, col)`, raw value, unit scale,
typed AST, canonical hashes và biểu thức pandas replay.

Bản cuối:

- gold: `artifacts/devset_p24/p24_tune_gold.final.jsonl`;
- canonical SHA256:
  `8a3725276c4baafbb3ecbeaa3ea8f3bfcc488f16406e3994bd09b3a1a255d331`;
- strict forensic validation: 100/100 complete;
- semantic branch audit: 21/21 complex cases;
- notebook executed: `notebooks/P2_4_TUNE_GOLD_AUDIT_FINAL.ipynb`;
- locked set: chưa mở.

## Các lớp code được thêm

- Forensic exact-cell loader cho numeric OCR cells bị serializer chuẩn bỏ.
- Standard-statement metric resolver với exact accounting labels, VAS-code fallback hẹp,
  OCR punctuation/whitespace tolerance và strict value-column classifier.
- Typed authoring cho filter, median partition, count, conditional aggregate,
  argmin/argmax-project và nested financial formulas.
- Independent forensic validator, descriptive quality audit và complete semantic branch audit.
- Reproducible executed notebook.

## Lỗi được audit bắt trước khi đóng gold

Predecessor đầu tiên replay đúng nhưng resolver chọn:

- OGC PAT = `18` từ cột `CHỈ TIÊU` thay vì giá trị năm 2024;
- HPX inventory = `5.8` từ cột `TM` thay vì giá trị hàng tồn kho.

Resolver v6 loại cột metadata/rỗng trước khi chọn amount column. Sau rebuild, ba answer đổi:

- ID 417: `1.60 -> 0.94`;
- ID 446: `50.02 -> 49.58`;
- ID 447: `127.18 -> 128.64`.

ID 397 giữ answer `146.61` vì nhánh thắng là NVL, nhưng evidence HPX đã được sửa. Artifact
tiền nhiệm có physical `DO_NOT_USE` marker.

## Audit cuối

- 578 evidence references;
- 502 exact cells duy nhất;
- 397 tables / 235 reports;
- output: 70 number, 16 percent, 6 ratio, 3 percentage-point, 3 year, 2 count;
- 21/21 complex cases có recomputation và branch selection riêng;
- không có complex evidence ở `CHỈ TIÊU/MS/TM`;
- duplicate invariants: ID 493=552 và 554=576;
- outlier hợp lệ được giữ làm review flags, không clip: ID 615 tăng 224.44%, ID 732 tỷ
  số 902.43 lần, ID 447 tỷ số 128.64 lần.

## Hai giả định nghiệp vụ được ghi rõ

- ID 425: phát hành thêm 10% cổ phiếu từ đầu năm, profit giữ nguyên, nên scenario EPS =
  reported basic EPS / 1.1; năm ROAE cao nhất là 2024, kết quả 4.49 nghìn đồng/cổ phiếu.
- ID 516: “tổng giá trị ghi nhận công cụ phái sinh” dùng total row của derivative and other
  financial assets; quỹ phúc lợi chọn năm 2022, thay đổi 100.072 so với 226.545 triệu đồng
  = -55.83%.

## Tài liệu vận hành

Xem `RUNBOOK_P2_4_TUNE_GOLD.md`. Hai file cũ `RUNBOOK.md` và
`P2_4_LABELING_GUIDE.md` không thể cập nhật bằng `apply_patch` do sandbox ACL; không dùng
shell overwrite để tránh vi phạm integrity. Runbook bổ sung là tài liệu chuẩn cho P2.4 final.
