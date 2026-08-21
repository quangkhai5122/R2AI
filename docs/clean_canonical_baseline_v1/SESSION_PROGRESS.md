# Tiến độ phiên triển khai 2026-08-21

## Provenance

- base: `main@0ce20aa72e636c33af659709b38d98da89e98c77`;
- branch: `codex/clean-canonical-baseline-v1`;
- canonical source: port có chọn lọc `vifinqa/finance/metrics.py` và tests từ `f08b927`;
- không port `783a773:vifinqa/codegen/formula_solver.py`;
- không port `ef77d55` six-ID allowlist/blend.

## G0 — freeze/quarantine

`experiments/clean_canonical_baseline_v1/registry.json` phân biệt repo-reported historical checkpoints với clean candidates. `.2806/.2866` là historical, artifact absent, public-derived và không đủ điều kiện làm clean seed.

## G1 — canonical production baseline

Historical schema-8 runner giữ nguyên. Clean path thêm environment fingerprint, exact CPU dependency lock, CI, B0 wrapper, submission wrapper và source-only payload. Các file `.orig` bị loại khỏi payload nhưng chưa xóa khỏi repository để tránh gộp cleanup với semantic change.

## G2 — canonical finance/reasoning layer

Đã đưa 139 metric vào source chính của clean path; route lưu canonical keys/qualifiers và retrieval mở rộng derived metric thành atomic components. Operator registry mô tả đúng typed Selection v2 surface và được hash trong payload. Không thêm một formula executor mới.

## Bằng chứng chạy

- `22 passed` cho canonical/clean contract;
- smoke retrieval config SHA-256 `05623407eb7e3e34aca1c8b839e9225bcb4ee14013b29b7fb8adae442332917c`;
- B0 smoke: 3 record, source distribution `rule: 3`;
- local environment SHA-256 `80be1c1682d39ad7a0f841432ce005326eeae8273a4c00e80298d3e7ab5411df`.

Smoke chỉ xác minh pipeline, không chứng minh accuracy hoặc generalization. B1 GPU và full 1.012-record build vẫn pending.
