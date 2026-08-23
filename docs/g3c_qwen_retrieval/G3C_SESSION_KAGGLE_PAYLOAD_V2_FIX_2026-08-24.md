# G3C Kaggle payload v2 transport fix - 2026-08-24

## Outcome

The Kaggle cell-2 failure is fixed through payload schema v2. The active dev
protocol and payload are ready for a new Kaggle dataset version.

No Qwen model was loaded and no retrieval output or metric was observed before
this amendment. The retrieval ladder, model revisions, prompts, thresholds,
gate, data split and promotion policy are unchanged.

## Observed failure

The first uploaded payload passed local validation but Kaggle stopped at:

    ValueError: payload file-set mismatch extra=[] missing=['dataset-metadata.json']

The failure occurred inside validate_gpu_payload before the runner or Qwen
models started.

## Root cause

dataset-metadata.json has two different roles:

- locally, it is a Kaggle CLI control file used to create or version a dataset;
- in /kaggle/input, Kaggle does not expose that control file as mounted dataset
  content.

Payload schema v1 included the file in the exact core-file inventory and read it
unconditionally. That made the local upload directory valid but the correctly
mounted Kaggle dataset invalid.

## Fix

Payload schema is now g3c_gpu_payload_v2.

The manifest separates:

- 247 mandatory core files, each with exact size and SHA-256;
- one hash-bound upload sidecar: dataset-metadata.json;
- the payload manifest itself.

The sidecar is validated when present in the local upload directory. Its absence
after Kaggle mounting is accepted. Any missing core file, unexpected non-sidecar
file, duplicate path, unsafe path or hash drift is still rejected.

The two build/freeze CLIs now default to:

    experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze_v2.json

This prevents accidental reuse of the failed v1 contract.

## Provenance preserved

Failed pre-inference v1 evidence remains unchanged:

    protocol freeze:
    experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze.json

    protocol fingerprint:
    116f26a5ed69c166087a68668e7b9c71bbe59a15749d440e78d1029db48c2b7e

    payload:
    artifacts/g3c_v1/dev_payload

    payload fingerprint:
    fb0f72ffe441e025d9d815b6f81264c15687448bf4075783068ccb311f0bbd13

Active v2 evidence:

    protocol freeze:
    experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze_v2.json

    behavior tree SHA-256:
    f1f99c3189393ccd092784b3cd756b6989fe928335f1693255c7da831506e0b5

    protocol fingerprint:
    af86c8ffc276cc0a92ceeb3cc0ddc3a7eeaa7b6a1e4430dc2cade8ac4c9621c5

    payload:
    artifacts/g3c_v1/dev_payload_v2

    payload fingerprint:
    5584010ab665510b592662cfc42327da19e1148c87fdef8b5cfe4155cdcbd9a4

    upload directory:
    249 files, 95,873,569 bytes, 91.43 MiB

No v1 file was overwritten.

## Verification

Measured local evidence:

- targeted G3C suite: 22 passed;
- full repository suite: 378 passed, zero failures/errors/skips, 34.168 seconds;
- local payload v2 validation with the sidecar present: passed;
- Kaggle-mount simulation with dataset-metadata.json absent: passed;
- restored local payload validation: passed;
- packaged three-question fake R0-R4 run: passed;
- fake result import validation: passed;
- all six stages had zero hard-constraint violations.

The fake backend and mount simulation are engineering evidence only. They do not
establish Qwen retrieval quality or scientific gate performance.

## Required next action

The dataset slug already exists. Upload a new version, not a new dataset:

    kaggle datasets version -p artifacts/g3c_v1/dev_payload_v2 -m "G3C payload schema v2 transport fix protocol af86c8ff payload 5584010a" --dir-mode zip

Then in Kaggle:

1. detach or refresh the old v1 dataset input;
2. attach the newest version of
   lequangkhai5122005/vifinqa-g3c-qwen-retrieval-dev-v1;
3. restart the notebook session;
4. enable GPU and Internet;
5. run all cells without editing the notebook;
6. verify cell 1 prints payload fingerprint 5584010a...;
7. verify cell 2 prints payload_valid with the same fingerprint.

The remote v2 execution remains unverified until those checks and the full Qwen
run complete. Do not build a promotion payload.

