"""Create the canonical Qwen 14B runtime-NF4 Kaggle notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "kaggle" / "history" / "clean_canonical_baseline_v1" /
          "vifinqa-clean-canonical-v1-nf4-v2.ipynb")
OUTPUT = ROOT / "kaggle" / "vifinqa-clean-canonical-b1-14b-nf4.ipynb"


def main() -> None:
    notebook = nbformat.read(SOURCE, as_version=4)
    if len(notebook.cells) != 8:
        raise RuntimeError(f"expected 8 cells, found {len(notebook.cells)}")

    notebook.cells[0].source = """# ViFinQA clean canonical B1 - Qwen 14B + runtime NF4

Schema 9, Selection v2, no ID masks, and no official-derived gold. This notebook
uses the base Qwen/Qwen2.5-Coder-14B-Instruct checkpoint with bitsandbytes NF4.
AWQ/GPTQ checkpoints, gptqmodel, and autoawq are outside the canonical runtime.
Run all cells top to bottom on a Kaggle GPU session."""

    notebook.cells[1].source = """from pathlib import Path
import hashlib, json, os, sys

SCHEMA_VERSION = 9
RUNTIME_PROFILE = 'hf-bitsandbytes-nf4-v1'
RUNTIME_LAUNCHER = 'code/kaggle_clean_codegen_nf4.py'
MODEL = os.environ.get(
    'VIFINQA_MODEL', 'Qwen/Qwen2.5-Coder-14B-Instruct'
)
SMOKE_LIMIT = 3
WORK = Path('/kaggle/working')
SMOKE_OUT = WORK / 'codegen_smoke_nf4.jsonl'
FULL_OUT = WORK / 'codegen_results_nf4.jsonl'
AUDIT_OUT = WORK / 'codegen_audit_nf4.json'
RUN_CONFIG_OUT = WORK / 'run_config_nf4.json'
SUBMISSION_DIR = WORK / 'submission_clean_nf4'

override = os.environ.get('VIFINQA_PAYLOAD', '').strip()
manifest_paths = ([Path(override) / 'payload-manifest.json'] if override else
                  list(Path('/kaggle/input').glob('**/payload-manifest.json')))
matches = []
for manifest_path in manifest_paths:
    try:
        obj = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        continue
    files = obj.get('files') or {}
    if (obj.get('schema_version') == SCHEMA_VERSION and
            obj.get('profile') == 'clean' and
            obj.get('runtime_profile') == RUNTIME_PROFILE and
            obj.get('runtime_launcher') == RUNTIME_LAUNCHER and
            RUNTIME_LAUNCHER in files):
        matches.append((manifest_path.parent, obj))
if len(matches) != 1:
    found = [str(path.parent) for path in manifest_paths]
    raise RuntimeError(
        f'Expected exactly one schema-9 {RUNTIME_PROFILE} payload with the '
        f'side-by-side NF4 launcher, found {len(matches)}. Attached manifests: {found}. '
        'Upload/import the new dataset version; set VIFINQA_PAYLOAD only when '
        'multiple payloads are intentionally attached.'
    )
PAYLOAD, MANIFEST = matches[0]
RUNNER = PAYLOAD / RUNTIME_LAUNCHER
VALIDATOR = PAYLOAD / 'code' / 'validate_clean_codegen.py'
for required in (RUNNER, VALIDATOR, PAYLOAD / 'code' / 'kaggle_clean_codegen.py',
                 PAYLOAD / 'retrieval.jsonl', PAYLOAD / 'store' / 'reports.parquet'):
    if not required.is_file():
        raise FileNotFoundError(required)
payload_manifest_file_sha256 = hashlib.sha256(
    (PAYLOAD / 'payload-manifest.json').read_bytes()
).hexdigest()
print('PAYLOAD:', PAYLOAD)
print('schema:', MANIFEST['schema_version'], '| profile:', MANIFEST['runtime_profile'])
print('launcher:', RUNNER, '| records:', MANIFEST['retrieval_records'], '| model:', MODEL)
"""

    notebook.cells[2].source = """# Dependency, GPU, and model-access preflight.
import importlib, importlib.metadata, importlib.util, platform, subprocess
from datetime import datetime, timezone
from packaging.specifiers import SpecifierSet
from packaging.version import Version

if importlib.util.find_spec('torch') is None:
    raise RuntimeError('Kaggle GPU image has no torch installation')
requirements = {
    'transformers': '==5.0.0',
    'accelerate': '==1.13.0',
    'bitsandbytes': '==0.50.1',
    'tqdm': '==4.67.3',
}
install_specs = []
for module, version_spec in requirements.items():
    try:
        installed = importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed is None or Version(installed) not in SpecifierSet(version_spec):
        install_specs.append(f'{module}{version_spec}')
if install_specs:
    print('Installing canonical NF4 dependencies:', install_specs)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *install_specs])
    importlib.invalidate_caches()
for module, version_spec in requirements.items():
    installed = importlib.metadata.version(module)
    if Version(installed) not in SpecifierSet(version_spec):
        raise RuntimeError(f'{module}={installed} does not satisfy {version_spec}')

import torch
import bitsandbytes
import transformers
from transformers import BitsAndBytesConfig
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable. Enable a Kaggle GPU accelerator, then restart Run all.')
gpus = []
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    gpus.append({'index': i, 'name': torch.cuda.get_device_name(i),
                 'capability': list(torch.cuda.get_device_capability(i)),
                 'memory_bytes': int(props.total_memory)})

model_revision = ''
if not Path(MODEL).exists():
    try:
        from huggingface_hub import model_info
        model_revision = model_info(MODEL).sha
    except Exception as exc:
        raise RuntimeError(
            f'Cannot access {MODEL}. Enable Kaggle Internet or attach a local model '
            'dataset and set VIFINQA_MODEL to that directory.'
        ) from exc
runtime_preflight = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version.split()[0], 'platform': platform.platform(),
    'model': MODEL, 'model_revision_observed': model_revision,
    'runtime_profile': RUNTIME_PROFILE, 'cuda': str(torch.version.cuda or ''),
    'gpus': gpus,
    'packages': {name: importlib.metadata.version(name) for name in
                 ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'tqdm')},
}
(WORK / 'runtime_preflight_nf4.json').write_text(
    json.dumps(runtime_preflight, ensure_ascii=False, indent=2), encoding='utf-8')
run_config = {
    'schema_version': 1,
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'payload': str(PAYLOAD),
    'payload_manifest_file_sha256': payload_manifest_file_sha256,
    'payload_declared_default_model': MANIFEST.get('default_model'),
    'effective_model': MODEL,
    'model_revision_observed': model_revision,
    'runtime_profile': RUNTIME_PROFILE,
    'generation': {
        'backend': 'hf', 'load_4bit': True, 'llm_mode': 'select_v2',
        'llm_target': 'all', 'k': 0, 'n': 2, 'temperature': 0.2,
        'max_tokens': 512, 'max_input_tokens': 5000, 'batch_size': 4,
        'checkpoint_every': 32, 'time_budget_min': 420, 'seed': 13,
    },
    'runtime_preflight': runtime_preflight,
    'execution': {'status': 'not_started'},
}
RUN_CONFIG_OUT.write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(runtime_preflight, ensure_ascii=False, indent=2))
print('run config ->', RUN_CONFIG_OUT)
"""

    notebook.cells[3].source = """# Exact-path smoke: payload hashes, NF4 load, and Selection v2.
common = [
    sys.executable, str(RUNNER), '--payload', str(PAYLOAD), '--backend', 'hf',
    '--model', MODEL, '--load-4bit', '--llm-mode', 'select_v2',
    '--llm-target', 'all', '--k', '0', '--n', '2', '--temperature', '0.2',
    '--max-tokens', '512', '--max-input-tokens', '5000', '--batch-size', '4',
    '--seed', '13',
]
smoke_cmd = [*common, '--out', str(SMOKE_OUT), '--limit', str(SMOKE_LIMIT),
             '--checkpoint-every', str(SMOKE_LIMIT), '--time-budget-min', '60',
             '--runtime-report', str(WORK / 'runtime_smoke_nf4.json'), '--no-resume']
subprocess.run(smoke_cmd, check=True)
subprocess.run([sys.executable, str(VALIDATOR), '--payload', str(PAYLOAD),
                '--codegen', str(SMOKE_OUT), '--expected-count', str(SMOKE_LIMIT),
                '--require-complete-llm'], check=True)
"""

    notebook.cells[4].source = """# Full B1 run. Safe to rerun only with the exact same signature.
full_started = datetime.now(timezone.utc)
full_status = 'failed'
try:
    full_cmd = [*common, '--out', str(FULL_OUT), '--checkpoint-every', '32',
                '--time-budget-min', '420',
                '--runtime-report', str(WORK / 'runtime_full_nf4.json')]
    subprocess.run(full_cmd, check=True)
    full_status = 'completed'
finally:
    full_finished = datetime.now(timezone.utc)
    run_config = json.loads(RUN_CONFIG_OUT.read_text(encoding='utf-8'))
    run_config['execution'] = {
        'status': full_status,
        'started_at_utc': full_started.isoformat(),
        'finished_at_utc': full_finished.isoformat(),
        'elapsed_seconds': (full_finished - full_started).total_seconds(),
    }
    RUN_CONFIG_OUT.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(run_config['execution'], indent=2))
"""

    notebook.cells[5].source = """# Fail closed if all 1,012 LLM attempts did not complete.
subprocess.run([sys.executable, str(VALIDATOR), '--payload', str(PAYLOAD),
                '--codegen', str(FULL_OUT), '--require-complete-llm',
                '--report', str(AUDIT_OUT)], check=True)
audit = json.loads(AUDIT_OUT.read_text(encoding='utf-8'))
assert audit['validated_records'] == MANIFEST['retrieval_records']
assert audit['llm_completed'] == MANIFEST['retrieval_records']
print('FULL CHECKPOINT VERIFIED:', audit['codegen_sha256'])
"""

    notebook.cells[6].source = """# Build, replay-validate, and inspect the final archive.
import zipfile
from tqdm.auto import tqdm
sys.path.insert(0, str(PAYLOAD / 'code'))
from vifinqa.submission.build import build_submission

with tqdm(total=3, desc='submission QA', unit='stage', dynamic_ncols=True) as bar:
    bar.set_postfix_str('build + replay')
    zip_path = build_submission(
        PAYLOAD / 'retrieval.jsonl', FULL_OUT, PAYLOAD / 'store', SUBMISSION_DIR,
        sub_k=5, pos_mode='line', expand_docs=False,
    )
    bar.update(1)
    if (SUBMISSION_DIR / 'DO_NOT_UPLOAD.txt').exists():
        raise RuntimeError('Submission was marked DO_NOT_UPLOAD')
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if 'results.json' not in names:
            raise RuntimeError('submission.zip has no results.json')
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in names):
            raise RuntimeError('unsafe path inside submission.zip')
        submitted = json.loads(archive.read('results.json'))
    bar.update(1)
    if len(submitted) != MANIFEST['retrieval_records']:
        raise RuntimeError(f'archive contains {len(submitted)} results')
    if len({row['id'] for row in submitted}) != len(submitted):
        raise RuntimeError('duplicate IDs in submission results.json')
    zip_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    handoff = {
        'submission_zip': str(zip_path), 'submission_sha256': zip_sha256,
        'entries': len(submitted), 'archive_members': len(names),
        'codegen_sha256': audit['codegen_sha256'],
        'run_signature': audit['run_signature'],
        'run_config_sha256': hashlib.sha256(RUN_CONFIG_OUT.read_bytes()).hexdigest(),
        'model': MODEL,
        'model_revision_observed': model_revision,
        'runtime_profile': RUNTIME_PROFILE,
    }
    (WORK / 'submission_manifest_nf4.json').write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2), encoding='utf-8')
    bar.update(1)
print(json.dumps(handoff, ensure_ascii=False, indent=2))
"""

    notebook.cells[7].source = """## Handoff

Download submission_clean_nf4/submission.zip only after the full validator and
archive QA pass. Preserve runtime_preflight_nf4.json, runtime_smoke_nf4.json,
runtime_full_nf4.json, run_config_nf4.json, codegen_results_nf4.jsonl,
codegen_audit_nf4.json, submission_manifest_nf4.json, and the executed notebook
or complete log. Resume only with the same full output and exact signature."""

    for cell in notebook.cells:
        if not cell.get("id"):
            raise RuntimeError("all notebook cells must have stable IDs")
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"notebook -> {OUTPUT}")


if __name__ == "__main__":
    main()
