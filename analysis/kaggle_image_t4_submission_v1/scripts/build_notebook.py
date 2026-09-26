"""Build a private, offline, submission-only notebook from frozen V10 sources.

Building is local and does not start Kaggle or submit the competition.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
V8 = HERE.parents[1] / 'kaggle_image_t4_v1'
V10 = HERE.parents[1] / 'kaggle_image_t4_vision_cache_v1'
FROZEN_COMMIT = '26dafd65b91e054577110f4c695f7d72a7660faf'
V8_BUILDER_SHA = 'a631b9dcd4b66380a63046c9bfad3e157284cd5d586e53f4cb25c5f9d5a0b6d3'
VISION_CACHE_SHA = 'b15b4bd8608e32ef85ef709f1dc8a5c19be71db95ae088e45236dd2b2afc3980'
ASSET_MANIFEST_SHA = '4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7'
SESSION_MINUTES = 530  # Future scored run: 9-hour ceiling minus 10-minute reserve.


def build(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an earlier notebook build')
    source = V8 / 'scripts' / 'build_notebook.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != V8_BUILDER_SHA:
        raise ValueError('Frozen Version 8 notebook builder changed')
    cache_path = V10 / 'scripts' / 'vision_cache.py'
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != VISION_CACHE_SHA:
        raise ValueError('Frozen Version 10 vision cache changed')
    spec = importlib.util.spec_from_file_location('frozen_submission_notebook_builder', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'v8.ipynb'
        original.build(path)
        notebook = json.loads(path.read_text())
    cells = notebook['cells']
    cells[0] = original.cell('markdown',
        '# Private RSNA Knee image submission — frozen V10 inference\n\n'
        f'Scientific/runtime snapshot: `{FROZEN_COMMIT}`. Two independent T4 '
        'workers discover actual test IDs at execution, split sorted IDs by '
        'alternating position and restore original test.csv order. Same weights, '
        'NF4/FP16 compute, existing FP32 residual policy, six images, processor, '
        'same-study vision cache, twelve prompts and No/Yes scores.\n\n'
        'No example prediction file, visible-ID list, repeated diagnostic or '
        'activation trace is needed. Numerical finiteness, asset/source hashes, '
        'worker completion, cache accounting and submission schema remain enforced. '
        'Any worker failure aborts without a final CSV. No retry, score repair, '
        'fallback, CPU/disk offload, training or automatic submission.\n\n'
        'Internet OFF; attach the original Private model asset and competition. '
        'The 530-minute internal deadline is a production ceiling for a later '
        'explicitly authorized scored execution, not current compute approval. '
        'A complete run writes /kaggle/working/submission.csv.')
    # All model/math/preprocessing sources remain byte-identical; only dispatch changes.
    cells = [c for c in cells[:-1] if "Path('/kaggle/working/isolated_runner.py')" not in c['source']]
    old_deadline = 'SESSION_DEADLINE_MONOTONIC=time.monotonic()+115*60'
    if cells[1]['source'].count(old_deadline) != 1:
        raise ValueError('Unexpected frozen deadline cell')
    cells[1]['source'] = cells[1]['source'].replace(old_deadline,
        f'SESSION_DEADLINE_MONOTONIC=time.monotonic()+{SESSION_MINUTES}*60')
    # Pin the same 32-file manifest as well as preserving the original per-file audit.
    asset_cells = [c for c in cells if 'ASSET_MANIFEST, ASSET = matches[0]' in c['source']]
    if len(asset_cells) != 1:
        raise ValueError('Unexpected frozen asset verification cell')
    asset_cells[0]['source'] = asset_cells[0]['source'].replace(
        'ASSET_MANIFEST, ASSET = matches[0]',
        'ASSET_MANIFEST, ASSET = matches[0]\n'
        f'assert hashlib.sha256(ASSET_MANIFEST.read_bytes()).hexdigest()=={ASSET_MANIFEST_SHA!r}, "Frozen asset manifest drift"')
    preflight = cells[-1]
    preflight['source'] = preflight['source'].replace('free_gib[:1]', 'free_gib')
    preflight['source'] += "\nassert len(devices)==2, 'Exactly two T4 replicas required'\n"
    runner = (HERE / 'submission_runner.py').read_text()
    cells.insert(-1, original.cell('code', original.writefile_cell('submission_runner.py', runner)))
    cells.insert(-1, original.cell('code', original.writefile_cell('vision_cache.py', cache_path.read_text())))
    cells.append(original.cell('code',
        "from datetime import datetime, timezone\n"
        "from submission_runner import run_replicas, verify_v8_sources\n"
        "assert RUN_INFERENCE is True\n"
        "verify_v8_sources(Path('/kaggle/working/inference_config.json'))\n"
        "OUT=Path('/kaggle/working')/('t4-submission-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))\n"
        "request={'config_path':'/kaggle/working/inference_config.json',\n"
        " 'data_dir':str(DATA),'base_dir':str(BASE),'adapter_dir':str(ADAPTER),\n"
        " 'output_root':str(OUT),'asset_verification_seconds':ASSET_VERIFY_SECONDS,\n"
        " 'dependency_install_seconds':DEPENDENCY_INSTALL_SECONDS,\n"
        f" 'vision_cache_sha256':{VISION_CACHE_SHA!r}}}\n"
        "request_path=Path('/kaggle/working/submission_request.json')\n"
        "request_path.write_text(json.dumps(request,indent=2))\n"
        "result=run_replicas(request_path,SESSION_DEADLINE_MONOTONIC)\n"
        "source=Path(result['final_submission_path'])\n"
        "assert sha256_file(source)==result['submission_sha256']\n"
        "assert time.monotonic()<SESSION_DEADLINE_MONOTONIC, 'Deadline reached before final export'\n"
        "os.link(source,Path('/kaggle/working/submission.csv'))\n"
        "print('INFERENCE_COMPLETE',json.dumps(result),flush=True)\n"))
    notebook['cells'] = cells
    notebook['metadata']['kaggle'].update(dataSources=[], isPrivate=True,
                                         isInternetEnabled=False, isGpuEnabled=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(notebook, indent=1))
    return {'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'frozen_commit': FROZEN_COMMIT, 'cells': len(cells),
            'vision_cache_sha256': VISION_CACHE_SHA, 'status': 'BUILT_NOT_EXECUTED'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
