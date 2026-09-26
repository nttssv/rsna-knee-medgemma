"""Package same-study vision memoization with the original offline V8 sources."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
V8 = HERE.parents[1] / 'kaggle_image_t4_v1'
REFERENCE_SHA = '1f3d823c4b1e689b252c041945440dcc3685d0e5810cb7fbd91583739ef739cd'
V8_BUILDER_SHA = 'a631b9dcd4b66380a63046c9bfad3e157284cd5d586e53f4cb25c5f9d5a0b6d3'


def build(output: Path, reference_csv: Path):
    if hashlib.sha256(reference_csv.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise ValueError('Use the exact downloaded Version 8 example CSV')
    source = V8/'scripts'/'build_notebook.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != V8_BUILDER_SHA:
        raise ValueError('Frozen Version 8 notebook builder changed')
    spec = importlib.util.spec_from_file_location('frozen_v8_builder', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'v8.ipynb'
        original.build(path)
        notebook = json.loads(path.read_text())
    cells = notebook['cells']
    cells[0] = original.cell('markdown',
        '# Private T4 same-study vision-cache diagnostic\n\n'
        'Two independent T4 workers; frozen Version 8 weights, arithmetic, processor, '
        'images and twelve target forwards. Only the identical same-study vision-tower '
        'output is reused, after exact pixel/context/weight checks. Projector and decoder '
        'still run for every target. A study transition or error clears the cache. '
        'No training, offload, model parallelism, retry or competition submission.\n\n'
        'Prospective gates: every one of 36 scores within absolute 1e-6 of Version 8 '
        '(rtol=0); every complete-study time strictly below 40 seconds for the goal. '
        'All measured studies start with a cold vision cache. GPU execution requires '
        'a new bounded user-authorized session; local preparation does not start it.')
    # Reuse V8 setup, pinned offline asset audit and numerical code byte-for-byte.
    cells = [c for c in cells[:-1] if "write_text(" not in c['source'] or
             "Path('/kaggle/working/isolated_runner.py')" not in c['source']]
    cells[1]['source'] = cells[1]['source'].replace('115*60', '25*60')
    preflight = cells[-1]
    preflight['source'] = preflight['source'].replace('free_gib[:1]', 'free_gib')
    preflight['source'] += "\nassert len(devices)==2, 'Exactly two T4 replicas required'\n"
    cells.insert(-1, original.cell('code', original.writefile_cell(
        'replica_runner.py', (HERE/'replica_runner.py').read_text())))
    cache_text = (HERE/'vision_cache.py').read_text()
    cache_sha = hashlib.sha256(cache_text.encode()).hexdigest()
    cells.insert(-1, original.cell('code', original.writefile_cell('vision_cache.py', cache_text)))
    cells.append(original.cell('code',
        "from datetime import datetime, timezone\n"
        "from replica_runner import run_replicas, verify_v8_sources\n"
        "verify_v8_sources(Path('/kaggle/working/inference_config.json'))\n"
        "REFERENCE=Path('/kaggle/working/version8_reference.csv')\n"
        f"REFERENCE.write_text({reference_csv.read_text()!r})\n"
        "OUT=Path('/kaggle/working')/('t4-vision-cache-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))\n"
        "request={'config_path':'/kaggle/working/inference_config.json',\n"
        " 'data_dir':str(DATA),'base_dir':str(BASE),'adapter_dir':str(ADAPTER),\n"
        " 'output_root':str(OUT),'asset_verification_seconds':ASSET_VERIFY_SECONDS,\n"
        " 'dependency_install_seconds':DEPENDENCY_INSTALL_SECONDS,\n"
        f" 'vision_cache_sha256':{cache_sha!r},\n"
        f" 'reference_csv':str(REFERENCE),'reference_sha256':{REFERENCE_SHA!r}}}\n"
        "request_path=Path('/kaggle/working/replica_request.json')\n"
        "request_path.write_text(json.dumps(request,indent=2))\n"
        "result=run_replicas(request_path,SESSION_DEADLINE_MONOTONIC)\n"
        "source=Path(result['final_submission_path'])\n"
        "assert sha256_file(source)==result['submission_sha256']\n"
        "os.link(source,Path('/kaggle/working/submission.csv'))\n"
        "print(result['status'],json.dumps(result),flush=True)\n"))
    notebook['cells'] = cells
    # Kaggle UI attaches the actual competition and existing private asset.
    notebook['metadata']['kaggle']['dataSources'] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Refusing to overwrite an earlier notebook build')
    output.write_text(json.dumps(notebook, indent=1))
    return {'path':str(output), 'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'reference_sha256':REFERENCE_SHA, 'cells':len(cells)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-csv', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.reference_csv), indent=2))
