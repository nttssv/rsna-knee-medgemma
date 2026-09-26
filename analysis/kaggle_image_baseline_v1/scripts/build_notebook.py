"""Build the private Kaggle notebook from reviewed text sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def writefile_cell(name: str, source: str) -> str:
    return f"from pathlib import Path\nPath('/kaggle/working/{name}').write_text({source!r})\nprint('Wrote {name}')"


def build(output: Path) -> dict:
    config_source = (ROOT / "configs" / "inference.json").read_text()
    requirements_source = (ROOT / "requirements-kaggle.txt").read_text()
    core_source = (HERE / "inference_core.py").read_text()
    runtime_source = (HERE / "submission_runtime.py").read_text()
    cells = [
        cell(
            "markdown",
            "# RSNA Knee — private MedGemma image baseline inference\n\n"
            "This notebook consumes only `test.csv`, `test_series.csv`, test DICOMs and the private offline model asset. "
            "It never reads reports, organizer labels or Qwen outputs. It uses the frozen image-pilot preprocessing, "
            "base revision and LoRA adapter. Any unscored study aborts the run; no 0.5 fallback is emitted.\n\n"
            "Expected status after a compatible run: `KAGGLE_RUN_PASS`. A hardware or asset failure is a blocker, not a submission.",
        ),
        cell(
            "code",
            "import os, sys, subprocess, json, glob, time\n"
            "from pathlib import Path\n"
            "os.environ['HF_HUB_OFFLINE']='1'\n"
            "os.environ['TRANSFORMERS_OFFLINE']='1'\n"
            "os.environ['HF_DATASETS_OFFLINE']='1'\n"
            "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
            "RUN_INFERENCE=True\n"
            "print('Internet-independent mode configured')",
        ),
        cell("code", writefile_cell("inference_config.json", config_source)),
        cell("code", writefile_cell("requirements-kaggle.txt", requirements_source)),
        cell(
            "code",
            "candidates=[]\n"
            "patterns=[\n"
            " '/kaggle/input/*/asset_manifest.json',\n"
            " '/kaggle/input/datasets/*/*/asset_manifest.json',\n"
            " '/kaggle/input/models/*/*/*/*/asset_manifest.json',\n"
            "]\n"
            "for pattern in patterns: candidates.extend(Path(p) for p in glob.glob(pattern))\n"
            "matches=[]\n"
            "for path in candidates:\n"
            " try:\n"
            "  manifest=json.loads(path.read_text())\n"
            "  if manifest.get('asset_contract')=='rsna-knee-medgemma-pilot-v1': matches.append((path,manifest))\n"
            " except Exception: pass\n"
            "assert len(matches)==1, f'Attach exactly one private pilot asset; found {len(matches)}'\n"
            "ASSET_MANIFEST, ASSET = matches[0]\n"
            "ASSET_ROOT=ASSET_MANIFEST.parent\n"
            "import hashlib\n"
            "root_resolved=ASSET_ROOT.resolve()\n"
            "for rel,row in ASSET.get('files',{}).items():\n"
            " path=(ASSET_ROOT/rel).resolve()\n"
            " assert root_resolved in path.parents, f'Unsafe manifest path: {rel}'\n"
            " assert path.is_file(), f'Missing asset: {rel}'\n"
            " assert path.stat().st_size==row['bytes'], f'Asset size mismatch: {rel}'\n"
            " digest=hashlib.sha256(path.read_bytes()).hexdigest()\n"
            " assert digest==row['sha256'], f'Asset checksum mismatch: {rel}'\n"
            "print('Verified',len(ASSET.get('files',{})),'offline asset files')\n"
            "wheelhouse=ASSET_ROOT/'wheels'\n"
            "assert wheelhouse.is_dir(), 'Offline wheelhouse missing'\n"
            "subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-index','--no-deps',f'--find-links={wheelhouse}',\n"
            "                       '-r','/kaggle/working/requirements-kaggle.txt'])\n"
            "print('Offline dependencies installed from',wheelhouse)",
        ),
        cell("code", writefile_cell("inference_core.py", core_source)),
        cell("code", writefile_cell("submission_runtime.py", runtime_source)),
        cell(
            "code",
            "import importlib, platform\n"
            "import numpy as np, pandas as pd, torch\n"
            "from importlib.metadata import version\n"
            "sys.path.insert(0,'/kaggle/working')\n"
            "from inference_core import (read_config, find_competition_dir, validate_tables, validate_hardware,\n"
            " audit_adapter, audit_base_model, validate_runtime_versions, sha256_file)\n"
            "CONFIG=read_config(Path('/kaggle/working/inference_config.json'))\n"
            "DATA=find_competition_dir(Path('/kaggle/input'))\n"
            "TEST,SERIES,SAMPLE=validate_tables(DATA)\n"
            "BASE=ASSET_ROOT/'base_model'; ADAPTER=ASSET_ROOT/'adapter'\n"
            "devices=[{'name':torch.cuda.get_device_name(i),\n"
            "          'capability_major':torch.cuda.get_device_capability(i)[0],\n"
            "          'capability_minor':torch.cuda.get_device_capability(i)[1],\n"
            "          'total_gib':torch.cuda.get_device_properties(i).total_memory/2**30}\n"
            "         for i in range(torch.cuda.device_count())]\n"
            "runtime={'python':platform.python_version(),'torch':torch.__version__,\n"
            "         'cuda':torch.version.cuda,'devices':devices,\n"
            "         'packages':{p:version(p) for p in ['transformers','peft','accelerate','bitsandbytes','pydicom','python-gdcm','sentencepiece','safetensors']}}\n"
            "hardware=validate_hardware(devices,CONFIG)\n"
            "versions=validate_runtime_versions(runtime,CONFIG)\n"
            "adapter_audit=audit_adapter(ADAPTER,CONFIG)\n"
            "base_audit=audit_base_model(BASE,CONFIG,ASSET)\n"
            "print(json.dumps({'data_dir':str(DATA),'test_studies':len(TEST),'runtime':runtime,\n"
            "                  'hardware':hardware,'versions':versions,'adapter':adapter_audit,'base':base_audit},indent=2))",
        ),
        cell(
            "code",
            "from datetime import datetime, timezone\n"
            "from submission_runtime import run_inference\n"
            "assert RUN_INFERENCE is True\n"
            "run_name='medgemma-image-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n"
            "OUT=Path('/kaggle/working')/run_name\n"
            "prediction,status=run_inference(CONFIG,TEST,SERIES,SAMPLE,DATA,BASE,ADAPTER,OUT)\n"
            "submission=Path('/kaggle/working/submission.csv')\n"
            "prediction.to_csv(submission,index=False)\n"
            "print(json.dumps(status,indent=2))\n"
            "print('KAGGLE_RUN_PASS',submission,sha256_file(submission))",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": ["competition:rsna-knee-abnormality-detection"],
                "dockerImageVersionId": None,
                "isInternetEnabled": False,
                "isGpuEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1))
    return {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "cells": len(cells),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
