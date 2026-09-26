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
            "# RSNA Knee — private MedGemma image baseline inference (T4 FP16)\n\n"
            "This notebook consumes only `test.csv`, `test_series.csv`, test DICOMs and the private offline model asset. "
            "It never reads reports, organizer labels or Qwen outputs. It uses the frozen image-pilot preprocessing, "
            "base revision and LoRA adapter. T4 uses NF4 double quantization with FP16 compute. The runner tests one T4 first, "
            "and after a CUDA OOM may try one explicit two-T4 map. CPU/disk offload is forbidden. Any unscored study aborts; "
            "no 0.5 fallback is emitted.\n\nExpected status after a compatible run: `KAGGLE_RUN_PASS`.",
        ),
        cell(
            "code",
            "import os, sys, subprocess, json, glob, time, hashlib\n"
            "import platform\n"
            "from pathlib import Path\n"
            "os.environ['HF_HUB_OFFLINE']='1'\n"
            "os.environ['TRANSFORMERS_OFFLINE']='1'\n"
            "os.environ['HF_DATASETS_OFFLINE']='1'\n"
            "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
            "os.environ['HF_HUB_DISABLE_TELEMETRY']='1'\n"
            "RUN_INFERENCE=True\n"
            "assert platform.machine() in ('x86_64','AMD64'), f'Unsupported wheel architecture: {platform.machine()}'\n"
            "assert sys.version_info[:2] in ((3,11),(3,12)), f'Offline wheelhouse supports Python 3.11/3.12, got {sys.version}'\n"
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
            "asset_verification_started=time.monotonic()\n"
            "root_resolved=ASSET_ROOT.resolve()\n"
            "for rel,row in ASSET.get('files',{}).items():\n"
            " path=(ASSET_ROOT/rel).resolve()\n"
            " assert root_resolved in path.parents, f'Unsafe manifest path: {rel}'\n"
            " assert path.is_file(), f'Missing asset: {rel}'\n"
            " assert path.stat().st_size==row['bytes'], f'Asset size mismatch: {rel}'\n"
            " hasher=hashlib.sha256()\n"
            " with path.open('rb') as handle:\n"
            "  for block in iter(lambda:handle.read(8*1024*1024),b''): hasher.update(block)\n"
            " digest=hasher.hexdigest()\n"
            " assert digest==row['sha256'], f'Asset checksum mismatch: {rel}'\n"
            "print('Verified',len(ASSET.get('files',{})),'offline asset files')\n"
            "ASSET_VERIFY_SECONDS=time.monotonic()-asset_verification_started\n"
            "wheelhouse=ASSET_ROOT/'wheels'\n"
            "assert wheelhouse.is_dir(), 'Offline wheelhouse missing'\n"
            "dependency_install_started=time.monotonic()\n"
            "subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-index','--no-deps',f'--find-links={wheelhouse}',\n"
            "                       '-r','/kaggle/working/requirements-kaggle.txt'])\n"
            "DEPENDENCY_INSTALL_SECONDS=time.monotonic()-dependency_install_started\n"
            "print('Offline dependencies installed from',wheelhouse,'in',round(DEPENDENCY_INSTALL_SECONDS,1),'s')",
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
            " audit_adapter, audit_base_model, validate_runtime_versions, validate_free_memory, sha256_file)\n"
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
            "free_gib=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]\n"
            "memory_readiness=validate_free_memory(free_gib[:1],CONFIG['minimum_free_gpu_gib_before_load'])\n"
            "versions=validate_runtime_versions(runtime,CONFIG)\n"
            "adapter_audit=audit_adapter(ADAPTER,CONFIG)\n"
            "base_audit=audit_base_model(BASE,CONFIG,ASSET)\n"
            "print(json.dumps({'data_dir':str(DATA),'test_studies':len(TEST),'runtime':runtime,\n"
            "                  'hardware':hardware,'free_memory_all_gpus_gib':free_gib,\n"
            "                  'single_gpu_free_memory_gate':memory_readiness,'versions':versions,\n"
            "                  'adapter':adapter_audit,'base':base_audit,\n"
            "                  'timing_seconds':{'asset_verification':ASSET_VERIFY_SECONDS,\n"
            "                   'dependency_install':DEPENDENCY_INSTALL_SECONDS}},indent=2))",
        ),
        cell(
            "code",
            "from datetime import datetime, timezone\n"
            "from submission_runtime import prepare_t4_teacher, run_inference\n"
            "assert RUN_INFERENCE is True\n"
            "run_name='medgemma-image-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n"
            "OUT=Path('/kaggle/working')/run_name\n"
            "teacher,load_seconds,diagnostic,placement_attempts=prepare_t4_teacher(\n"
            " CONFIG,SERIES,DATA,BASE,ADAPTER,str(TEST.StudyInstanceUID.iloc[0]),\n"
            " '/kaggle/working/t4_placement_attempts.json')\n"
            "frozen_runtime={'runtime_variant':CONFIG['runtime_variant'],'placement':teacher.placement,\n"
            " 'hf_device_map':teacher.device_map,'dtype_report':teacher.dtype_report,\n"
            " 'load_seconds':load_seconds,'diagnostic':diagnostic,'attempts':placement_attempts}\n"
            "frozen_json=json.dumps(frozen_runtime,sort_keys=True,indent=2)\n"
            "frozen_sha=hashlib.sha256(frozen_json.encode()).hexdigest()\n"
            "Path('/kaggle/working/t4_configuration_frozen.json').write_text(frozen_json)\n"
            "print('FROZEN_T4_CONFIG_SHA256',frozen_sha)\n"
            "print('FROZEN_T4_CONFIG',json.dumps({'placement':teacher.placement,\n"
            " 'hf_device_map':teacher.device_map,'dtype_report':teacher.dtype_report,\n"
            " 'load_seconds':load_seconds,'diagnostic':diagnostic,\n"
            " 'attempts':placement_attempts},indent=2))\n"
            "prediction,status=run_inference(CONFIG,TEST,SERIES,SAMPLE,DATA,BASE,ADAPTER,OUT,\n"
            " teacher,load_seconds,ASSET_VERIFY_SECONDS,DEPENDENCY_INSTALL_SECONDS,diagnostic)\n"
            "(OUT/'t4_configuration_frozen.json').write_text(frozen_json)\n"
            "(OUT/'placement_diagnostic.json').write_text(json.dumps({'attempts':placement_attempts},indent=2))\n"
            "submission=Path('/kaggle/working/submission.csv')\n"
            "assert not submission.exists(), 'Refusing to overwrite an existing submission.csv'\n"
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
                "isPrivate": True,
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
