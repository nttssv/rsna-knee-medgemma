import importlib.util
import json
from pathlib import Path
import struct
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from inference_core import (  # noqa: E402
    LABELS,
    REQUIRED_BASE_FILES,
    ContractError,
    audit_adapter,
    audit_base_model,
    read_config,
    sha256_file,
    validate_hardware,
    validate_runtime_versions,
    validate_submission,
    validate_tables,
)


CONFIG_PATH = ROOT / "configs" / "inference.json"


def config():
    return read_config(CONFIG_PATH)


def write_safetensors(path: Path, names):
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, (i + 1) * 4]} for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (len(names) * 4))


def test_frozen_real_adapter_is_present_and_structurally_valid():
    adapter = ROOT.parents[1] / "state" / "runs" / "20260913T135620Z" / "teacher_adapter"
    if not adapter.is_dir():
        pytest.skip("Private pilot adapter is intentionally absent from public clones")
    result = audit_adapter(adapter, config())
    assert result["tensor_count"] == 272
    assert result["lora_a_count"] == result["lora_b_count"] == 136
    assert result["adapter_sha256"] == config()["adapter_sha256"]


def test_hardware_guard_does_not_pool_two_t4s():
    t4s = [
        {"name": "Tesla T4", "capability_major": 7, "capability_minor": 5, "total_gib": 15.0},
        {"name": "Tesla T4", "capability_major": 7, "capability_minor": 5, "total_gib": 15.0},
    ]
    with pytest.raises(ContractError, match="native BF16"):
        validate_hardware(t4s, config())


def test_hardware_guard_accepts_one_native_bf16_48_gib_gpu():
    observed = validate_hardware(
        [{"name": "NVIDIA RTX A6000", "capability_major": 8, "capability_minor": 6, "total_gib": 47.5}],
        config(),
    )
    assert observed["selected_device"] == 0


def test_runtime_version_drift_fails_closed():
    expected = config()["expected_training_versions"]
    runtime = {
        "torch": expected["torch"],
        "cuda": expected["cuda"],
        "packages": {
            "transformers": expected["transformers"],
            "peft": expected["peft"],
            "bitsandbytes": expected["bitsandbytes"],
            "pydicom": expected["pydicom"],
        },
    }
    assert validate_runtime_versions(runtime, config())["cuda"] == "12.8"
    runtime["torch"] = "2.7.0+cu126"
    with pytest.raises(ContractError, match="Torch version drift"):
        validate_runtime_versions(runtime, config())


def test_tables_and_submission_are_bound_to_actual_test_ids(tmp_path):
    ids = ["test-z", "test-a", "test-q", "test-extra"]
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame(
        {
            "StudyInstanceUID": ids,
            "SeriesInstanceUID": [f"series-{i}" for i in range(4)],
            "Fluid_Sensitive": [1, 0, 1, 0],
            "Fat_Suppression": [1, 0, 0, 1],
            "Anatomical_Plane": ["Axial", "Sagittal", "Coronal", "Axial"],
        }
    ).to_csv(tmp_path / "test_series.csv", index=False)
    sample = pd.DataFrame({"StudyInstanceUID": ids, **{label: [0.5] * 4 for label in LABELS}})
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    test, _, _ = validate_tables(tmp_path)
    prediction = pd.DataFrame(
        {"StudyInstanceUID": ids, **{label: [0.1, 0.2, 0.3, 0.4] for label in LABELS}}
    )
    result = validate_submission(prediction, test)
    assert result["rows"] == 4
    assert result["unique_ids"] == 4


@pytest.mark.parametrize("bad", [float("nan"), -0.01, 1.01])
def test_invalid_scores_fail(tmp_path, bad):
    test = pd.DataFrame({"StudyInstanceUID": ["x"]})
    prediction = pd.DataFrame({"StudyInstanceUID": ["x"], **{label: [0.5] for label in LABELS}})
    prediction.loc[0, "ACL"] = bad
    with pytest.raises(ContractError):
        validate_submission(prediction, test)


def test_base_asset_manifest_binds_every_file(tmp_path):
    base = tmp_path / "base_model"
    base.mkdir()
    for name in REQUIRED_BASE_FILES:
        if name == "config.json":
            (base / name).write_text(json.dumps({"model_type": "gemma3", "text_config": {"num_hidden_layers": 34}}))
        else:
            (base / name).write_bytes((name + "\n").encode())
    manifest = {
        "model_id": config()["model_id"],
        "model_revision": config()["model_revision"],
        "files": {
            f"base_model/{name}": {"bytes": (base / name).stat().st_size, "sha256": sha256_file(base / name)}
            for name in REQUIRED_BASE_FILES
        },
    }
    assert audit_base_model(base, config(), manifest)["hidden_layers"] == 34
    (base / "config.json").write_text("{}")
    with pytest.raises(ContractError):
        audit_base_model(base, config(), manifest)


def test_submission_runtime_has_no_report_or_qwen_input_path():
    source = (SCRIPTS / "submission_runtime.py").read_text()
    assert "train.csv" not in source
    assert "organizer" not in source.lower()
    assert "qwen" not in source.lower()
    assert "['Report']" not in source and '["Report"]' not in source


def test_notebook_builder_emits_private_offline_notebook(tmp_path):
    spec = importlib.util.spec_from_file_location("builder", SCRIPTS / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "inference.ipynb"
    result = module.build(out)
    notebook = json.loads(out.read_text())
    assert result["cells"] == len(notebook["cells"])
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False
    joined = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "RUN_INFERENCE=True" in joined
    assert "submission.csv" in joined
    assert "Asset checksum mismatch" in joined
    assert "sample_submission.csv" not in joined or "find_competition_dir" in joined
