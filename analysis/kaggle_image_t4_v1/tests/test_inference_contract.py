import importlib.util
import ast
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

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
    validate_free_memory,
    yes_probability_from_logits,
    validate_submission,
    validate_tables,
)
from submission_runtime import validate_device_map  # noqa: E402


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


def test_hardware_guard_accepts_kaggle_t4_pair_without_pooling():
    t4s = [
        {"name": "Tesla T4", "capability_major": 7, "capability_minor": 5, "total_gib": 15.0},
        {"name": "Tesla T4", "capability_major": 7, "capability_minor": 5, "total_gib": 15.0},
    ]
    observed = validate_hardware(t4s, config())
    assert observed["visible_devices"] == 2
    assert observed["device_memory_gib"] == [15.0, 15.0]


def test_hardware_guard_rejects_unlisted_gpu():
    with pytest.raises(ContractError, match="Tesla T4"):
        validate_hardware(
            [{"name": "NVIDIA A100-SXM4-40GB", "capability_major": 8, "capability_minor": 0, "total_gib": 40.0}],
            config(),
        )


def test_device_map_uses_only_explicit_gpus_and_keeps_blocks_whole():
    assert validate_device_map({"": 0}, "single_gpu") == {"": "0"}
    dual = {
        "model.vision_tower.vision_model.encoder.layers.0": 0,
        "model.language_model.layers.0": 0,
        "model.language_model.layers.17": 1,
        "lm_head": 1,
    }
    assert set(validate_device_map(dual, "two_gpu").values()) == {"0", "1"}
    with pytest.raises(ContractError, match="CPU/disk offload"):
        validate_device_map({"": "cpu"}, "single_gpu")
    with pytest.raises(ContractError, match="block was split"):
        validate_device_map({"model.language_model.layers.0.self_attn": 0, "lm_head": 1}, "two_gpu")


def test_frozen_science_configuration_matches_baseline_v1():
    base_config = json.loads((ROOT.parents[1] / "analysis/kaggle_image_baseline_v1/configs/inference.json").read_text())
    t4_config = config()
    for key in ["snapshot_commit", "model_id", "model_revision", "adapter_sha256", "image_size", "slices_per_plane", "labels", "attention_implementation"]:
        assert t4_config[key] == base_config[key]
    base_runtime = (ROOT.parents[1] / "analysis/kaggle_image_baseline_v1/scripts/submission_runtime.py").read_text()
    t4_runtime = (SCRIPTS / "submission_runtime.py").read_text()
    for source in [base_runtime, t4_runtime]:
        tree = ast.parse(source)
        definitions = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFINITIONS" for t in node.targets))
        planes = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PLANES" for t in node.targets))
        if source == base_runtime:
            expected_definitions, expected_planes = definitions, planes
        else:
            assert definitions == expected_definitions
            assert planes == expected_planes


def test_free_memory_guard_requires_headroom():
    assert validate_free_memory([13.0], 12.0)["free_gib_by_device"] == [13.0]
    with pytest.raises(ContractError, match="Insufficient free"):
        validate_free_memory([13.0, 11.9], 12.0)


def test_hardware_guard_accepts_one_t4_without_claiming_bf16():
    observed = validate_hardware(
        [{"name": "Tesla T4", "capability_major": 7, "capability_minor": 5, "total_gib": 15.0}],
        config(),
    )
    assert observed["selected_device"] == 0


def test_runtime_version_drift_fails_closed():
    expected = config()["expected_training_versions"]
    runtime = {
        "torch": "2.10.0+cu128",
        "cuda": "12.8",
        "packages": expected,
    }
    runtime["packages"] = config()["expected_training_versions"]
    assert validate_runtime_versions(runtime, config())["cuda"] == "12.8"
    runtime["cuda"] = "12.6"
    with pytest.raises(ContractError, match="Expected CUDA 12.8"):
        validate_runtime_versions(runtime, config())
    runtime["cuda"] = "12.8"
    runtime["torch"] = "2.7.1+cu128"
    with pytest.raises(ContractError, match="Torch 2.8 or newer"):
        validate_runtime_versions(runtime, config())


def test_native_nonfinite_logits_are_not_hidden():
    import torch

    logits = torch.tensor([0.0, 2.0, float("inf")], dtype=torch.float16)
    no_logit, yes_logit, score = yes_probability_from_logits(logits, 0, 1)
    assert no_logit == 0.0 and yes_logit == 2.0
    assert 0.88 < score < 0.89
    with pytest.raises(ContractError, match="Native No/Yes logits"):
        yes_probability_from_logits(torch.tensor([0.0, float("inf")]), 0, 1)


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
    assert notebook["metadata"]["kaggle"]["isPrivate"] is True
    joined = "\n".join(cell["source"] for cell in notebook["cells"])
    for index, entry in enumerate(notebook["cells"]):
        if entry["cell_type"] == "code":
            compile(entry["source"], f"cell-{index}", "exec")
    assert "RUN_INFERENCE=True" in joined
    assert "submission.csv" in joined
    assert "Asset checksum mismatch" in joined
    assert "prepare_t4_teacher" in joined
    assert "Refusing to overwrite an existing submission.csv" in joined
    assert "RUN_INFERENCE=True" in joined
    assert "sample_submission.csv" not in joined or "find_competition_dir" in joined


@pytest.mark.parametrize("drift", [None, "huggingface-hub", "tokenizers"])
def test_generated_environment_cell_records_all_pins_before_real_validator(tmp_path, drift):
    """Exercise the generated inventory, not a manually complete test dictionary."""
    from build_notebook import build

    out = tmp_path / "notebook.ipynb"
    build(out)
    notebook = json.loads(out.read_text())
    source = next(c["source"] for c in notebook["cells"] if "runtime={'python'" in c["source"])
    nodes = ast.parse(source).body
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(nodes) if assigns(n, "devices"))
    end = next(i for i, n in enumerate(nodes) if assigns(n, "versions")) + 1
    generated = compile(ast.Module(body=nodes[start:end], type_ignores=[]), "generated-environment-cell", "exec")
    cfg = config()
    queried = []
    def observed_version(name):
        queried.append(name)
        return "0.0.0" if name == drift else cfg["expected_training_versions"][name]
    cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_name=lambda i: "Tesla T4",
        get_device_capability=lambda i: (7, 5),
        get_device_properties=lambda i: SimpleNamespace(total_memory=15 * 2**30),
        mem_get_info=lambda i: (14 * 2**30, 15 * 2**30),
    )
    namespace = dict(
        CONFIG=cfg, torch=SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"), cuda=cuda),
        platform=SimpleNamespace(python_version=lambda: "3.12.12"), version=observed_version,
        Path=lambda path: tmp_path / Path(path).name, json=json,
        ASSET_VERIFY_SECONDS=1.25, DEPENDENCY_INSTALL_SECONDS=2.5,
        validate_hardware=validate_hardware, validate_free_memory=validate_free_memory,
        validate_runtime_versions=validate_runtime_versions,
    )
    if drift:
        with pytest.raises(ContractError, match=f"Package version drift for {drift}"):
            exec(generated, namespace)
    else:
        exec(generated, namespace)
        assert namespace["versions"]["packages"] == cfg["expected_training_versions"]
    assert set(queried) == set(cfg["expected_training_versions"])
    recorded = json.loads((tmp_path / "environment_observed.json").read_text())
    assert recorded["runtime"]["packages"] == namespace["runtime"]["packages"]
    assert recorded["free_memory_all_gpus_gib"] == [14.0, 14.0]
    assert recorded["timing_seconds"] == {"asset_verification": 1.25, "dependency_install": 2.5}
