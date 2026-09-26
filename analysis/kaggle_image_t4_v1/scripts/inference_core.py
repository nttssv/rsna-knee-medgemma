"""Fail-closed contract helpers for the first image-only Kaggle submission."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct

import pandas as pd


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

REQUIRED_COMPETITION_FILES = (
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
)

REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

REQUIRED_BASE_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


class ContractError(RuntimeError):
    """A deterministic submission prerequisite was not met."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config["labels"] != LABELS:
        raise ContractError("Inference label order differs from the frozen schema")
    if config["fallback_policy"] != "fail_entire_run_on_unscored_study":
        raise ContractError("Unexpected fallback policy")
    return config


def safetensors_header(path: Path) -> dict:
    """Read only the JSON header; this verifies structure without loading tensors."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ContractError("Truncated Safetensors header")
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes <= 0 or header_bytes > 16 * 1024 * 1024:
            raise ContractError("Invalid Safetensors header length")
        header = json.loads(handle.read(header_bytes))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ContractError("Adapter contains no tensors")
    return tensors


def audit_adapter(adapter_dir: Path, config: dict) -> dict:
    adapter_dir = Path(adapter_dir)
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise ContractError(f"Missing adapter artifacts: {missing}")
    weight_path = adapter_dir / "adapter_model.safetensors"
    observed_sha = sha256_file(weight_path)
    if observed_sha != config["adapter_sha256"]:
        raise ContractError("Adapter weight SHA-256 mismatch")
    adapter = json.loads((adapter_dir / "adapter_config.json").read_text())
    if adapter.get("base_model_name_or_path") != config["model_id"]:
        raise ContractError("Adapter names a different base model")
    if adapter.get("peft_type") != "LORA" or adapter.get("r") != 8:
        raise ContractError("Adapter is not the frozen rank-8 LoRA")
    tensors = safetensors_header(weight_path)
    a_count = sum(".lora_A." in name for name in tensors)
    b_count = sum(".lora_B." in name for name in tensors)
    if a_count != 136 or b_count != 136 or len(tensors) != 272:
        raise ContractError("Unexpected LoRA tensor inventory")
    if not all(".lora_" in name for name in tensors):
        raise ContractError("Adapter contains non-LoRA tensors")
    return {
        "base_model": adapter["base_model_name_or_path"],
        "adapter_sha256": observed_sha,
        "tensor_count": len(tensors),
        "lora_a_count": a_count,
        "lora_b_count": b_count,
        "adapter_bytes": weight_path.stat().st_size,
    }


def audit_base_model(base_dir: Path, config: dict, manifest: dict | None = None) -> dict:
    base_dir = Path(base_dir)
    missing = [name for name in REQUIRED_BASE_FILES if not (base_dir / name).is_file()]
    if missing:
        raise ContractError(f"Missing pinned base-model artifacts: {missing}")
    model_config = json.loads((base_dir / "config.json").read_text())
    if model_config.get("model_type") != "gemma3":
        raise ContractError("Unexpected base model architecture")
    if model_config.get("text_config", {}).get("num_hidden_layers") != 34:
        raise ContractError("Unexpected base model layer count")
    if manifest is not None:
        if manifest.get("model_id") != config["model_id"]:
            raise ContractError("Asset manifest model ID mismatch")
        if manifest.get("model_revision") != config["model_revision"]:
            raise ContractError("Asset manifest revision mismatch")
        expected = manifest.get("files", {})
        for name in REQUIRED_BASE_FILES:
            row = expected.get(f"base_model/{name}")
            if not row:
                raise ContractError(f"Asset manifest does not bind base_model/{name}")
            path = base_dir / name
            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ContractError(f"Base-model integrity mismatch: {name}")
    return {
        "model_type": model_config["model_type"],
        "hidden_layers": model_config["text_config"]["num_hidden_layers"],
        "weight_bytes": sum((base_dir / name).stat().st_size for name in REQUIRED_BASE_FILES if name.endswith(".safetensors")),
    }


def find_competition_dir(input_root: Path) -> Path:
    input_root = Path(input_root)
    candidates = [
        input_root / "competitions" / "rsna-knee-abnormality-detection",
        input_root / "rsna-knee-abnormality-detection",
    ]
    matches = [p for p in candidates if all((p / n).is_file() for n in REQUIRED_COMPETITION_FILES)]
    if len(matches) != 1:
        raise ContractError(f"Expected one competition input, found {len(matches)}")
    return matches[0]


def validate_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    test = pd.read_csv(data_dir / "test.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(
        data_dir / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    expected_columns = ["StudyInstanceUID", *LABELS]
    if list(test.columns) != ["StudyInstanceUID"]:
        raise ContractError("test.csv must contain only StudyInstanceUID")
    if list(sample.columns) != expected_columns:
        raise ContractError("sample_submission.csv column order changed")
    if not test["StudyInstanceUID"].is_unique or test["StudyInstanceUID"].isna().any():
        raise ContractError("test.csv study IDs must be unique and non-null")
    if not sample["StudyInstanceUID"].is_unique:
        raise ContractError("sample submission IDs are not unique")
    if set(sample["StudyInstanceUID"]) != set(test["StudyInstanceUID"]):
        raise ContractError("Sample/test study ID sets differ")
    required_series = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }
    if set(series.columns) != required_series:
        raise ContractError("test_series.csv schema changed")
    if not set(series["StudyInstanceUID"]).issubset(set(test["StudyInstanceUID"])):
        raise ContractError("test_series.csv contains non-test study IDs")
    if set(series["StudyInstanceUID"]) != set(test["StudyInstanceUID"]):
        raise ContractError("At least one test study has no series descriptor")
    return test, series, sample


def validate_hardware(devices: list[dict], config: dict) -> dict:
    if not devices:
        raise ContractError("No CUDA GPU is visible")
    if len(devices) not in config["allowed_gpu_count"]:
        raise ContractError(f"Expected 1 or 2 Kaggle T4 GPUs, observed {len(devices)}")
    allowed_fragments = config["allowed_gpu_name_fragments"]
    if any(not any(fragment in str(device.get("name", "")) for fragment in allowed_fragments) for device in devices):
        raise ContractError("This runtime is approved only for NVIDIA Tesla T4 devices")
    for index, device in enumerate(devices):
        if device.get("capability_major") != 7 or device.get("capability_minor") != 5:
            raise ContractError(f"GPU {index} is not a T4 compute capability 7.5 device")
        if float(device.get("total_gib", 0)) < float(config["minimum_gpu_gib"]):
            raise ContractError(f"GPU {index} is below the configured T4 memory guard")
    first = devices[0]
    return {
        "selected_device": 0,
        "visible_devices": len(devices),
        "device_memory_gib": [float(device["total_gib"]) for device in devices],
        **first,
    }


def validate_runtime_versions(runtime: dict, config: dict) -> dict:
    """Check package pins and T4 CUDA runtime; record the Kaggle Torch build."""
    expected = config["expected_training_versions"]
    torch_version = str(runtime.get("torch", ""))
    cuda_version = str(runtime.get("cuda", ""))
    if not torch_version or "+cu" not in torch_version:
        raise ContractError("Expected a CUDA-enabled Torch build from the Kaggle image")
    torch_base = torch_version.split("+", 1)[0].split(".")
    torch_pair = tuple(int(part) for part in torch_base[:2])
    if torch_pair < (2, 8):
        raise ContractError(f"Torch 2.8 or newer is required, observed {torch_version}")
    if not cuda_version.startswith("12.8"):
        raise ContractError(f"Expected CUDA 12.8 for the pinned T4 wheelhouse, observed {cuda_version}")
    observed_packages = runtime.get("packages", {})
    package_map = {name: name for name in expected}
    for config_name, package_name in package_map.items():
        if observed_packages.get(package_name) != expected[config_name]:
            raise ContractError(
                f"Package version drift for {package_name}: expected "
                f"{expected[config_name]}, observed {observed_packages.get(package_name)}"
            )
    return {
        "torch": torch_version,
        "cuda": cuda_version,
        "packages": {name: observed_packages[name] for name in package_map.values()},
    }


def yes_probability_from_logits(logits, no_token_id: int, yes_token_id: int):
    """Validate native selected logits, then calculate stable FP32 softmax."""
    import torch

    pair = logits[[no_token_id, yes_token_id]]
    if not bool(torch.isfinite(pair).all()):
        raise ContractError("Native No/Yes logits contain NaN or Inf")
    pair_fp32 = pair.float()
    probabilities = pair_fp32.softmax(0)
    if not bool(torch.isfinite(probabilities).all()):
        raise ContractError("FP32 No/Yes probabilities contain NaN or Inf")
    return float(pair_fp32[0].cpu()), float(pair_fp32[1].cpu()), float(probabilities[1].cpu())


def validate_free_memory(free_gib_by_device: list[float], minimum_free_gib: float) -> dict:
    if not free_gib_by_device:
        raise ContractError("No GPU memory values were observed")
    if any(float(value) < float(minimum_free_gib) for value in free_gib_by_device):
        raise ContractError(
            f"Insufficient free GPU memory before model load: {free_gib_by_device}; "
            f"requires {minimum_free_gib} GiB on each participating GPU"
        )
    return {"free_gib_by_device": [float(value) for value in free_gib_by_device]}


def validate_submission(frame: pd.DataFrame, test: pd.DataFrame) -> dict:
    expected_columns = ["StudyInstanceUID", *LABELS]
    if list(frame.columns) != expected_columns:
        raise ContractError("submission.csv columns/order are incorrect")
    if len(frame) != len(test):
        raise ContractError("submission.csv row count differs from test.csv")
    if not frame["StudyInstanceUID"].is_unique:
        raise ContractError("submission.csv contains duplicate IDs")
    if set(frame["StudyInstanceUID"]) != set(test["StudyInstanceUID"]):
        raise ContractError("submission.csv IDs differ from test.csv")
    values = frame[LABELS].to_numpy(dtype=float)
    if not all(math.isfinite(float(x)) for x in values.ravel()):
        raise ContractError("submission.csv contains NaN or non-finite scores")
    if ((values < 0) | (values > 1)).any():
        raise ContractError("submission scores must be in [0, 1]")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "unique_ids": int(frame["StudyInstanceUID"].nunique()),
        "minimum_score": float(values.min()),
        "maximum_score": float(values.max()),
    }


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)
