"""Audit the private pilot checkpoint and its pinned base architecture locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from accelerate import init_empty_weights
from peft import PeftConfig
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_core import audit_adapter, audit_base_model, read_config, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    manifest_path = args.asset_root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    adapter_dir = args.asset_root / "adapter"
    base_dir = args.asset_root / "base_model"
    adapter = audit_adapter(adapter_dir, config)
    base = audit_base_model(base_dir, config, manifest)
    processor = AutoProcessor.from_pretrained(
        adapter_dir, local_files_only=True, use_fast=False
    )
    answers = {
        answer: processor.tokenizer.encode(answer, add_special_tokens=False)
        for answer in ["No", "Yes"]
    }
    if any(len(ids) != 1 for ids in answers.values()):
        raise RuntimeError("Answer-token boundary differs from the executed pilot")
    model_config = AutoConfig.from_pretrained(base_dir, local_files_only=True)
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(
            model_config, attn_implementation=config["attention_implementation"]
        )
    modules = {name for name, _ in model.named_modules()}
    missing_targets = [
        name
        for name in peft_config.target_modules
        if name not in modules and not any(module.endswith("." + name) for module in modules)
    ]
    if missing_targets:
        raise RuntimeError(f"Adapter targets absent from pinned architecture: {missing_targets[:5]}")
    print(
        json.dumps(
            {
                "status": "CHECKPOINT_STRUCTURE_PASS",
                "full_cuda_weight_load": "NOT_RUN_ON_LOCAL_APPLE_SILICON",
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "model_class": type(model).__name__,
                "processor_class": type(processor).__name__,
                "tokenizer_class": type(processor.tokenizer).__name__,
                "answer_token_ids": answers,
                "missing_adapter_targets": missing_targets,
                "adapter": adapter,
                "base": base,
                "asset_manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
