"""Image-only MedGemma pilot inference for the RSNA Knee code competition.

This module intentionally accepts imaging inputs only. Any study that
cannot be scored aborts the run; it never substitutes neutral scores.
"""

from __future__ import annotations

import json
import gc
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

from inference_core import LABELS, ContractError, atomic_json, validate_submission


PLANES = ["Sagittal", "Coronal", "Axial"]
DEFINITIONS = {
    "ACL": "high-grade partial (>50%) or complete anterior cruciate ligament tear; exclude isolated low-grade sprain or degeneration",
    "MCL": "high-grade acute partial or complete medial collateral ligament tear; exclude low-grade sprain or chronic healed injury",
    "Medial Meniscus": "definite medial meniscus tear with surface-reaching signal on at least two images or abnormal morphology; exclude isolated intrameniscal degeneration",
    "Lateral Meniscus": "definite lateral meniscus tear with surface-reaching signal on at least two images or abnormal morphology; exclude isolated intrameniscal degeneration",
    "Medial OA": "medial compartment cartilage loss of high grade (>50% thickness) over a moderate or large area (at least 1 cm)",
    "Lateral OA": "lateral compartment cartilage loss of high grade (>50% thickness) over a moderate or large area (at least 1 cm)",
    "PF OA": "patellofemoral cartilage loss of high grade (>50% thickness) over a moderate or large area (at least 1 cm)",
    "Effusion": "moderate or large joint effusion",
    "Synovitis": "synovitis with synovial thickening or inflammation",
    "Baker's": "moderate or large popliteal (Baker) cyst",
    "Contusion": "traumatic bone marrow edema (bone contusion) without a fracture line; exclude degenerative edema",
    "Fracture": "an acute fracture",
}


class Imaging:
    def __init__(self, data_dir: Path, slices_per_plane: int = 2):
        self.data_dir = Path(data_dir)
        self.slices_per_plane = slices_per_plane
        self.issues: list[dict] = []

    @staticmethod
    def _imports():
        import pydicom
        from pydicom.pixels import apply_modality_lut

        return pydicom, apply_modality_lut

    def header(self, path):
        pydicom, _ = self._imports()
        tags = [
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "InstanceNumber",
            "PixelSpacing",
            "PatientID",
            "Manufacturer",
            "ManufacturerModelName",
            "MagneticFieldStrength",
        ]
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)

    @staticmethod
    def rank_series(rows):
        rows = rows.copy()
        rows["_rank"] = (
            pd.to_numeric(rows["Fluid_Sensitive"], errors="coerce").fillna(0) * 2
            + pd.to_numeric(rows["Fat_Suppression"], errors="coerce").fillna(0)
        )
        return rows.sort_values(["_rank", "SeriesInstanceUID"], ascending=[False, True])

    def ordered_paths(self, folder: Path, uid: str):
        paths = sorted(folder.glob("*.dcm"))
        if not paths:
            raise ValueError("No DICOM slices")
        headers = [self.header(path) for path in paths]
        if all(hasattr(h, "ImagePositionPatient") and hasattr(h, "ImageOrientationPatient") for h in headers):
            orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
            normal = np.cross(orientation[:3], orientation[3:])
            if np.linalg.norm(normal) < 0.9:
                raise ValueError("Invalid slice orientation")
            if not all(np.allclose(np.asarray(h.ImageOrientationPatient, float), orientation, atol=0.05) for h in headers):
                raise ValueError("Mixed orientation within a series")
            positions = [float(np.dot(np.asarray(h.ImagePositionPatient, float), normal)) for h in headers]
        elif all(hasattr(h, "InstanceNumber") for h in headers):
            positions = [float(h.InstanceNumber) for h in headers]
            self.issues.append({"study": uid, "issue": "InstanceNumber ordering fallback"})
        else:
            raise ValueError("Neither geometry nor complete InstanceNumber is available")
        order = np.argsort(positions, kind="stable")
        return [paths[index] for index in order]

    @staticmethod
    def canonical_pixels(ds):
        _, apply_modality_lut = Imaging._imports()
        array = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected one grayscale slice, received {array.shape}")
        array = np.nan_to_num(array, nan=0, posinf=0, neginf=0)
        spacing = np.asarray(getattr(ds, "PixelSpacing", [1, 1]), dtype=float)
        if not np.isfinite(spacing).all() or (spacing <= 0).any():
            spacing = np.ones(2)
        orientation = getattr(ds, "ImageOrientationPatient", None)
        if orientation is not None:
            col, row = np.array(orientation[:3], float), np.array(orientation[3:], float)
            if np.argmax(abs(col)) > np.argmax(abs(row)):
                array = array.T
                col, row = row, col
                spacing = spacing[::-1]
            if col[np.argmax(abs(col))] < 0:
                array = np.fliplr(array)
            if row[np.argmax(abs(row))] < 0:
                array = np.flipud(array)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            array = -array
        return array.copy(), spacing

    def render_series(self, paths, size):
        pydicom, _ = self._imports()
        indices = np.unique(
            np.linspace(
                0.25 * (len(paths) - 1),
                0.75 * (len(paths) - 1),
                min(self.slices_per_plane, len(paths)),
            ).round().astype(int)
        )
        decoded = [self.canonical_pixels(pydicom.dcmread(paths[i])) for i in indices]
        values = np.concatenate([a.ravel()[:: max(1, a.size // 10000)] for a, _ in decoded])
        low, high = np.percentile(values, [0.5, 99.5])
        if high <= low:
            raise ValueError("Constant-intensity series")
        rendered = []
        for array, spacing in decoded:
            array = np.uint8(np.clip((array - low) / (high - low), 0, 1) * 255)
            physical_h, physical_w = np.array(array.shape) * spacing
            new_h = max(1, round(size * physical_h / max(physical_h, physical_w)))
            new_w = max(1, round(size * physical_w / max(physical_h, physical_w)))
            image = Image.fromarray(array).resize((new_w, new_h), Image.Resampling.BILINEAR)
            canvas = Image.new("L", (size, size))
            canvas.paste(image, ((size - new_w) // 2, (size - new_h) // 2))
            rendered.append(np.asarray(canvas))
        return rendered, indices / max(1, len(paths) - 1)

    def study_images(self, uid: str, descriptors: pd.DataFrame, size: int):
        count = len(PLANES) * self.slices_per_plane
        images = np.zeros((count, size, size), np.uint8)
        mask = np.zeros(count, bool)
        metadata = np.zeros((count, 6), np.float32)
        rows = descriptors[descriptors.StudyInstanceUID == uid]
        for plane_index, plane in enumerate(PLANES):
            choices = self.rank_series(rows[rows.Anatomical_Plane.astype(str).str.lower() == plane.lower()])
            for _, row in choices.iterrows():
                try:
                    paths = self.ordered_paths(
                        self.data_dir / "test_series" / uid / row.SeriesInstanceUID,
                        uid,
                    )
                    rendered, positions = self.render_series(paths, size)
                    section = slice(
                        plane_index * self.slices_per_plane,
                        plane_index * self.slices_per_plane + len(rendered),
                    )
                    images[section] = np.stack(rendered)
                    mask[section] = True
                    metadata[section, plane_index] = 1
                    metadata[section, 3] = positions
                    metadata[section, 4] = float(row.Fluid_Sensitive) if pd.notna(row.Fluid_Sensitive) else 0
                    metadata[section, 5] = float(row.Fat_Suppression) if pd.notna(row.Fat_Suppression) else 0
                    break
                except Exception as error:
                    self.issues.append({
                        "study": uid,
                        "issue": f"{plane}: {type(error).__name__}: {str(error)[:160]}",
                    })
        if not mask.any():
            raise ContractError("No usable MRI series for a test study")
        return images, mask, metadata


class ImageTeacher:
    def __init__(self, config: dict, series: pd.DataFrame, data_dir: Path, base_dir: Path, adapter_dir: Path, placement: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.config = config
        self.series = series
        self.placement = placement
        self.imaging = Imaging(data_dir, config["slices_per_plane"])
        self.images: dict[str, list] = {}
        self.preprocessing_seconds = 0.0
        self.inference_seconds = 0.0
        self.processor = AutoProcessor.from_pretrained(
            adapter_dir, local_files_only=True, use_fast=False
        )
        self.processor.tokenizer.padding_side = "right"
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        device_map, max_memory = self._device_map(base_dir, placement, config, torch)
        model = AutoModelForImageTextToText.from_pretrained(
            base_dir,
            local_files_only=True,
            quantization_config=quantization,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            attn_implementation=config["attention_implementation"],
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model.config.use_cache = False
        if not getattr(model, "peft_config", None):
            raise ContractError("The audited LoRA adapter was not attached to the model")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.device_map = {str(key): str(value) for key, value in model.hf_device_map.items()}
        validate_device_map(self.device_map, placement)
        self.input_device = model.get_input_embeddings().weight.device
        self.dtype_report = self._dtype_report(model, torch)
        self.answer_ids = []
        for answer in ["No", "Yes"]:
            ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ContractError("Answer tokenization changed")
            self.answer_ids.append(ids[0])

    @staticmethod
    def _device_map(base_dir: Path, placement: str, config: dict, torch):
        if placement == "single_gpu":
            return {"": 0}, {0: config["single_gpu_max_memory"]}
        if placement != "two_gpu":
            raise ContractError(f"Unknown device placement: {placement}")
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig, AutoModelForImageTextToText

        cfg = AutoConfig.from_pretrained(base_dir, local_files_only=True)
        with init_empty_weights():
            empty = AutoModelForImageTextToText.from_config(
                cfg,
                attn_implementation="eager",
            )
        empty.tie_weights()
        mapping = infer_auto_device_map(
            empty,
            max_memory={
                0: config["dual_gpu_max_memory_each"],
                1: config["dual_gpu_max_memory_each"],
            },
            no_split_module_classes=empty._no_split_modules,
            dtype=torch.float16,
            offload_buffers=False,
        )
        devices = {str(device) for device in mapping.values()}
        if devices != {"0", "1"}:
            raise ContractError(f"Dual-T4 placement did not distribute modules across both devices: {mapping}")
        # infer_auto_device_map respects Gemma3DecoderLayer and SiglipEncoderLayer
        # no-split classes. Fail if any decoder/vision block key is split below a block.
        validate_device_map(mapping, "two_gpu")
        return dict(mapping), {
            0: config["dual_gpu_max_memory_each"],
            1: config["dual_gpu_max_memory_each"],
        }

    @staticmethod
    def _dtype_report(model, torch):
        import bitsandbytes as bnb

        quant_param_ids = set()
        for module in model.modules():
            if isinstance(module, bnb.nn.Linear4bit):
                quant_param_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
        groups = {"quantized_layers": {}, "vision_encoder": {}, "lora": {}, "other": {}}
        for name, parameter in model.named_parameters():
            if id(parameter) in quant_param_ids:
                group = "quantized_layers"
            elif "lora_" in name.lower():
                group = "lora"
            elif "vision_tower" in name:
                group = "vision_encoder"
            else:
                group = "other"
            key = f"{parameter.dtype}@{parameter.device}"
            row = groups[group].setdefault(key, {"tensor_count": 0, "parameter_count": 0})
            row["tensor_count"] += 1
            row["parameter_count"] += parameter.numel()
        return groups

    def messages(self, uid: str, label: str):
        if uid not in self.images:
            started = time.monotonic()
            arrays, mask, metadata = self.imaging.study_images(uid, self.series, self.config["image_size"])
            self.images[uid] = [
                (Image.fromarray(arrays[i]).convert("RGB"), metadata[i])
                for i in np.flatnonzero(mask)
            ]
            self.preprocessing_seconds += time.monotonic() - started
        content = [{
            "type": "text",
            "text": "These are ordered sampled slices from one knee MRI study. Some anatomy may not be visible.",
        }]
        for image, metadata in self.images[uid]:
            content.extend([
                {
                    "type": "text",
                    "text": (
                        f"{PLANES[int(np.argmax(metadata[:3]))]}, slice position {metadata[3]:.2f}, "
                        f"fluid sensitive {int(metadata[4])}, fat suppression {int(metadata[5])}."
                    ),
                },
                {"type": "image", "image": image},
            ])
        content.append({
            "type": "text",
            "text": f"Is there {DEFINITIONS[label]}? Answer with only Yes or No.",
        })
        return [{"role": "user", "content": content}]

    @staticmethod
    def _on_gpu(batch, torch, device):
        return {
            key: value.to(device, dtype=torch.float16) if value.is_floating_point() else value.to(device)
            for key, value in batch.items()
        }

    def score_logits(self, uid: str, label: str):
        torch = self.torch
        encoded = self.processor.apply_chat_template(
            self.messages(uid, label),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_dtypes = {key: str(value.dtype) for key, value in encoded.items()}
        batch = self._on_gpu(encoded, torch, self.input_device)
        model_input_dtypes = {key: str(value.dtype) for key, value in batch.items()}
        inference_started = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1]
        self.inference_seconds += time.monotonic() - inference_started
        from inference_core import yes_probability_from_logits

        try:
            no_logit, yes_logit, yes_probability = yes_probability_from_logits(
                logits, self.answer_ids[0], self.answer_ids[1]
            )
        except ContractError as error:
            raise ContractError(f"{error} for {uid}/{label}") from error
        result = {
            "native_no_logit": no_logit,
            "native_yes_logit": yes_logit,
            "yes_probability": yes_probability,
            "native_output_logits_dtype": str(logits.dtype),
            "input_dtypes_before_transfer": input_dtypes,
            "model_input_dtypes": model_input_dtypes,
            "input_device": str(self.input_device),
        }
        del encoded, batch, logits
        return result

    def diagnostic(self, uid: str):
        torch = self.torch
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        first = self.score_logits(uid, LABELS[0])
        second = self.score_logits(uid, LABELS[0])
        max_logit_delta = max(
            abs(first["native_no_logit"] - second["native_no_logit"]),
            abs(first["native_yes_logit"] - second["native_yes_logit"]),
        )
        score_delta = abs(first["yes_probability"] - second["yes_probability"])
        if max_logit_delta > 0.02 or score_delta > 1e-3:
            raise ContractError(
                f"Repeatability failed: max_logit_delta={max_logit_delta}, score_delta={score_delta}"
            )
        free_by_device = []
        peaks = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            free_by_device.append(free / 2**30)
            peaks.append({
                "device": index,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
                "free_after_diagnostic_gib": free / 2**30,
                "total_gib": total / 2**30,
            })
        if min(free_by_device) < self.config["minimum_free_gpu_gib_after_diagnostic"]:
            raise ContractError(f"Diagnostic left less than the required free GPU reserve: {peaks}")
        return {
            "study_id": uid,
            "label": LABELS[0],
            "native_logits_finite": True,
            "fp32_softmax_finite": True,
            "adapter_attached": bool(getattr(self.model, "peft_config", None)),
            "active_adapters": list(getattr(self.model, "active_adapters", [])),
            "max_logit_repeat_delta": max_logit_delta,
            "yes_score_repeat_delta": score_delta,
            "placement": self.placement,
            "hf_device_map": self.device_map,
            "input_dtypes_before_transfer": first["input_dtypes_before_transfer"],
            "model_input_dtypes": first["model_input_dtypes"],
            "native_output_logits_dtype": first["native_output_logits_dtype"],
            "input_device": first["input_device"],
            "dtype_report": self.dtype_report,
            "gpu_memory": peaks,
        }

    def predict(self, test: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        rows, timings = [], []
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        with torch.inference_mode():
            for study_index, uid in enumerate(test.StudyInstanceUID, 1):
                started = time.monotonic()
                record = {"StudyInstanceUID": uid}
                for label in LABELS:
                    result = self.score_logits(uid, label)
                    record[label] = result["yes_probability"]
                rows.append(record)
                timings.append({
                    "StudyInstanceUID": uid,
                    "study_seconds": time.monotonic() - started,
                    "cumulative_preprocessing_seconds": self.preprocessing_seconds,
                    "cumulative_model_forward_seconds": self.inference_seconds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
                    "peak_allocated_gpu1_gib": torch.cuda.max_memory_allocated(1) / 2**30 if torch.cuda.device_count() > 1 else None,
                    "peak_reserved_gpu1_gib": torch.cuda.max_memory_reserved(1) / 2**30 if torch.cuda.device_count() > 1 else None,
                })
                pd.DataFrame(rows).to_csv(output_dir / "submission.csv.partial", index=False)
                pd.DataFrame(timings).to_csv(output_dir / "study_timings.csv.partial", index=False)
                print(f"Scored {study_index}/{len(test)} studies", flush=True)
                self.images.pop(uid, None)
        frame = pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS])
        return frame, timings


def prepare_t4_teacher(config, series, data_dir, base_dir, adapter_dir, first_test_uid, attempt_log_path=None):
    """Try one T4 first; after a CUDA OOM only, make one explicit dual-T4 attempt."""
    import torch

    if torch.cuda.device_count() not in config["allowed_gpu_count"]:
        raise ContractError("Visible T4 count is outside the frozen 1-or-2-GPU scope")
    placements = ["single_gpu"]
    if torch.cuda.device_count() == 2:
        placements.append("two_gpu")
    attempts = []
    attempt_log_path = Path(attempt_log_path) if attempt_log_path else None

    def persist_attempts():
        if attempt_log_path is not None:
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = attempt_log_path.with_suffix(attempt_log_path.suffix + ".partial")
            temporary.write_text(json.dumps({"attempts": attempts}, indent=2, sort_keys=True))
            temporary.replace(attempt_log_path)

    for placement in placements:
        participating = 1 if placement == "single_gpu" else 2
        free_gib = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(participating)]
        from inference_core import validate_free_memory

        validate_free_memory(free_gib, config["minimum_free_gpu_gib_before_load"])
        teacher = None
        load_started = time.monotonic()
        try:
            teacher = ImageTeacher(config, series, data_dir, base_dir, adapter_dir, placement)
            load_seconds = time.monotonic() - load_started
            diagnostic = teacher.diagnostic(first_test_uid)
            attempts.append({
                "placement": placement,
                "status": "pass",
                "load_seconds": load_seconds,
                "hf_device_map": teacher.device_map,
                "diagnostic": diagnostic,
            })
            persist_attempts()
            return teacher, load_seconds, diagnostic, attempts
        except Exception as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                isinstance(error, RuntimeError)
                and "out of memory" in str(error).lower()
                and "cuda" in str(error).lower()
            )
            attempts.append({
                "placement": placement,
                "status": "oom" if is_oom else "failed",
                "load_or_diagnostic_seconds": time.monotonic() - load_started,
                "error_type": type(error).__name__,
                "error": str(error)[:500],
                "gpu_memory_at_failure": [
                    {
                        "device": index,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
                        "free_gib": torch.cuda.mem_get_info(index)[0] / 2**30,
                    }
                    for index in range(participating)
                ],
            })
            persist_attempts()
            if teacher is not None:
                del teacher
            gc.collect()
            torch.cuda.empty_cache()
            if not is_oom:
                raise
            if placement != "single_gpu" or len(placements) == 1:
                raise
    raise ContractError(f"No T4 placement passed: {attempts}")


def validate_device_map(device_map: dict, placement: str) -> dict:
    normalized = {str(key): str(value) for key, value in device_map.items()}
    if any(value in {"cpu", "disk"} for value in normalized.values()):
        raise ContractError("CPU/disk offload is forbidden for this T4 runtime")
    if placement == "single_gpu" and set(normalized.values()) != {"0"}:
        raise ContractError(f"Single-GPU attempt must stay on GPU 0: {normalized}")
    if placement == "two_gpu" and not {"0", "1"}.issubset(set(normalized.values())):
        raise ContractError(f"Explicit two-GPU map did not use both T4s: {normalized}")
    for key in normalized:
        if ".layers." in key:
            tail = key.split(".layers.", 1)[1]
        elif ".encoder.layers." in key:
            tail = key.split(".encoder.layers.", 1)[1]
        else:
            continue
        if tail.split(".", 1)[0].isdigit() and "." in tail:
            raise ContractError(f"A residual/vision block was split across devices: {key}")
    return normalized


def run_inference(
    config: dict,
    test,
    series,
    sample,
    data_dir,
    base_dir,
    adapter_dir,
    output_dir,
    teacher,
    load_seconds: float,
    asset_verification_seconds: float,
    dependency_install_seconds: float,
    diagnostic: dict,
):
    """Load the frozen pilot, score every actual test ID, and commit only a valid CSV."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {
        "status": "running",
        "planned_studies": len(test),
        "completed_studies": 0,
        "fallback_policy": config["fallback_policy"],
        "runtime_variant": config["runtime_variant"],
        "load_seconds": load_seconds,
        "asset_verification_seconds": asset_verification_seconds,
        "dependency_install_seconds": dependency_install_seconds,
        "pre_run_diagnostic": diagnostic,
    }
    atomic_json(output_dir / "status.json", status)
    try:
        prediction, timings = teacher.predict(test, output_dir)
        # Preserve test.csv order, never the visible example count.
        prediction = test[["StudyInstanceUID"]].merge(prediction, on="StudyInstanceUID", how="left", validate="one_to_one")
        schema = validate_submission(prediction, test)
        prediction.to_csv(output_dir / "submission.csv", index=False)
        pd.DataFrame(timings).to_csv(output_dir / "study_timings.csv", index=False)
        pd.DataFrame(teacher.imaging.issues).to_csv(output_dir / "dicom_issues.csv", index=False)
        total_seconds = time.monotonic() - started
        status.update({
            "status": "complete",
            "completed_studies": len(test),
            "total_seconds": total_seconds,
            "dicom_issue_rows": len(teacher.imaging.issues),
            "preprocessing_seconds": teacher.preprocessing_seconds,
            "inference_seconds": teacher.inference_seconds,
            "placement": teacher.placement,
            "hf_device_map": teacher.device_map,
            "dtype_report": teacher.dtype_report,
            "gpu_memory": [
                {
                    "device": i,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30,
                }
                for i in range(torch.cuda.device_count())
            ],
            "submission_schema": schema,
            "observed_seconds_per_study": float(np.mean([row["study_seconds"] for row in timings])),
        })
        atomic_json(output_dir / "status.json", status)
        return prediction, status
    except BaseException as error:
        partial = output_dir / "submission.csv.partial"
        completed = len(pd.read_csv(partial)) if partial.is_file() else 0
        status.update({
            "status": "failed",
            "completed_studies": completed,
            "error_type": type(error).__name__,
            "total_seconds": time.monotonic() - started,
        })
        atomic_json(output_dir / "status.json", status)
        raise
