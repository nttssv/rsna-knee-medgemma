"""Image-only MedGemma pilot inference for the RSNA Knee code competition.

This module intentionally accepts imaging inputs only. Any study that
cannot be scored aborts the run; it never substitutes neutral scores.
"""

from __future__ import annotations

import json
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
    def __init__(self, config: dict, series: pd.DataFrame, data_dir: Path, base_dir: Path, adapter_dir: Path):
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.config = config
        self.series = series
        self.imaging = Imaging(data_dir, config["slices_per_plane"])
        self.images: dict[str, list] = {}
        self.processor = AutoProcessor.from_pretrained(
            adapter_dir, local_files_only=True, use_fast=False
        )
        self.processor.tokenizer.padding_side = "right"
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            base_dir,
            local_files_only=True,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=config["attention_implementation"],
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model.config.use_cache = False
        self.model = model
        self.answer_ids = []
        for answer in ["No", "Yes"]:
            ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ContractError("Answer tokenization changed")
            self.answer_ids.append(ids[0])

    def messages(self, uid: str, label: str):
        if uid not in self.images:
            arrays, mask, metadata = self.imaging.study_images(uid, self.series, self.config["image_size"])
            self.images[uid] = [
                (Image.fromarray(arrays[i]).convert("RGB"), metadata[i])
                for i in np.flatnonzero(mask)
            ]
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
    def _on_gpu(batch, torch):
        return {
            key: value.to("cuda:0", dtype=torch.bfloat16) if value.is_floating_point() else value.to("cuda:0")
            for key, value in batch.items()
        }

    def predict(self, test: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        rows, timings = [], []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for study_index, uid in enumerate(test.StudyInstanceUID, 1):
                started = time.monotonic()
                record = {"StudyInstanceUID": uid}
                for label in LABELS:
                    encoded = self.processor.apply_chat_template(
                        self.messages(uid, label),
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    batch = self._on_gpu(encoded, torch)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    pair = logits[self.answer_ids]
                    if not torch.isfinite(pair).all():
                        raise ContractError("Non-finite model logits")
                    record[label] = float(pair.softmax(0)[1].cpu())
                    del encoded, batch, logits, pair
                rows.append(record)
                timings.append({
                    "StudyInstanceUID": uid,
                    "study_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                })
                pd.DataFrame(rows).to_csv(output_dir / "submission.csv.partial", index=False)
                pd.DataFrame(timings).to_csv(output_dir / "study_timings.csv.partial", index=False)
                print(f"Scored {study_index}/{len(test)} studies", flush=True)
        frame = pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS])
        return frame, timings


def run_inference(config: dict, test, series, sample, data_dir, base_dir, adapter_dir, output_dir):
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
    }
    atomic_json(output_dir / "status.json", status)
    try:
        load_started = time.monotonic()
        teacher = ImageTeacher(config, series, data_dir, base_dir, adapter_dir)
        load_seconds = time.monotonic() - load_started
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
            "load_seconds": load_seconds,
            "total_seconds": total_seconds,
            "dicom_issue_rows": len(teacher.imaging.issues),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
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
