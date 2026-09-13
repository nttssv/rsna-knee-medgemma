"""MRI preprocessing shared with the verified MedGemma pilot. Research use."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from pydicom.pixels import apply_modality_lut

PLANES = ["Sagittal", "Coronal", "Axial"]
SLICES_PER_PLANE = 2
DATA = Path(".")

ISSUES = []
HEADER_TAGS = [
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "InstanceNumber",
    "PixelSpacing",
    "PatientID",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
]


def header(path):
    return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)


def rank_series(rows):
    rows = rows.copy()
    rows["_rank"] = pd.to_numeric(rows["Fluid_Sensitive"], errors="coerce").fillna(
        0
    ) * 2 + pd.to_numeric(rows["Fat_Suppression"], errors="coerce").fillna(0)
    return rows.sort_values(["_rank", "SeriesInstanceUID"], ascending=[False, True])


def ordered_paths(folder, uid):
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        raise ValueError("No DICOM slices")
    hs = [header(p) for p in paths]
    if all(
        (
            hasattr(h, "ImagePositionPatient") and hasattr(h, "ImageOrientationPatient")
            for h in hs
        )
    ):
        orient = np.asarray(hs[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        if np.linalg.norm(normal) < 0.9:
            raise ValueError("Invalid slice orientation")
        if not all(
            (
                np.allclose(
                    np.asarray(h.ImageOrientationPatient, float), orient, atol=0.05
                )
                for h in hs
            )
        ):
            raise ValueError("Mixed orientation within a series")
        positions = [
            float(np.dot(np.asarray(h.ImagePositionPatient, float), normal)) for h in hs
        ]
    elif all((hasattr(h, "InstanceNumber") for h in hs)):
        positions = [float(h.InstanceNumber) for h in hs]
        ISSUES.append(dict(study=uid, issue="InstanceNumber ordering fallback"))
    else:
        raise ValueError("Neither geometry nor complete InstanceNumber is available")
    order = np.argsort(positions, kind="stable")
    return [paths[i] for i in order]


def canonical_pixels(ds):
    arr = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected single grayscale slice, received shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    spacing = np.asarray(getattr(ds, "PixelSpacing", [1, 1]), dtype=float)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        spacing = np.ones(2)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None:
        col, row = (np.array(iop[:3], float), np.array(iop[3:], float))
        if np.argmax(abs(col)) > np.argmax(abs(row)):
            arr = arr.T
            col, row = (row, col)
            spacing = spacing[::-1]
        if col[np.argmax(abs(col))] < 0:
            arr = np.fliplr(arr)
        if row[np.argmax(abs(row))] < 0:
            arr = np.flipud(arr)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = -arr
    return (arr.copy(), spacing)


def render_series(paths, size):
    idx = np.unique(
        np.linspace(
            0.25 * (len(paths) - 1),
            0.75 * (len(paths) - 1),
            min(SLICES_PER_PLANE, len(paths)),
        )
        .round()
        .astype(int)
    )
    decoded = [canonical_pixels(pydicom.dcmread(paths[i])) for i in idx]
    values = np.concatenate([a.ravel()[:: max(1, a.size // 10000)] for a, _ in decoded])
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        raise ValueError("Constant-intensity series")
    rendered = []
    for arr, spacing in decoded:
        arr = np.uint8(np.clip((arr - lo) / (hi - lo), 0, 1) * 255)
        ph, pw = np.array(arr.shape) * spacing
        nh, nw = (
            max(1, round(size * ph / max(ph, pw))),
            max(1, round(size * pw / max(ph, pw))),
        )
        im = Image.fromarray(arr).resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (size, size))
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        rendered.append(np.asarray(canvas))
    return (rendered, idx / max(1, len(paths) - 1))


def study_images(uid, descriptors, split="train", size=224):
    n = len(PLANES) * SLICES_PER_PLANE
    images = np.zeros((n, size, size), np.uint8)
    mask = np.zeros(n, bool)
    meta = np.zeros((n, 6), np.float32)
    rows = descriptors[descriptors.StudyInstanceUID == uid]
    for pi, plane in enumerate(PLANES):
        choices = rank_series(
            rows[rows.Anatomical_Plane.astype(str).str.lower() == plane.lower()]
        )
        for _, r in choices.iterrows():
            try:
                paths = ordered_paths(
                    DATA / f"{split}_series" / uid / r.SeriesInstanceUID, uid
                )
                ims, pos = render_series(paths, size)
                sl = slice(pi * SLICES_PER_PLANE, pi * SLICES_PER_PLANE + len(ims))
                images[sl] = np.stack(ims)
                mask[sl] = True
                meta[sl, pi] = 1
                meta[sl, 3] = pos
                meta[sl, 4] = (
                    float(r.Fluid_Sensitive) if pd.notna(r.Fluid_Sensitive) else 0
                )
                meta[sl, 5] = (
                    float(r.Fat_Suppression) if pd.notna(r.Fat_Suppression) else 0
                )
                break
            except Exception as e:
                ISSUES.append(
                    dict(
                        study=uid, issue=f"{plane}: {type(e).__name__}: {str(e)[:160]}"
                    )
                )
    if not mask.any():
        raise ValueError("No usable MRI series for this study")
    return (images, mask, meta)
