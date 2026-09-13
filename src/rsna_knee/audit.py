from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .constants import LABELS, PLANES
from . import imaging
from ._split import create_split


def validate_split(split, available):
    required = {"StudyInstanceUID", "split", "group", "scanner_group"}
    if not required.issubset(split.columns):
        raise ValueError("Split columns missing")
    if not split.StudyInstanceUID.is_unique:
        raise ValueError("Duplicate studies in split")
    if not set(split.StudyInstanceUID).issubset(set(available.StudyInstanceUID)):
        raise ValueError("Split includes absent or completely unlabeled studies")
    if set(split["split"]) != {"train", "validation"}:
        raise ValueError("Both split partitions required")
    if split["group"].isna().any():
        raise ValueError("Missing split groups")
    tr = split.loc[split["split"] == "train", "group"]
    va = split.loc[split["split"] == "validation", "group"]
    if set(tr) & set(va):
        raise ValueError("Group leakage across training and validation")


def run(cfg, data, out, split_file=None):
    data = Path(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    imaging.DATA = data
    imaging.SLICES_PER_PLANE = cfg.slices_per_plane
    imaging.ISSUES.clear()
    train = pd.read_csv(data / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(
        data / "train_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    if not train.StudyInstanceUID.is_unique:
        raise ValueError("Training study IDs are not unique")
    if not set(series.StudyInstanceUID).issubset(set(train.StudyInstanceUID)):
        raise ValueError("Series reference unknown studies")
    for label in LABELS:
        train[label] = pd.to_numeric(train[label], errors="raise")
        if not train[label].dropna().isin([0, 1]).all():
            raise ValueError(f"Invalid label: {label}")
    if (data / "test.csv").exists():
        test = pd.read_csv(data / "test.csv", dtype={"StudyInstanceUID": str})
        if set(train.StudyInstanceUID) & set(test.StudyInstanceUID):
            raise ValueError("Train/test overlap")
    if (data / "sample_submission.csv").exists():
        if (
            list(pd.read_csv(data / "sample_submission.csv", nrows=0).columns)
            != ["StudyInstanceUID"] + LABELS
        ):
            raise ValueError("Unexpected competition submission schema")
    pd.DataFrame(
        {
            "known": train[LABELS].notna().sum(),
            "positive": train[LABELS].eq(1).sum(),
            "negative": train[LABELS].eq(0).sum(),
            "missing": train[LABELS].isna().sum(),
        }
    ).to_csv(out / "label_audit.csv", index_label="label")
    available = train.loc[train[LABELS].notna().any(axis=1)].copy()
    if split_file:
        split = pd.read_csv(split_file, dtype=str)
        validate_split(split, available)
        pilot = split.merge(
            available, on="StudyInstanceUID", validate="one_to_one", sort=False
        )
    else:
        pilot = available.sample(
            n=min(len(available), cfg.max_studies), random_state=cfg.seed
        ).reset_index(drop=True)
        if len(pilot) < 12:
            raise ValueError("Too few labeled studies for the pilot")
        pilot = create_split(pilot, data, cfg.seed, cfg.holdout_fraction)
    validate_split(pilot, available)
    pilot[["StudyInstanceUID", "split", "group", "scanner_group"]].to_csv(
        out / "development_split.csv", index=False
    )
    # Keep organizer labels in private run state; raw reports are never exported.
    pilot[["StudyInstanceUID"] + LABELS].to_csv(out / "ground_truth.csv", index=False)
    coverage = []
    for n, uid in enumerate(pilot.StudyInstanceUID, 1):
        images, mask, meta = imaging.study_images(uid, series, size=cfg.image_size)
        coverage.append(
            dict(
                StudyInstanceUID=uid,
                valid_slices=int(mask.sum()),
                sha256=hashlib.sha256(
                    images.tobytes() + mask.tobytes() + meta.tobytes()
                ).hexdigest(),
            )
        )
        if n % 10 == 0 or n == len(pilot):
            print(f"Preflight: {n}/{len(pilot)}", flush=True)
    pd.DataFrame(coverage).to_csv(out / "input_fingerprints.csv", index=False)
    pd.DataFrame(imaging.ISSUES, columns=["study", "issue"]).to_csv(
        out / "preprocessing_issues.csv", index=False
    )
    if any(r["valid_slices"] != len(PLANES) * cfg.slices_per_plane for r in coverage):
        raise ValueError(
            "Every pilot study must have all expected slices; inspect coverage"
        )
    tr = pilot["split"].eq("train")
    va = pilot["split"].eq("validation")
    status = dict(
        status="data_audit_complete",
        model=cfg.model,
        studies=len(train),
        organizer_labeled_studies=len(available),
        pilot_train=int(tr.sum()),
        pilot_validation=int(va.sum()),
        slices_per_plane=cfg.slices_per_plane,
        scanner_groups_shared=len(
            set(pilot.loc[tr, "scanner_group"]) & set(pilot.loc[va, "scanner_group"])
        ),
    )
    (out / "audit_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return pilot, series
