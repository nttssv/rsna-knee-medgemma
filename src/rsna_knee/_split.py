"""Original pilot grouping; reports are used for deduplication only."""

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .imaging import header


def create_split(pilot, DATA, SEED, HOLDOUT_FRACTION):
    def digest(s):
        return hashlib.sha256(str(s).encode()).hexdigest()[:20]

    parent = list(range(len(pilot)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(b)] = root(a)

    seen = {}
    scanner = []
    for i, r in pilot.iterrows():
        keys = []
        report = str(r.get("Report", "")) if pd.notna(r.get("Report", None)) else ""
        normalized = " ".join(report.lower().split())
        if normalized:
            keys.append("report:" + digest(normalized))
        paths = sorted((DATA / "train_series" / r.StudyInstanceUID).glob("*/*.dcm"))
        h = header(paths[0]) if paths else None
        fields = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"]
        sig = "|".join(str(getattr(h, k, "")) for k in fields)
        scanner.append(digest(sig) if sig.strip("|") else "unknown")
        pid = str(getattr(h, "PatientID", "")).strip()
        if pid and pid.lower() not in {
            "anonymous",
            "anonymized",
            "unknown",
            "none",
            "0",
            "1",
        }:
            keys.append("patient:" + digest(sig + "|" + pid))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = np.array([digest(root(i)) for i in range(len(pilot))])
    assert len(set(groups)) >= 4, (
        "Too few independent groups; revise split before training."
    )
    tr, va = next(
        GroupShuffleSplit(
            n_splits=1, test_size=HOLDOUT_FRACTION, random_state=SEED
        ).split(pilot, groups=groups)
    )
    pilot["split"] = "train"
    pilot.loc[va, "split"] = "validation"
    pilot["group"] = groups
    pilot["scanner_group"] = scanner
    assert set(pilot.loc[tr, "group"]).isdisjoint(pilot.loc[va, "group"])

    return pilot
