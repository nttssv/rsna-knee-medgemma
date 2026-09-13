from pathlib import Path
from importlib.resources import files
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from .constants import LABELS, PLANES, DEFINITIONS
from . import imaging


def export(cfg, data, run):
    data = Path(data)
    run = Path(run)
    out = run / "case_review"
    out.mkdir(exist_ok=True)
    imaging.DATA = data
    imaging.SLICES_PER_PLANE = cfg.slices_per_plane
    if cfg.slices_per_plane != 2 or cfg.image_size != 448:
        raise ValueError(
            "This viewer currently supports the recorded six-image 448px recipe only"
        )
    series = pd.read_csv(
        data / "train_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    truth = pd.read_csv(data / "train.csv").set_index("StudyInstanceUID")
    split = pd.read_csv(run / "development_split.csv")
    uids = split.loc[split["split"] == "validation", "StudyInstanceUID"]
    after = pd.read_csv(run / "validation_after.csv").set_index("StudyInstanceUID")
    before = (
        pd.read_csv(run / "validation_before.csv").set_index("StudyInstanceUID")
        if (run / "validation_before.csv").exists()
        else None
    )
    if set(uids) != set(after.index):
        raise ValueError("Incomplete or mismatched validation predictions")
    if before is not None and set(uids) != set(before.index):
        raise ValueError("Mismatched baseline predictions")
    cases = []
    comparison = []
    for n, uid in enumerate(uids, 1):
        arr, mask, meta = imaging.study_images(uid, series, size=cfg.image_size)
        if not mask.all():
            raise ValueError("Cannot display an incomplete input as six real slices")
        c = dict(case=n, uid=uid, slices=[], truth={}, before={}, after={})
        for label in LABELS:
            y = truth.loc[uid, label]
            y = None if pd.isna(y) else int(y)
            c["truth"][label] = y
            c["before"][label] = (
                float(before.loc[uid, label]) if before is not None else None
            )
            c["after"][label] = float(after.loc[uid, label])
            p = int(c["after"][label] >= 0.5)
            comparison.append(
                dict(
                    case=n,
                    StudyInstanceUID=uid,
                    condition=label,
                    ground_truth=y,
                    before=c["before"][label],
                    after=c["after"][label],
                    threshold=0.5,
                    prediction=p,
                    comparison="unknown"
                    if y is None
                    else "match"
                    if p == y
                    else "false_positive"
                    if p
                    else "missed_positive",
                )
            )
        for i, (a, m) in enumerate(zip(arr, meta)):
            plane = PLANES[int(np.argmax(m[:3]))]
            path = f"case-{n:02d}/slice-{i + 1}.png"
            (out / path).parent.mkdir(exist_ok=True)
            Image.fromarray(a).convert("RGB").save(out / path)
            c["slices"].append(
                dict(
                    path=path,
                    plane=plane,
                    position=float(m[3]),
                    fluid_sensitive=int(m[4]),
                    fat_suppression=int(m[5]),
                    prompt_text=f"{plane}, slice position {m[3]:.2f}, fluid sensitive {int(m[4])}, fat suppression {int(m[5])}.",
                    sha256=hashlib.sha256((out / path).read_bytes()).hexdigest(),
                )
            )
        cases.append(c)
    payload = dict(
        model=cfg.model,
        run_id=run.name,
        labels=LABELS,
        definitions=DEFINITIONS,
        cases=cases,
    )
    (out / "cases.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    (out / "data.js").write_text(
        "const MEDGEMMA_DATA = "
        + json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
        + ";\n"
    )
    pd.DataFrame(comparison).to_csv(out / "prediction_comparison.csv", index=False)
    install_template(out, len(cases), cfg.max_steps)
    print(f"Exported {len(cases)} held-out cases to {out}")


def install_template(out, count=15, steps=20):
    template = files("rsna_knee").joinpath("assets/case_review.html").read_text()
    template = template.replace("all 15 held-out", f"all {count} held-out").replace(
        "20-step pilot", f"{steps}-step pilot"
    )
    template = template.replace("Run: 20260913T135620Z. ", "")
    (Path(out) / "index.html").write_text(template)


def serve(run, port=7862):
    folder = Path(run) / "case_review"
    if not (folder / "index.html").exists():
        raise ValueError(
            "Export cases first, or restore the existing case_review folder"
        )
    print(f"Case review: http://127.0.0.1:{port}/ (local only)", flush=True)
    ThreadingHTTPServer(
        ("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(folder))
    ).serve_forever()
