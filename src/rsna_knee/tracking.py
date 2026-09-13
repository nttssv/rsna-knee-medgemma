"""Import recorded scalar metrics or serve a local-only Trackio dashboard."""

from pathlib import Path
import argparse
import contextlib
import json
import os
import re
import time
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["import", "serve"])
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--tracking-dir", type=Path, default=Path("state/tracking"))
    p.add_argument("--project", default="rsna-knee-medgemma")
    p.add_argument("--name")
    p.add_argument("--port", type=int, default=7861)
    a = p.parse_args()
    a.tracking_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(a.tracking_dir.resolve())
    os.environ.pop("TRACKIO_SPACE_ID", None)
    os.environ.pop("TRACKIO_SERVER_URL", None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import trackio

    if a.command == "serve":
        with open(os.devnull, "w") as hidden, contextlib.redirect_stdout(hidden):
            app, _, _, _ = trackio.show(
                project=a.project,
                host="127.0.0.1",
                server_port=a.port,
                share=False,
                open_browser=False,
                block_thread=False,
            )
        print(f"Trackio: http://127.0.0.1:{a.port}/?project={a.project}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            app.close()
        return
    if not a.run_dir:
        p.error("--run-dir is required for import")
    name = a.name or a.run_dir.name
    marker = a.tracking_dir / (
        "import_" + re.sub(r"[^a-zA-Z0-9_-]", "_", name) + ".json"
    )
    if marker.exists():
        print("Already imported; use a new run name for a changed record")
        return
    history = pd.read_csv(a.run_dir / "training_history.csv")
    metrics = pd.read_csv(a.run_dir / "validation_metrics.csv")
    events = {
        int(row.step): {"train/loss": float(row.loss)} for row in history.itertuples()
    }
    for tag, frame in metrics.groupby("model"):
        step = 0 if tag in ["base", "initial"] else int(history.step.max())
        events.setdefault(step, {}).update(
            {"validation/mean_auc": float(frame.auc.mean())}
        )
        for row in frame.itertuples():
            if pd.notna(row.auc):
                events[step][
                    "validation/auc_"
                    + re.sub(r"[^a-z0-9]+", "_", row.label.lower()).strip("_")
                ] = float(row.auc)
    trackio.init(
        project=a.project,
        name=name,
        config={"source": "recorded local metrics"},
        embed=False,
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    try:
        for step, values in sorted(events.items()):
            trackio.log(values, step=step)
    finally:
        trackio.finish()
    marker.write_text(json.dumps({"project": a.project, "name": name}))
    print(f"Imported {len(events)} steps")


if __name__ == "__main__":
    main()
