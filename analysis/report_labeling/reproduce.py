"""Replay the frozen report-labeling experiment from private state, without training."""

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile


CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parents[1]
SNAPSHOT_NAME = "report-labeling-20260913-v1"
PRIVATE_INPUTS = {"splits.csv", "split_manifest.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe path in experiment manifest")
    return path


def verify_code(code_root=CODE_ROOT):
    manifest = json.loads((code_root / "frozen_code_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        path = code_root / relative_name(name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Frozen code differs: {name}. Keep v1 unchanged; use a separate version for further work.")
    return manifest


def prepare_experiment(code_root, snapshot, target):
    """Join tracked frozen code and private split metadata in a temporary directory."""
    public = verify_code(code_root)
    frozen = json.loads((snapshot / "freeze_manifest.json").read_text())
    expected = set(public["files_sha256"]) | PRIVATE_INPUTS
    if set(frozen["files_sha256"]) != expected:
        raise ValueError("Private snapshot has an unexpected set of frozen files")
    sources = {}
    for name, digest in frozen["files_sha256"].items():
        relative_name(name)
        source = (snapshot if name in PRIVATE_INPUTS else code_root) / name
        if name not in PRIVATE_INPUTS and digest != public["files_sha256"][name]:
            raise ValueError("Snapshot refers to a different code version")
        if not source.is_file() or sha(source) != digest:
            raise ValueError(f"Snapshot input differs: {name}")
        sources[name] = source
    target.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(snapshot / "freeze_manifest.json", target / "freeze_manifest.json")
    return frozen


def compare_results(snapshot, output):
    expected = json.loads((snapshot / "evaluation_manifest.json").read_text())["output_sha256"]
    # These two files belong to split preparation, not the evaluation replay.
    expected = {name: digest for name, digest in expected.items()
                if name.endswith(".csv") and name not in {"development_gold.csv", "splits.csv"}}
    actual = {path.name for path in output.glob("*.csv")}
    if actual != set(expected):
        raise ValueError("Reproduction CSV filenames differ from the recorded evaluation outputs")
    matches = {}
    for path in sorted(output.glob("*.csv")):
        matches[path.name] = path.name in expected and sha(path) == expected[path.name]
    if not matches or not all(matches.values()):
        raise ValueError("Reproduction CSVs differ from the recorded result; inspect dependency versions before making claims.")
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path(os.environ.get("RSNA_STATE_DIR", REPO_ROOT / "state")))
    parser.add_argument("--source", type=Path, help="Defaults to STATE/data/train.csv")
    parser.add_argument("--snapshot", type=Path, help="Defaults to the preserved v1 run under STATE/runs/")
    parser.add_argument("--output", type=Path, help="New output directory inside STATE; a timestamped run is the default")
    args = parser.parse_args()
    state = args.state_dir.resolve()
    source = (args.source or state / "data/train.csv").resolve()
    snapshot = (args.snapshot or state / "runs" / SNAPSHOT_NAME).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or state / "runs" / f"report-labeling-reproduction-{stamp}").resolve()
    if not output.is_relative_to(state):
        raise ValueError("Keep per-study outputs inside RSNA_STATE_DIR")
    if output.exists():
        raise FileExistsError("Choose a new output directory; prior results are never replaced")
    if not source.is_file() or not (snapshot / "freeze_manifest.json").is_file():
        raise FileNotFoundError("Restore the private data and report-labeling snapshot first; see analysis/report_labeling/README.md")
    state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="report-labeling-", dir=state) as temp:
        experiment = Path(temp) / "experiment"
        frozen = prepare_experiment(CODE_ROOT, snapshot, experiment)
        if sha(source) != frozen["source_sha256"]:
            raise ValueError("Source CSV differs from the frozen experiment")
        subprocess.run([sys.executable, str(experiment / "src/pipeline.py"), "evaluate",
                        "--source", str(source), "--experiment", str(experiment), "--out", str(output)], check=True)
    matches = compare_results(snapshot, output)
    record = {"method": "frozen_v1_reproduction_not_new_validation", "csv_matches_snapshot": matches,
              "frozen_code_manifest_sha256": sha(CODE_ROOT / "frozen_code_manifest.json"),
              "studies_processed": 58, "unlabeled_studies_processed": 0, "training_started": False}
    (output / "reproduction_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified {len(matches)} CSVs identical to the original frozen result. Private output: {output}")


if __name__ == "__main__":
    main()
