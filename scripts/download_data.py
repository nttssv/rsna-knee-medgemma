"""Download only labeled pilot studies from the authorized Kaggle ZIP."""

import argparse
import csv
import getpass
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from rsna_knee.rangefile import RangeFile
from rsna_knee.constants import LABELS, PLANES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--metadata-only", action="store_true")
    args = p.parse_args()
    os.umask(0o077)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("KAGGLE_API_TOKEN") or getpass.getpass(
        "Kaggle API token (hidden, used in memory only): "
    )
    if not token:
        raise ValueError(
            "Kaggle API token required; competition access must already be granted"
        )
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.competitions.types.competition_api_service import (
        ApiDownloadDataFilesRequest,
    )

    with KaggleClient(api_token=token) as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = "rsna-knee-abnormality-detection"
        url = client.competitions.competition_api_client.download_data_files(
            request
        ).url
    del token
    remote = RangeFile(url)
    with zipfile.ZipFile(remote) as archive:
        for name in [
            "train.csv",
            "train_series.csv",
            "test.csv",
            "test_series.csv",
            "sample_submission.csv",
        ]:
            info = archive.getinfo(name)
            if info.file_size > 100_000_000:
                raise ValueError("Unexpectedly large CSV")
            (args.data_dir / name).write_bytes(archive.read(name))
        if args.metadata_only:
            return
        with (args.data_dir / "train.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        studies = {
            r["StudyInstanceUID"]
            for r in rows
            if any(r[c] not in ["", None] for c in LABELS)
        }
        with (args.data_dir / "train_series.csv").open(newline="") as f:
            series = [r for r in csv.DictReader(f) if r["StudyInstanceUID"] in studies]
        chosen = set()
        for study in studies:
            for plane in PLANES:
                choices = [
                    r
                    for r in series
                    if r["StudyInstanceUID"] == study
                    and r["Anatomical_Plane"].lower() == plane.lower()
                ]
                if not choices:
                    raise ValueError("Labeled study missing a required plane")
                row = sorted(
                    choices,
                    key=lambda r: (
                        -(
                            float(r["Fluid_Sensitive"] or 0) * 2
                            + float(r["Fat_Suppression"] or 0)
                        ),
                        r["SeriesInstanceUID"],
                    ),
                )[0]
                chosen.add((study, row["SeriesInstanceUID"]))
        files = []
        for item in archive.infolist():
            bits = item.filename.split("/")
            if (
                len(bits) == 4
                and bits[0] == "train_series"
                and (bits[1], bits[2]) in chosen
                and bits[3].endswith(".dcm")
            ):
                files.append(
                    dict(
                        name=item.filename,
                        header_offset=item.header_offset,
                        compress_size=item.compress_size,
                        file_size=item.file_size,
                        compress_type=item.compress_type,
                        crc=item.CRC,
                    )
                )
    if {(f["name"].split("/")[1], f["name"].split("/")[2]) for f in files} != chosen:
        raise ValueError("Incomplete ZIP series")
    manifest = dict(
        studies=len(studies),
        series=len(chosen),
        files=files,
        compressed_bytes=sum(f["compress_size"] for f in files),
        extracted_bytes=sum(f["file_size"] for f in files),
    )
    if manifest["extracted_bytes"] > 25_000_000_000:
        raise ValueError("Pilot exceeds 25 GB cap; review dataset changes")
    (args.data_dir / "pilot_manifest.json").write_text(json.dumps(manifest))
    print(
        f"Selected {len(studies)} studies, {len(files)} DICOMs, {manifest['extracted_bytes'] / 1e9:.2f} GB extracted",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("download_ranges.py")),
            "--data-dir",
            str(args.data_dir),
            "--workers",
            str(args.workers),
        ],
        input=json.dumps({"url": url}) + "\n",
        text=True,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # HTTP exception messages can contain signed credentials.
        print(
            f"Download failed ({type(error).__name__}); verify access and retry. Existing CRC-verified files are reused.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
