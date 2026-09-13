"""Download the selected private DICOM pilot using ZIP ranges and CRC checks.

The signed URL is read from stdin JSON and retained in memory only.
No Kaggle refresh token is needed on the server. The full archive is never saved.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import threading
import time
import zlib

import requests


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "pilot_manifest.json").read_text())
    assert manifest["extracted_bytes"] <= 25_000_000_000
    assert 1 <= args.workers <= 16
    url = json.loads(sys.stdin.readline())["url"]
    local = threading.local()

    def download(item):
        relative = PurePosixPath(item["name"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] != "train_series"
        ):
            raise ValueError("Unsafe manifest path")
        target = args.data_dir.joinpath(*relative.parts)
        if target.is_file() and target.stat().st_size == item["file_size"]:
            if zlib.crc32(target.read_bytes()) == item["crc"]:
                return "cached"
        if not hasattr(local, "session"):
            local.session = requests.Session()
        start = item["header_offset"]
        # ZIP extras are usually short. Reject unexpectedly large headers safely.
        length = 30 + len(item["name"].encode("utf-8")) + 4096 + item["compress_size"]
        for attempt in range(4):
            try:
                with local.session.get(
                    url,
                    headers={"Range": f"bytes={start}-{start + length - 1}"},
                    stream=True,
                    timeout=90,
                ) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    block = r.content
                header = struct.unpack("<4s5H3I2H", block[:30])
                if header[0] != b"PK\x03\x04":
                    raise ValueError("Wrong ZIP entry signature")
                offset = 30 + header[-2] + header[-1]
                stored_name = block[30 : 30 + header[-2]].decode("utf-8")
                if stored_name != item["name"] or header[3] != item["compress_type"]:
                    raise ValueError("ZIP entry does not match the manifest")
                payload = block[offset : offset + item["compress_size"]]
                if len(payload) != item["compress_size"]:
                    raise ValueError("Incomplete entry payload")
                if item["compress_type"] == 8:
                    decoded = zlib.decompress(payload, -15)
                elif item["compress_type"] == 0:
                    decoded = payload
                else:
                    raise ValueError("Unsupported ZIP compression")
                if (
                    len(decoded) != item["file_size"]
                    or zlib.crc32(decoded) != item["crc"]
                ):
                    raise ValueError("DICOM size or CRC mismatch")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(".dcm.partial")
                partial.write_bytes(decoded)
                partial.replace(target)
                return "downloaded"
            except Exception:
                if attempt == 3:
                    # Request exceptions can contain the secret signed URL.
                    raise RuntimeError(
                        "A pilot file failed after four attempts"
                    ) from None
                time.sleep(2**attempt)

    started = time.monotonic()
    files = manifest["files"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in files]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 250 == 0 or i == len(files):
                print(
                    f"Pilot DICOMs verified: {i}/{len(files)}; elapsed {time.monotonic() - started:.0f}s",
                    flush=True,
                )
    status = dict(
        status="complete",
        files=len(files),
        studies=manifest["studies"],
        series=manifest["series"],
        bytes=manifest["extracted_bytes"],
        crc_verified=True,
    )
    (args.data_dir / "download_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
