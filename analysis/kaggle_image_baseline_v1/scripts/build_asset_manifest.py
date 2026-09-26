"""Hash a private offline asset package without publishing the model files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_core import REQUIRED_ADAPTER_FILES, REQUIRED_BASE_FILES, sha256_file


def build(root: Path, config_path: Path) -> dict:
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    files = {}
    for folder, names in [("base_model", REQUIRED_BASE_FILES), ("adapter", REQUIRED_ADAPTER_FILES)]:
        for name in names:
            path = root / folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files[f"{folder}/{name}"] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    wheels = sorted((root / "wheels").glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("Offline wheelhouse is empty")
    for path in wheels:
        files[f"wheels/{path.name}"] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "asset_contract": "rsna-knee-medgemma-pilot-v1",
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "adapter_sha256": config["adapter_sha256"],
        "files": files,
    }
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.asset_root, args.config)
    print(json.dumps({"files": len(value["files"]), "manifest": str(args.asset_root / 'asset_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
