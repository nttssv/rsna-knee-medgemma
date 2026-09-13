"""Create and verify a private migration archive; caches are re-downloadable."""

from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import tarfile
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create(state, archive, include_images=False):
    state = Path(state).resolve()
    archive = Path(archive).resolve()
    if archive.exists():
        raise FileExistsError(archive)
    paths = []
    for folder in ["runs", "data"]:
        for p in sorted((state / folder).rglob("*")):
            if p.is_symlink():
                raise ValueError("Symlinks cannot be bundled")
            if not p.is_file():
                continue
            rel = p.relative_to(state)
            if folder == "data" and p.suffix.lower() == ".dcm" and not include_images:
                continue
            # Runtime directories are defined by this project; reject stray credentials.
            secret_names = {
                "token",
                "token.json",
                "token.txt",
                "access_token",
                "refresh_token",
                "archive_url.json",
                "pilot_archive_url.json",
            }
            if (
                any(x.startswith(".") for x in rel.parts)
                or "credential" in p.name.lower()
                or p.name.lower() in secret_names
            ):
                continue
            paths.append(p)
    if not paths:
        raise ValueError("Nothing to back up")
    manifest = {
        "files": {str(p.relative_to(state)): sha(p) for p in paths},
        "dicom_included": include_images,
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent) as temp:
        temp = Path(temp)
        (temp / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        with tarfile.open(temp / "bundle.tar.gz", "w:gz") as tar:
            tar.add(temp / "BUNDLE_MANIFEST.json", arcname="BUNDLE_MANIFEST.json")
            for p in paths:
                tar.add(p, arcname=str(p.relative_to(state)), recursive=False)
        (temp / "bundle.tar.gz").replace(archive)
    archive.chmod(0o600)
    print(
        f"Private bundle: {archive}; {len(paths)} files; DICOM included: {include_images}"
    )


def restore(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Restore into a new directory to protect existing state")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temp = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                p = PurePosixPath(m.name)
                if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
                    raise ValueError("Unsafe archive member")
            if len({m.name for m in members}) != len(members):
                raise ValueError("Duplicate archive member")
            tar.extractall(temp, filter="data")
        manifest = json.loads((temp / "BUNDLE_MANIFEST.json").read_text())
        actual = {
            str(p.relative_to(temp))
            for p in temp.rglob("*")
            if p.is_file() and p.name != "BUNDLE_MANIFEST.json"
        }
        if actual != set(manifest["files"]):
            raise ValueError("Archive contents differ from manifest")
        for relative, digest in manifest["files"].items():
            if sha(temp / relative) != digest:
                raise ValueError(f"Checksum mismatch: {relative}")
        temp.rename(destination)
    destination.chmod(0o700)
    print(f"Verified and restored {len(actual)} files to {destination}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    c = s.add_parser("create")
    c.add_argument("--state-dir", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--include-images", action="store_true")
    r = s.add_parser("restore")
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument("--state-dir", type=Path, required=True)
    a = p.parse_args()
    if a.command == "create":
        create(a.state_dir, a.output, a.include_images)
    else:
        restore(a.archive, a.state_dir)


if __name__ == "__main__":
    main()
