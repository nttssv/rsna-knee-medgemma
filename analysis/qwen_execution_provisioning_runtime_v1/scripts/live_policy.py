"""Source-bound disabled production gates and durable operation replay protection."""
from pathlib import Path
import hashlib
import json
import os
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
SWITCHES = ("provisioning_enabled", "resume_enabled", "execution_enabled",
            "external_shutdown_enabled", "live_transport_enabled")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source_bundle(expected_sha256=None):
    path = ROOT / "configs/source_manifest.json"
    actual = file_sha(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise PermissionError("Live source bundle differs from approval")
    manifest = json.loads(path.read_text())
    files = manifest.get("files", {})
    expected_files = {str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.py")}
    expected_files.add("configs/runtime.json")
    if set(files) != expected_files:
        raise PermissionError("Live source manifest has missing or extra implementation files")
    for name, expected in files.items():
        target = ROOT / name
        if target.is_symlink() or file_sha(target) != expected:
            raise PermissionError("Live source or policy drift detected")
    return actual


def require_live(operation):
    if operation not in SWITCHES:
        raise PermissionError("Unknown live operation")
    cfg = json.loads((ROOT / "configs/runtime.json").read_text())
    if cfg.get(operation) is not True or cfg.get("live_transport_enabled") is not True:
        raise PermissionError("Live operation remains disabled in the committed policy")
    verify_source_bundle()
    return cfg


def private_directory(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise PermissionError("Absolute non-symlink operation ledger required")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.getuid()):
        raise PermissionError("Operation ledger must be an owner-only 0700 directory")
    return path


class OnceLedger:
    """Never delete claims, including after ambiguous calls or allocator death.

    The absolute ledger directory is bound in the private outer approval. Claims
    are keyed by intent, not output directory or object identity, so changing the
    output path cannot replay an approved operation. There is no recovery/retry API.
    """
    def __init__(self, root, intent_name, bindings):
        if not re.fullmatch(r"qwen-provision-[0-9a-f]{32}", intent_name):
            raise PermissionError("Exact high-entropy intent required")
        self.root = private_directory(root)
        self.intent_name, self.bindings = intent_name, dict(bindings)
        path = self.root / (self.intent_name + "-binding.json")
        data = (json.dumps(self.bindings, sort_keys=True, allow_nan=False) + "\n").encode()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid() or path.read_bytes() != data):
                raise PermissionError("Intent ledger is already bound to different authorization/source")
        else:
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
            parent = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)

    def claim(self, operation):
        if operation not in {"create", "resume", "handoff"}:
            raise PermissionError("Unsupported once-only operation")
        path = self.root / (self.intent_name + "-" + operation + ".json")
        data = (json.dumps({"operation": operation, "intent_name": self.intent_name,
                           "bindings": self.bindings}, sort_keys=True, allow_nan=False) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
        parent = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return path
