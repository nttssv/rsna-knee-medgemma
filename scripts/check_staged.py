"""Reject accidentally staged private state, secrets, notebook output, or large files."""

from pathlib import Path
import json
import re
import subprocess
import sys

names = (
    subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    )
    .decode()
    .split("\0")
)
patterns = [
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    rb"hf_[A-Za-z0-9]{24,}",
    rb"gh[pousr]_[A-Za-z0-9]{30,}",
    rb"github_pat_[A-Za-z0-9_]{30,}",
    rb"CfDJ[A-Za-z0-9_-]{80,}",
    rb'https?://[^\s"<>]+[?&](?:X-Amz-Signature|X-Goog-Signature|Signature)=',
]
failed = []
for name in filter(None, names):
    data = subprocess.check_output(["git", "show", ":" + name])
    p = Path(name)
    if p.parts[0] in {"state", "data", "runs", "cache", "private", "backups"}:
        failed.append(name + ": private directory")
    if len(data) > 2_000_000:
        failed.append(name + ": exceeds 2 MB source limit")
    if any(re.search(pattern, data) for pattern in patterns):
        failed.append(name + ": possible credential")
    if p.suffix == ".ipynb":
        notebook = json.loads(data)
        if any(
            c.get("outputs") or c.get("execution_count") is not None
            for c in notebook["cells"]
            if c["cell_type"] == "code"
        ):
            failed.append(name + ": notebook contains outputs")
if failed:
    print("\n".join(failed), file=sys.stderr)
    raise SystemExit(1)
print(
    f"Checked {len(list(filter(None, names)))} staged files: no detected secrets, private state, large files, or notebook outputs."
)
