#!/usr/bin/env bash
set -euo pipefail
RSNA_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RSNA_REPO"
RSNA_PYTHON="${RSNA_PYTHON:-python3.12}"
"$RSNA_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ "${1:-cpu}" == gpu ]]; then
  # Linux CUDA environment. The host must already have a compatible NVIDIA driver.
  .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -c requirements/pilot-constraints.txt -e '.[gpu,tracking,kaggle,test]'
else
  .venv/bin/python -m pip install -c requirements/pilot-constraints.txt -e '.[tracking,test]'
fi
.venv/bin/rsna-knee doctor

