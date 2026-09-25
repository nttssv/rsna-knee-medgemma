# One-session SOP — verified 25 September 2026

This records the completed session commands with private connection paths abbreviated, not authorization to repeat it. Private session paths, keys and source reports stay local. Never reuse the completed approval. No creation/resume or generation retry was performed by these commands.

1. Connect using the selected pod's actual Connect address and a session-specific SSH config. The provider had no SSH key; the operator installed only the ephemeral public key through the authenticated Jupyter terminal API, then pinned the host key. Check `nvidia-smi`, actual Torch/CUDA/BF16, free disk and absence of another workload. Read pod identity from `/proc/1/environ`: noninteractive SSH did not inherit `RUNPOD_POD_ID`. Check RunPod API with `User-Agent: rsna-qwen-evidence-selection/1.0`; default urllib produced Cloudflare 1010/403 and was not evidence of invalid credentials.

2. Record original T0/deadline and rates. Arm local external shutdown and pod-local `stop_at.py --arm` using the exact session/plan before setup. Keep outside-pod API and console stop accessible. Transfer source/input archives; verify their hashes before extraction into `/workspace/repo` and `/workspace/private`.

3. Actual setup script (saved alongside outputs) used:

```sh
python -m venv --system-site-packages /workspace/qwen-venv
/workspace/qwen-venv/bin/python -m pip install --no-cache-dir transformers==5.12.0 huggingface-hub==1.19.0 accelerate==1.15.0 jinja2==3.1.6 numpy==2.5.3 pillow==11.0.0 safetensors==0.8.0 sentencepiece==0.2.2 tokenizers==0.22.2
```

`setup.sh` then checked free disk, called `snapshot_download()` with the frozen manifest's exact model/revision/file list and `max_workers=4`, and required `load_preflight.audit_cache(..., 'qwen')['complete']`. This downloaded only Qwen, not MRI/MedGemma. The exact working setup script is preserved in private session storage.

4. From `/workspace/repo`, this real CLI first passed dry-run without the final five live arguments, then ran through `launch.py` in offline mode:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /workspace/qwen-venv/bin/python analysis/qwen_evidence_selection_v1/scripts/run_ab.py \
 --prepared /workspace/private/prepared --source-prepared /workspace/private/source-prepared \
 --cache /workspace/qwen-cache \
 --plan-sha256 6d551f7bf19db648e11eed9766e6a47b984567e6b8fbc6e4dd660c738bf9c54a \
 --run --session /workspace/private/session.json --observation /workspace/private/observation.json \
 --shutdown-receipt /workspace/private/pod-stop-receipt.json --output /workspace/private/qwen-output
```

The detached parent/worker supervisor enforced 1,799 seconds; watchdog identity was checked before each generation. New output path only; no retry. Observe attempts/raw/parsed ledgers; schema/evidence failures are retained.

5. `scp -r -F .../ssh_config qwen-session:/workspace/private/qwen-output/A1 .../block-copies/` succeeded; repeated for completed B1/B2. After final `supervisor_result.json`, `python /workspace/private/package_results.py` created an archive and 33-file SHA manifest. Copy with `scp -F .../ssh_config qwen-session:/workspace/private/results-final.tar.gz .../results-final.tar.gz`; verify archive SHA and every extracted file before stop. The private package script and SSH configuration remain in this session directory.

6. Frozen `stop_at.stop_once(exact_pod_id, read_key(private_key_path))` succeeded. Read exact pod via REST with the application User-Agent; confirm EXITED plus console total $0/hour. Only then write the local watchdog cancellation marker. Disable the temporary control key. Do not terminate/delete the pod or claim killing Python stops billing.

Temporary API-key creation/private intake/shutdown/disable steps are in the [Luna/Sol operator guide](../../../../docs/RUNPOD_TEMPORARY_KEY_SOP.md).
