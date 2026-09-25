# Qwen live-environment preparation v1

**Read-only preparation complete; live readiness blocked.** This separate milestone starts from `428c493b3fc390e6dff8f9c320299223d309188b`. All 322 previously tracked files remain byte-identical. No provider mutation, model load, cache download, grant, or execution occurred. Every new and existing current-stage live switch remains false.

The [results](RESULTS.md) distinguish exact-pod facts, API gaps, partial image inspection, and deployment prerequisites. The [protocol](PROTOCOL.md) describes evidence and failure behavior. The [deployment checklist](DEPLOYMENT_CHECKLIST.md) is preparatory only; it is not a runnable start procedure or permission to spend.

| Tool | Purpose |
|---|---|
| [observe_provider.py](scripts/observe_provider.py) | Fixed read-only GraphQL queries; exact-pod resource facts, exact catalogue VRAM join, explicit unknown state/billing fields |
| [inspect_bootstrap.py](scripts/inspect_bootstrap.py) | Pinned OCI manifest/config and bounded small-layer inspection; no container execution or extraction |
| [prepare_deployment.py](scripts/prepare_deployment.py) | Frozen-source, five-report package and offline cache inventory; no transfers or model initialization |

From the repository root, these commands only read sources/provider information and create exclusive local evidence files. Choose new output filenames each time. Exit code **2** means evidence was saved but readiness remains blocked; exceptions indicate collection/verification failure. The scripts never return live readiness or produce a runtime-compatible approval/observation record.

```sh
python analysis/qwen_live_environment_v1/scripts/observe_provider.py \
  --pod-id "$EXACT_POD_ID" --key-file "$PRIVATE_RUNPOD_KEY_FILE" \
  --output state/operator_checks/environment-observation-new.json

python analysis/qwen_live_environment_v1/scripts/inspect_bootstrap.py \
  --output state/operator_checks/bootstrap-inspection-new.json

python analysis/qwen_live_environment_v1/scripts/prepare_deployment.py \
  --prepared state/runs/qwen-runtime-v1-20260915-prepared-final \
  --cache "$PRIVATE_HF_CACHE" \
  --output state/operator_checks/deployment-inventory-new.json

python -m pytest -q analysis/qwen_live_environment_v1/tests
```

Use only an existing authorized key file with mode `0600`. Never put its value in an argument, output, URL, or Git. Exact pod IDs and raw signed-in observations remain in ignored `state/`; this public directory contains code, hashes and aggregate findings only.

The [preparation policy](configs/preparation.json), [protected baseline](configs/protected_sources.json), [image evidence summary](configs/bootstrap_summary.json), [deployment inventory](configs/deployment_inventory.json), and [source manifest](configs/source_manifest.json) provide reproducible bindings. This milestone does not replace or modify the frozen provider adapter, execution plan, prompts, parsers, package or split.
