# Live-environment preparation results

**BLOCKED — preparation only, not execution-ready.** Base commit `428c493b3fc390e6dff8f9c320299223d309188b`; checks performed 2026-09-25 UTC. All 322 previously tracked files and the frozen scientific/runtime artifacts are unchanged. No grants, provider mutations, deployments, cache downloads, model loads or generations occurred. Every live switch remains false.

## Provider facts independently observed

The exact existing pod was read through authenticated fixed GraphQL queries at **03:15:44 UTC**, with an earlier same-session signed-in console check. Its identifier and raw receipts stay private. The visible list contained six existing RTX/A100 pods and no A40 allocation. The existing pod is **not** the proposed A40/CA-MTL-1 resource.

| Required fact | Observed evidence | Limit |
|---|---|---|
| Exact pod ID | API response matched the requested existing pod exactly; same inspector ID | No new A40 pod ID exists |
| Provider state | API desired `EXITED`, runtime null; console compute/storage both “Not running” | Actual machine-state enum is not proven by API; no RUNNING observation |
| Current total billing | Exact-pod console displayed **$0.00/hour** | Human-readable snapshot only; machine path keeps unknown |
| Actual compute hourly rate | API running cost/adjusted cost both **$0.84/hour**; console Start quote **$0.84/hour** | Quote is not an independently itemized actual compute-rate fact |
| Actual storage hourly rate | Console “Not running” | No numeric actual storage rate; unknown |
| GPU / count | `NVIDIA RTX 6000 Ada Generation`, **1** | Mismatches proposed A40 |
| GPU VRAM | **48 GB** via exact returned GPU-type ID joined to provider catalogue | Specification, not measured device/free VRAM |
| Cloud / region | **Secure**, **US-WA-1** | Region mismatches CA-MTL-1 |
| Container disk | **80 GB** | Verified allocation field |
| Persistent volume | **0 GB** | Verified `volumeInGb` |
| Network volume | Explicit **null** | No network volume attached to this pod |
| Host health | Console warned of a critical machine error | Does not establish capacity or resumability |

The collector's machine-readable result has four explicit unknowns: actual provider state, current total billing, actual compute rate, actual storage rate. It cannot be supplied to the frozen live runtime. No missing field was copied from a requested allocation, catalogue price or budget. The [schema and evidence protocol](PROTOCOL.md) documents these distinctions.

## Bootstrap result: partial static verification

The exact immutable Linux amd64 manifest and config hashes match the frozen proposal:

- Manifest: `4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`
- Config: `02b731f844d2fbc4dd0de87253081363864327c7931b1ae2b5daa60abc600c51`

At **03:07:17 UTC**, 19 hash-verified small layers totaling **168,359 compressed bytes** were inspected without execution/extraction. Fifteen layers totaling **10,564,172,581 compressed bytes** were omitted by the bounded inspector. The [layer/file digest summary](configs/bootstrap_summary.json) records exact coverage.

Inspected startup artifacts:

| Artifact | SHA-256 / finding |
|---|---|
| NVIDIA entrypoint | `de08317631ccc093aba39d5f2bc1e2ca9070cff49bc9a11eb550e6e8794de7f9`; processes `.txt`/`.sh` hooks in sorted order, then execs supplied arguments |
| Seven NVIDIA hook/text files | Banner/version/copyright/license/driver/internal/deprecation checks; exact individual hashes in the summary |
| Default `/start.sh` | `254b15032479ca9f586162e373ee6893e9e4a2dae4c2a4e795e300c7e63ce0c0`; starts nginx, optional pre-start script, SSH/Jupyter setup and sleep |

No explicit Qwen download/load/inference appears in those inspected NVIDIA entrypoint/hook versions. That does **not** prove the final image or actual provider startup is free of such activity: omitted layers could add/replace files and dependencies.

The [RunPod API command contract](https://docs.runpod.io/api-reference/pods/POST/pods), pinned config and frozen create-body test imply this argv:

```text
/opt/nvidia/nvidia_entrypoint.sh /bin/sleep infinity
```

This is documented/static command mapping, **not observed live mapping**. The default `/start.sh` should be bypassed under that contract; SSH/service startup therefore cannot be assumed. No container was executed, no runtime PID 1 inspected, and absence of automatic model activity is not certified.

## Deployment preparation and tests

The unchanged inner validator passed for prepared plan `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc` and execution plan `ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5`. Package/source/token artifact/split fingerprints passed; no tokenization or model settings were changed.

The [deployment inventory](configs/deployment_inventory.json) binds the frozen worker, transport, supervisor, handoff, inner runner, original runpodctl receipt verifier and watchdog. It is an inventory, not a deployed bundle. The exact cache requires **13 files / 29,548,135,507 bytes**, including **29,536,665,640 weight bytes**. Both checked local cache roots had **0/13** verified files for the pinned revision. This does not claim that no other offline copy exists. Nothing was downloaded or loaded.

Validation: **62 new CPU tests**, plus **284 frozen focused regression tests**, **346 passed**. Tests cover exact-pod/source/time/body binding, unknown fields, missing-versus-null volume, requested-versus-observed resource drift, catalogue identity/cardinality, fixed-query-only HTTP, shared hard supervision, no retry, private exclusive writes, image digest/size/archive hazards, incomplete-layer refusal, command mapping and all protected baseline hashes. Synthetic tests are not provider-state, live-stop or runtime bootstrap evidence. Existing CI does not discover this new analysis test directory automatically; the explicit command below was run locally.

```sh
python -m pytest -q analysis/qwen_live_environment_v1/tests \
  analysis/qwen_execution_provisioning_runtime_v1/tests \
  analysis/qwen_execution_runtime_alt_v1/tests \
  analysis/qwen_execution_resource_alt_v1/tests \
  tests/test_runpod_stopped_preflight.py
```

## Remaining concrete blockers

1. No matching exact A40/CA-MTL-1 allocation was observed. The old RTX/US-WA-1 pod is mismatched and has a host warning. No capacity/start inference is permitted.
2. Provider observation cannot yet prove actual state plus numeric current billing and separate actual compute/storage rates automatically. The existing frozen REST adapter remains unchanged and blocked.
3. Final-image startup contents and actual RunPod command mapping/no-model-activity behavior are not fully verified.
4. No verified independent Linux shutdown host or pod-local runpodctl/watchdog deployment exists. Sleep-only startup with no ports/public IP has no proven remote transfer/control channel.
5. The frozen pre-resume handoff requires a live pod-local watchdog/environment on a stopped pod. That sequencing cannot be satisfied as written; documented in the [checklist](DEPLOYMENT_CHECKLIST.md), not redesigned here.
6. Complete cache staging/transfer, post-restart integrity and timing within the unchanged 60-minute/$1.50 boundary remain unverified. Container disk persistence cannot be assumed across the required stop/resume.
