# Future deployment checklist — preparation only

All items below remain gated. This file contains no spending authorization and no executable provider mutation command. Nothing was deployed during this milestone.

| Component | Prepared source/inventory | Required future evidence |
|---|---|---|
| External shutdown worker | Frozen provisioning worker, transport and network-supervisor hashes in [inventory](configs/deployment_inventory.json) | An independent Linux control host with verified `/proc` identity, detached worker challenge/response and private credentials; remains alive if allocator/pod fails |
| Exact-pod stop preflight | Frozen `stop_watchdog.verify_stop_preflight()` and `FrozenInnerControl` | Genuine read/stop/read-back on an already-stopped exact pod, same credential context, exact runpodctl executable path and modern/legacy syntax; private receipt with unchanged required fields |
| Pod-local watchdog | Unchanged `analysis/qwen_execution_v1/scripts/stop_watchdog.py` | Exact `RUNPOD_POD_ID`, live OS-verified PID/argv/environment and receipt; stop exactly outer deadline minus 300 seconds; checked before every generation |
| Prepared inputs/source | Validated fixed five-report package and protected baseline | Copy privately; rehash source/plan/inputs/prompts/token artifacts/split after transfer; no re-preparation or report selection |
| Model cache | Frozen Qwen weight manifest: 13 files, 29,548,135,507 total bytes | Full exact revision and digest audit on the pod; no substitution, offload or download inside the inference runner |
| Inner runner/handoff | Unchanged reviewed plan and co-located handoff | Exact A40/CA-MTL-1, hardware/BF16/CUDA/Torch/dependency/cache/token checks, independent decode, parent supervision and original grants/receipts |

## Sequence prerequisites to resolve before any enablement

1. Obtain independently proven exact A40/CA-MTL-1 state, current total billing and actual compute/storage rates. No A40 allocation exists in the observed pod list. The older RTX/US-WA-1 pod is not a substitute. Do not create one merely to complete this checklist.
2. Complete final-image startup inspection and verify the effective sleep command. No automatic Qwen download/load/inference is permitted during initial provisioning. The inspected default `/start.sh` starts services and may run `/pre_start.sh`; do not invoke it as a shortcut.
3. Establish a reviewed remote deployment/control channel compatible with the frozen request. It exposes no ports/public IP and sleep bypasses the default SSH/service setup. A working SSH server cannot be assumed. No key/image/command/port edits are authorized here.
4. Resolve the pre-resume co-location dependency: the existing handoff's `validate_before_resume()` requires `RUNPOD_POD_ID`, pod-local files and a live pod-local watchdog while the controller has not resumed that pod. A stopped pod cannot run that code. No external receipt substitution or silent reordering is permitted. This is a concrete deployment blocker requiring a separately reviewed integration decision, not a scientific change made here.
5. Prepare the complete cache on an authorized staging host before paid time when possible. Neither checked local cache contains the pinned 13-file set. Transfer/download timing and disk headroom are unmeasured. The container disk is ephemeral across restarts; do not rely on pre-stop staging surviving the single later resume. The [provider disk contract](https://docs.runpod.io/api-reference/pods/POST/pods) documents this behavior.
6. Verify byte-identical source/package/cache at the final destination after resume, within the same original clock. These future operations must fit the single outer <=3,600-second window from first create intent, including setup, first stop, later resume and inference, with the final 300 seconds reserved for shutdown. Cumulative spend stays <=$1.50; compute <=$0.49/hour and storage <=$0.012/hour. No clock/budget reset, second create/resume, fallback, alternate host or retry.
7. Only a separate explicit operator decision may authorize exact resources and reviewed hashes. This milestone creates no grant, preflight receipt, live observation or watchdog receipt. Do not use an inventory JSON as one.
8. After a future authorized attempt, preserve/hash/copy partial or complete artifacts when possible, stop immediately on completion/failure, independently verify stopped state and $0/hour, then revoke the temporary control credential. This stage does none of those mutations.

The standalone historical `scripts/runpod_stopped_preflight.py` uses GraphQL and does **not** produce the frozen runpodctl receipt schema. It is not a substitute for the pod-local control path. No CLI stop test was executed in this milestone.
