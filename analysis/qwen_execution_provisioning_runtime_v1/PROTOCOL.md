# Live component protocol (disabled)

This implementation makes no change to the experiment or resource ceilings. No actual approval record is created by code preparation. All production switches in [runtime.json](configs/runtime.json), the transport mutation switch, the external-worker switch and the frozen inner execution switch remain false.

## Frozen bindings

- Provisioning proposal: `0770846122bb4bb4953c7e728afa1c254ae569c8f756b3e905e6c095a2ed2842`.
- Inner execution plan: `ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5`.
- Prepared plan: `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`.
- Alternate resource proposal: `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`.

Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, SDPA, batch size 1, five reports per block, control-1/candidate-1/candidate-2/control-2, maximum 20 generations, token caps, prompts, parsers and split remain frozen. No training, validation, full-development, bulk extraction, repair or inference retry is introduced.

## Source-bound authorization and single dispatch

The prospective private outer approval retains exact resource/rate, image/CMD, intent, timestamps, budget and frozen hash fields. It additionally binds `live_source_manifest_sha256` and an absolute private `operation_ledger_dir`. This is an authorization design; no real grant is generated. The source manifest covers every Python implementation file and the outer runtime policy. The unchanged inner approval schema is not extended.

The ledger uses an immutable intent-level binding and separate exclusive, fsynced create/resume/handoff claims. Claims remain consumed after failure or allocator death. A new object or output directory cannot replay the same intent. No resume/recovery of an interrupted controller is implemented. The later resume belongs to the original uninterrupted controller session.

The controller writes its outer intent before the initial exact-name preread; this conservatively includes read/guard setup latency in the same at-most-60-minute clock. A nonempty preread fails without creation or stopping. It verifies the detached worker, writes a durable create-attempt receipt, and requires a fresh worker acknowledgment of cached creation ownership before dispatching exactly one create request. A lost response triggers exact-name reconciliation, never a second request. A uniquely owned mismatched allocation is bound only for cleanup and is not accepted for use.

The exact allocation is Secure A40/CA-MTL-1, count 1, 48 GB, disk 80 GB, zero persistent and null network volume, pinned image digest and sleep CMD. Returned unknown fields fail closed. Once bound, physical machine identity cannot change. Stop requests are permitted on failure even when local receipt writing fails. A stopped mutation response alone is never treated as independent billing confirmation.

## Hard transport supervision

Every REST call runs in a new POSIX process group. Its external bound is the smaller of the requested call limit (at most 30 seconds) and remaining outer window, with cleanup time reserved. A monotonic bound prevents clock rollback from extending the initial allowance. The supervisor kills the group and reaps the direct child on success, timeout and failure; Linux subreaping also reaps adopted descendants. On macOS the system reaps orphan descendants after group kill. There is no Python thread/signal timeout assumption for an HTTP worker.

The worker has bounded input/output, no inherited credential environment, no proxy environment, no redirects and no automatic HTTP retry. Credentials are read only from an owner-only regular 0600 file, supplied by private path rather than as a token in argv. Errors never return the token, request URL, provider error body or raw stderr. The documented REST endpoint is fixed. HTTP fixtures run on loopback only.

## Independent external worker

The worker starts in a separate process group/session before the create request, with stdin detached and private configuration. Verification checks OS PID/start identity, command, environment nonce, owner and session, plus a fresh private Unix-socket HMAC challenge. Configuration binds intent/hash/deadlines, source snapshot and credential identity. A JSON PID receipt alone cannot pass.

The worker caches the verified empty preread and durable creation-attempt ownership before acknowledging it. It survives allocator exit and later ledger I/O damage. Original-source drift triggers cleanup using its pinned private transport snapshot. At T−300 seconds, or earlier allocator death, it reconciles only the exact high-entropy owned name. Zero/multiple matches never authorize guessing an ID. Read-only reconciliation can repeat for delayed visibility; create/resume cannot. The worker makes at most three bounded stop attempts and verifies stopped state/$0 through a separate read path. Unverifiable billing remains an explicit failure through the hard deadline. The independent worker does not replace the frozen pod-local watchdog.

## Later resume and handoff

The same controller requires a new valid inner grant/observation, genuine stopped readback, original runpodctl stopped preflight, live external guard and unchanged source/ledger. It preserves the original outer deadline and T−300 stop time and requires `accrued_upper_usd + inner.maximum_usd <= outer.maximum_usd <= 1.50`. Cost uses Decimal and rounds upward at the $0.502/hour resource ceiling only when the actual allocation/rates were verified. Unknown or mismatched prices produce an unverified cost bound and prohibit progress.

Freshness is rechecked immediately at dispatch after reads/preflight/guard work. One exclusive resume claim precedes the only resume call. A changed physical machine, failed/ambiguous response or incomplete inner run ends the experiment and triggers immediate stop; there is no fallback or continuation.

`InnerRunnerHandoff` passes the exact prepared-plan/execution-plan hashes and original grant, watchdog, preflight, observation, cache and output paths to the unchanged inner `launch_supervised()`. The inner launcher still owns parent attestation, hard inference timeout, per-generation watchdog checks, cache/token/GPU checks, durable attempt/raw/parse ordering and independent decoding. The handoff refuses the frozen disabled gate before paid resume. It requires co-located pod files/control environment and never converts external-worker receipts into pod-local receipts. Remote deployment and model-cache preparation are outside this implementation.

## Concrete blockers before any live operation

1. **All switches remain false and there is no current source-bound spending approval.** Source implementation is not enablement or authorization.
2. **Provider observation:** documented REST fields do not prove actual state/current zero billing, separate actual rates or VRAM. The factory rejects the known gap before create. A trustworthy current observation source must be integrated; unknown facts cannot be filled from the requested configuration.
3. **Deployment:** the live external worker requires a Linux operator environment for OS attestation. The unchanged inner watchdog, exact runpodctl preflight, prepared package, complete pinned cache and handoff must be deployed on the approved pod. No remote bootstrap/SSH transfer or cache download was added.
4. **Image startup/capacity:** inherited entrypoint/hooks and effective provider CMD behavior still need verification; sleep CMD alone does not prove the absence of other startup activity. Exact allocation/physical-host availability is not established by CPU tests.

After a future authorized operation, preserve/hash/copy artifacts when possible, independently verify provider stopped/$0, and revoke the temporary control credential. This implementation does not revoke credentials or modify any provider resource.

Official API references: [create](https://docs.runpod.io/api-reference/pods/POST/pods), [read](https://docs.runpod.io/api-reference/pods/GET/pods/podId), [list](https://docs.runpod.io/api-reference/pods/GET/pods), [start](https://docs.runpod.io/api-reference/pods/POST/pods/podId/start), [stop](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop).
