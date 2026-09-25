# Evidence protocol

## Provider observation

The collector runs only two fixed GraphQL **queries** through TLS at `api.runpod.io/graphql`: exact pod lookup, then GPU catalogue lookup by the GPU type ID actually returned for that pod. The HTTP verb is POST because GraphQL uses POST; the body cannot be supplied by a caller and contains no mutation. No arbitrary URL, query, method, redirect, proxy environment, secret field or account billing aggregate is accepted. Credentials remain in the subprocess, read from an owner-only `0600` regular file. Failures are sanitized.

Each operation is supervised by the source-pinned frozen process-group network supervisor, with a 14-second hard bound and one shared 30-second collection deadline. No query retry occurs. Responses are bounded, duplicate keys/nonfinite JSON and GraphQL errors are rejected. Query/raw-response/normalized-body hashes, endpoint, identity, request and receipt times are retained. The body hash detects accidental drift; these are local capture records, **not provider-signed attestations**. Freshness requires timezone-aware timestamps, ordered collection times, age <=600 seconds and collection duration <=30 seconds. Persisted evidence is historical and must not be used as a current live receipt.

Pod identity must match exactly. Missing/null/invalid resource values remain unverified, except explicit `networkVolumeId: null`, which independently establishes no network-volume attachment. Pod system RAM is never used as GPU VRAM. Catalogue memory is joined only through the actual assigned type ID and must have exactly one matching result; it is a specification, not measured hardware/free VRAM.

The [provider schema](https://graphql-spec.runpod.io/) exposes desired status, runtime telemetry, running cost fields and allocation metadata, but not a trustworthy exact-pod actual-state/current-billing/component-rate tuple. RunPod describes `desiredStatus` as expected status and `costPerHr` as running cost; neither supports silently substituting actual state, storage rate or current zero billing. See the [official Pod API](https://docs.runpod.io/api-reference/pods/GET/pods/podId).

Consequently `provider_state`, current total, actual compute rate and actual storage rate remain unknown in this machine-readable path, for both expected STOPPED and RUNNING cases. `require_complete()` always refuses this intentionally partial schema, even if someone edits a ready flag. Requested A40/CA-MTL-1 values are used only for mismatch reporting. Price-list arithmetic, account-wide totals, an absent runtime, desired EXITED or a successful stop response never fill missing facts.

The signed-in console may provide independent human-readable billing/state evidence. That evidence is documented separately; it does not turn this collector into a live adapter. Full integration requires another reviewed provider evidence source and unchanged fail-closed safeguards. No frozen runtime is patched here.

## Image inspection

Anonymous Docker Hub pull authorization stays in memory. The exact Linux amd64 manifest and config are SHA-256 verified. Every fetched layer is size/hash verified. Archive members are inspected in memory, never extracted or executed. Unsafe paths, nonregular startup artifacts, oversize scripts and digest drift fail. Whiteouts and links remain visible; no merged filesystem is inferred. CDN redirects drop authorization.

Inspection is bounded to layers <=128 KiB under a 60-second outer process supervisor. Nineteen small layers were inspected; fifteen large layers were omitted. This deliberately produces **partial inspection**, not an attestation of the final image. Later overwrites, additional hooks, whiteouts, interpreters and library/environment effects in skipped layers remain unverified.

The [RunPod create contract](https://docs.runpod.io/api-reference/pods/POST/pods) says `dockerStartCmd` overrides CMD and the image ENTRYPOINT remains when not overridden. The frozen request sends `['/bin/sleep','infinity']`, omits an entrypoint override, uses no template and an empty explicit environment. Config plus that documented contract implies NVIDIA entrypoint followed by sleep. Actual RunPod mapping/PID 1 has not been observed for the pinned allocation. Static inspection cannot certify absence of all startup/model activity.

## Deployment preparation

The inventory hashes every baseline tracked file, validates the unchanged inner execution plan and calls its existing read-only prepared-package check. It calls only `audit_cache(..., 'qwen')`, never tokenizer/backend/model constructors or inference. Private reports are not printed. Prepared-file hashes and cache completeness summaries may be published; reports, credentials, weights and raw provider evidence may not.

The deploy inventory is not an archive uploader. It names the external worker, its transport/supervisor dependencies, controller/handoff, frozen inner runner, pod-local watchdog/preflight verifier and cache manifest. All live and deployment flags remain false. Source file presence is not proof of deployed Linux process identity, control access or usable remote transport. The [checklist](DEPLOYMENT_CHECKLIST.md) records those remaining prerequisites without running them.
