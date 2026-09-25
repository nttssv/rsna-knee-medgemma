# RunPod API 403/1010 recovery — 25 September 2026

At 11:35–11:37 UTC (19:35–19:37 Singapore time), a read-only local comparison reproduced HTTP 403 with body `error code: 1010` using Python urllib's default User-Agent. The same public GraphQL query returned HTTP 200, 50 GPU types and no GraphQL errors with `User-Agent: rsna-qwen-evidence-selection/1.0`. An authenticated exact-pod query with that header then returned the stopped pod's correct identity, GPU, region, disk and volume fields. An authenticated REST read also succeeded.

The immediate cause of the failed diagnostic was its default HTTP client identification. The earlier conclusion that the account key or watchdog could not access RunPod was unsupported: those probes omitted the application User-Agent already present in the reviewed watchdog. This recovery was also documented in the historical [control preflight](../analysis/report_labeling_llm_v3_execution_v1/CONTROL_PREFLIGHT.md). Cloudflare describes [1010](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/) as a client-signature denial; a 403 alone does not establish invalid credentials or denied stop permissions.

No runtime-source change is needed. `stop_at.py` already identifies this application explicitly. Do not impersonate a browser, disable TLS checks, fabricate receipts, change prompts, or reset a session clock to resolve this issue. A successful read is not a successful live-stop test.

At 11:38:58 UTC, the unchanged `stop_at.stop_once()` successfully stopped the user-selected incompatible 24 GB pod. An independent GraphQL read returned `EXITED`; a refreshed signed-in console showed compute and storage not running and total `$0.00/hour`. This verifies the live API stop transport for that observed pod and credential, not every future allocation. No Qwen cache, report transfer or generation occurred on that pod. The user chose to select another 48 GB pod.

## Read-only diagnostic used successfully

From the repository root, set `RUNPOD_KEY_FILE` to the existing private owner-only credential file and `RUNPOD_POD_ID` to the user-selected pod. Never paste the key into a command, URL, terminal history or Git.

```bash
python3 - <<'PY'
import json, os, stat
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

key_file = Path(os.environ['RUNPOD_KEY_FILE'])
assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
key = key_file.read_text().strip()
query = '''query($input: PodFilter) {
  pod(input: $input) {
    id name desiredStatus gpuCount containerDiskInGb volumeInGb
    networkVolumeId costPerHr adjustedCostPerHr
    machine { gpuTypeId secureCloud dataCenterId }
  }
}'''
request = Request('https://api.runpod.io/graphql', method='POST',
    data=json.dumps({'query': query,
        'variables': {'input': {'podId': os.environ['RUNPOD_POD_ID']}}}).encode(),
    headers={'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json',
        'User-Agent': 'rsna-qwen-evidence-selection/1.0'})
try:
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    if result.get('errors'):
        raise RuntimeError('GraphQL errors; do not infer provider readiness')
    print(json.dumps(result['data']['pod'], indent=2))
except HTTPError as exc:
    # Do not print request headers or credential-bearing URLs.
    raise SystemExit(f'RunPod read failed: HTTP {exc.code}') from None
PY
```

Keep the exact-pod output private. `costPerHr` is not proof of current total billing or separate storage pricing. Confirm those using the signed-in console. Verify the actual GPU's VRAM and allowlist membership before setup: a 24 GB GPU cannot satisfy this experiment's ≥36 GiB free-memory gate.

## Recovery order

1. While compute is stopped, use the application User-Agent for both the public diagnostic and authenticated exact-pod query. Distinguish HTTP rejection from a GraphQL authorization error.
2. Confirm the selected pod, actual GPU/VRAM, region, rates and storage. Retain the original cumulative budget/deadline when the user requests a switch within the session.
3. Validate the real shutdown transport rather than drawing conclusions from a differently configured ad hoc probe. Keep the external stop route and required pod-local watchdog.
4. Use the [existing A/B runner](../analysis/qwen_evidence_selection_v1/README.md) for setup/cache audit/input parity/inference/copy/stop. Do not retry a model generation to recover an infrastructure issue.
5. Record attempted/completed/unrun counts, copy and hash outputs, verify provider stopped state and billing, and revoke temporary control access after the session.

The reviewed execution commit is `1ab7b5bbc9638fc81eac17ec52e3a104c56d622a`; execution-plan SHA remains `6d551f7bf19db648e11eed9766e6a47b984567e6b8fbc6e4dd660c738bf9c54a`. This document records an operational correction, not new generation results or a new approval.
