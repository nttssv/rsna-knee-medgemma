"""Read-only exact-pod evidence. This is NOT a live-runtime observation adapter.

Unknown state/billing fields deliberately prevent a ready receipt. No requested
resource, catalogue price, desired state or account total fills missing facts.
"""
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'analysis/qwen_execution_provisioning_runtime_v1/scripts'
PINS = {
    'network_supervisor': '674db9abc3bacc4c9ed2a5aa36acd9451a021b26846515891426db19938d21e3',
    'provider_transport': '12310cc5fb7824f5e296553fafee5648d41bb8efc78fc527d786d1585232661a',
}
URL = 'https://api.runpod.io/graphql'
POD_QUERY = '''query QwenEnvironmentPod($input:PodFilter) {
 pod(input:$input) { id desiredStatus gpuCount containerDiskInGb volumeInGb
 networkVolumeId costPerHr adjustedCostPerHr imageName dockerArgs
 runtime { uptimeInSeconds }
 machine { gpuTypeId gpuDisplayName secureCloud dataCenterId } }
}'''
GPU_QUERY = '''query QwenEnvironmentGpu($ids:[String!]!) {
 gpuTypes(input:{ids:$ids}) { id displayName memoryInGb }
}'''
REQUIRED = ('pod_id', 'provider_state', 'current_total_usd_per_hour',
    'actual_compute_usd_per_hour', 'actual_storage_usd_per_hour', 'gpu_name',
    'gpu_count', 'gpu_vram_gb', 'cloud_type', 'region', 'container_disk_gb',
    'persistent_volume_gb', 'network_volume_id')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_frozen(name):
    # Verify both because provider_transport itself imports network_supervisor.
    for module, digest in PINS.items():
        if sha((FROZEN / (module + '.py')).read_bytes()) != digest:
            raise ValueError('Frozen transport source drift')
    spec = importlib.util.spec_from_file_location('prep_' + name, FROZEN / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aware(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return result


def private_write(path, value):
    """Exclusive local evidence, never overwrite, never a spending grant."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        raise


def http_worker(payload):
    """No caller-supplied GraphQL, URL, headers, HTTP verb or mutation path."""
    transport = load_frozen('provider_transport')
    kind, identity = payload['kind'], payload['identity']
    if kind == 'pod' and isinstance(identity, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identity):
        query, variables = POD_QUERY, {'input': {'podId': identity}}
    elif kind == 'gpu' and isinstance(identity, str) and re.fullmatch(r'[A-Za-z0-9 ._-]{1,100}', identity):
        query, variables = GPU_QUERY, {'ids': [identity]}
    else:
        raise ValueError('Invalid read-only request')
    key = transport._private_key(Path(payload['key_path']))
    request = urllib.request.Request(URL,
        data=json.dumps({'query': query, 'variables': variables}).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                 'User-Agent': 'rsna-qwen-environment-audit/1.0', 'Cache-Control': 'no-cache, no-store'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), transport._NoRedirect())
    started = datetime.now(timezone.utc).isoformat()
    with opener.open(request, timeout=10) as response:
        if response.status != 200 or response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Unverified response')
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or key.encode() in raw:
        raise ValueError('Unverified response')
    data = json.loads(raw, object_pairs_hook=transport._json_object_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(data, dict) or data.get('errors') or not isinstance(data.get('data'), dict):
        raise ValueError('Incomplete GraphQL response')
    return {'ok': True, 'endpoint': URL, 'operation': kind,
            'identity': identity, 'started_at': started,
            'received_at': datetime.now(timezone.utc).isoformat(),
            'query_sha256': sha(query.encode()), 'response_sha256': sha(raw),
            'body_sha256': sha(json.dumps(data['data'], sort_keys=True, allow_nan=False).encode()),
            'body': data['data']}


def validate_envelope(envelope, kind, identity, now):
    expected_query = POD_QUERY if kind == 'pod' else GPU_QUERY
    if (envelope.get('ok') is not True or envelope.get('endpoint') != URL
        or envelope.get('operation') != kind or envelope.get('identity') != identity
        or envelope.get('query_sha256') != sha(expected_query.encode())
        or not re.fullmatch(r'[0-9a-f]{64}', envelope.get('response_sha256', ''))):
        raise ValueError('Evidence provenance mismatch')
    if envelope.get('body_sha256') != sha(json.dumps(envelope['body'], sort_keys=True, allow_nan=False).encode()):
        raise ValueError('Evidence body changed')
    start, end = aware(envelope['started_at']), aware(envelope['received_at'])
    if not start <= end <= now or (now - start).total_seconds() > 600:
        raise ValueError('Evidence stale or future-dated')
    if (end - start).total_seconds() > 30:
        raise ValueError('Evidence collection exceeded bound')


def normalize(pod_id, pod_envelope, gpu_envelope=None, *, now=None):
    now = now or datetime.now(timezone.utc)
    validate_envelope(pod_envelope, 'pod', pod_id, now)
    raw = pod_envelope['body'].get('pod')
    if not isinstance(raw, dict) or raw.get('id') != pod_id:
        raise ValueError('Exact pod not found')
    machine = raw.get('machine') if isinstance(raw.get('machine'), dict) else {}
    fields = {k: {'value': None, 'verified': False, 'source': None} for k in REQUIRED}

    def put(name, value, source, valid=True):
        if valid:
            fields[name] = {'value': value, 'verified': True, 'source': source}

    put('pod_id', raw['id'], 'pod.id')
    for name, key in [('gpu_count', 'gpuCount'), ('container_disk_gb', 'containerDiskInGb'),
                      ('persistent_volume_gb', 'volumeInGb')]:
        value = raw.get(key)
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        if name == 'gpu_count':
            valid = type(value) is int and value > 0
        put(name, value, 'pod.' + key, valid)
    if 'networkVolumeId' in raw:
        value = raw['networkVolumeId']
        put('network_volume_id', value, 'pod.networkVolumeId', value is None or (isinstance(value, str) and bool(value)))
    for name, key in [('gpu_name', 'gpuTypeId'), ('region', 'dataCenterId')]:
        value = machine.get(key)
        put(name, value, 'pod.machine.' + key, isinstance(value, str) and bool(value))
    secure = machine.get('secureCloud')
    put('cloud_type', 'SECURE' if secure is True else 'COMMUNITY',
        'pod.machine.secureCloud', type(secure) is bool)
    if gpu_envelope is not None:
        identity = machine.get('gpuTypeId')
        validate_envelope(gpu_envelope, 'gpu', identity, now)
        matches = gpu_envelope['body'].get('gpuTypes')
        if not isinstance(matches, list) or len(matches) != 1 or matches[0].get('id') != identity:
            raise ValueError('GPU catalogue join is not exact and unique')
        memory = matches[0].get('memoryInGb')
        put('gpu_vram_gb', memory, 'gpuTypes[id == pod.machine.gpuTypeId].memoryInGb',
            type(memory) is int and memory > 0)
    # desiredStatus is intent, runtime != null is telemetry, costPerHr is an
    # undivided running quote. None proves the required actual state/rate split.
    return {'schema': 'qwen_environment_observation_v1', 'observed_at': pod_envelope['received_at'],
        'fields': fields, 'provider_reported': {
            'desired_state': raw.get('desiredStatus'), 'runtime': raw.get('runtime'),
            'cost_per_hour': raw.get('costPerHr'), 'adjusted_cost_per_hour': raw.get('adjustedCostPerHr'),
            'image': raw.get('imageName'), 'docker_args': raw.get('dockerArgs')},
        'evidence': {'pod': pod_envelope, 'gpu': gpu_envelope},
        'unknown_fields': [k for k, v in fields.items() if not v['verified']],
        'ready_for_live_use': False, 'execution_enabled': False,
        'limitations': ['Not a provider-signed receipt; local hashes prove provenance/integrity only.',
                       'Catalogue VRAM describes the assigned type, not measured device/free memory.',
                       'Actual state and separate rates/current billing unavailable from this API contract.']}


def resource_differences(observation):
    """Compare known provider facts; never copy expected values into evidence."""
    policy = json.loads((Path(__file__).resolve().parents[1] / 'configs/preparation.json').read_text())
    result = {}
    for name, expected in policy['expected_resource'].items():
        field = observation['fields'][name]
        if not field['verified']:
            result[name] = 'UNKNOWN'
        elif field['value'] != expected:
            result[name] = 'MISMATCH'
    return result


def require_complete(observation):
    # Never trust a user-edited ready flag. This schema deliberately cannot clear
    # the state/billing gap; a separately reviewed source adapter is required.
    raise ValueError('This read-only evidence schema cannot authorize live progression')


def collect(pod_id, key_path):
    network = load_frozen('network_supervisor')
    deadline = datetime.now(timezone.utc) + timedelta(seconds=30)
    def call(kind, identity):
        value = network.supervise([sys.executable, str(Path(__file__).resolve()), '--worker'],
            {'kind': kind, 'identity': identity, 'key_path': str(Path(key_path).absolute())},
            timeout=14, deadline=deadline)
        if not value.get('ok'):
            raise ValueError('Read-only provider query failed; no raw error retained')
        return value
    pod = call('pod', pod_id)
    partial = normalize(pod_id, pod)
    identity = partial['fields']['gpu_name']['value']
    gpu = call('gpu', identity) if identity else None
    return normalize(pod_id, pod, gpu)


def main():
    if sys.argv[1:] == ['--worker']:
        try:
            raw = sys.stdin.buffer.read(65537)
            if len(raw) > 65536:
                raise ValueError('Oversize request')
            value = http_worker(json.loads(raw))
        except Exception:
            value = {'ok': False, 'error': 'read_only_query_unverified'}
        print(json.dumps(value, allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.pod_id, args.key_file)
    result['resource_differences'] = resource_differences(result)
    private_write(args.output, result)
    print(json.dumps({'ready_for_live_use': False, 'unknown_fields': result['unknown_fields']}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
