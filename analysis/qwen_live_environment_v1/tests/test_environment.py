"""CPU only: API fixtures, malicious/missing evidence and OCI archives."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import urllib.error

import pytest

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / 'scripts'))
import observe_provider as obs
import inspect_bootstrap as boot
import prepare_deployment as deploy

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


def envelope(kind, identity, body):
    return {'ok': True, 'endpoint': obs.URL, 'operation': kind, 'identity': identity,
        'started_at': (NOW - timedelta(seconds=2)).isoformat(),
        'received_at': (NOW - timedelta(seconds=1)).isoformat(),
        'query_sha256': obs.sha((obs.POD_QUERY if kind == 'pod' else obs.GPU_QUERY).encode()),
        'response_sha256': '0' * 64,
        'body_sha256': obs.sha(json.dumps(body, sort_keys=True, allow_nan=False).encode()), 'body': body}


def receipts(pod=None):
    raw = {'id': 'synthetic-pod', 'desiredStatus': 'EXITED', 'runtime': None,
        'gpuCount': 1, 'containerDiskInGb': 80, 'volumeInGb': 0, 'networkVolumeId': None,
        'costPerHr': 0.49, 'adjustedCostPerHr': 0.49,
        'machine': {'gpuTypeId': 'NVIDIA A40', 'secureCloud': True, 'dataCenterId': 'CA-MTL-1'}}
    if pod:
        pod(raw)
    identity = raw.get('machine', {}).get('gpuTypeId', 'NVIDIA A40')
    return envelope('pod', raw['id'], {'pod': raw}), envelope('gpu', identity,
        {'gpuTypes': [{'id': identity, 'memoryInGb': 48}]})


def normalize(p=None, g=None):
    if p is None:
        p, g = receipts()
    return obs.normalize('synthetic-pod', p, g, now=NOW)


def test_known_facts_never_complete():
    result = normalize()
    assert set(result['unknown_fields']) == {'provider_state', 'current_total_usd_per_hour',
        'actual_compute_usd_per_hour', 'actual_storage_usd_per_hour'}
    assert result['fields']['gpu_vram_gb']['value'] == 48
    assert obs.resource_differences(result) == {}
    with pytest.raises(ValueError):
        obs.require_complete(result)


@pytest.mark.parametrize('state,runtime,cost', [('EXITED', None, 0), ('RUNNING', {'uptimeInSeconds': 60}, .49), ('STOPPED', None, 0)])
def test_state_and_billing_not_inferred(state, runtime, cost):
    p, g = receipts(lambda r: r.update(desiredStatus=state, runtime=runtime, costPerHr=cost,
        provider_state='STOPPED', current_total_usd_per_hour=0,
        actual_compute_usd_per_hour=.49, actual_storage_usd_per_hour=.012))
    result = normalize(p, g)
    for name in ['provider_state', 'current_total_usd_per_hour', 'actual_compute_usd_per_hour', 'actual_storage_usd_per_hour']:
        assert result['fields'][name]['verified'] is False
        assert result['fields'][name]['value'] is None


@pytest.mark.parametrize('key', ['gpuCount', 'containerDiskInGb', 'volumeInGb', 'networkVolumeId'])
def test_missing_fields_unknown(key):
    p, g = receipts(lambda r: r.pop(key))
    result = normalize(p, g)
    expected = {'gpuCount': 'gpu_count', 'containerDiskInGb': 'container_disk_gb',
                'volumeInGb': 'persistent_volume_gb', 'networkVolumeId': 'network_volume_id'}[key]
    assert result['fields'][expected]['verified'] is False
    assert obs.resource_differences(result)[expected] == 'UNKNOWN'


@pytest.mark.parametrize('value', [True, '1', -1, 0, 1.0, None])
def test_bad_gpu_count_unknown(value):
    p, g = receipts(lambda r: r.update(gpuCount=value))
    assert not normalize(p, g)['fields']['gpu_count']['verified']


@pytest.mark.parametrize('field,value', [('gpuCount', 2), ('containerDiskInGb', 81), ('volumeInGb', 100), ('networkVolumeId', 'other-volume')])
def test_resource_drift(field, value):
    p, g = receipts(lambda r: r.update({field: value}))
    assert 'MISMATCH' in obs.resource_differences(normalize(p, g)).values()


@pytest.mark.parametrize('field,value', [('gpuTypeId', 'NVIDIA RTX 6000 Ada Generation'), ('secureCloud', False), ('dataCenterId', 'US-WA-1')])
def test_host_drift(field, value):
    p, g = receipts(lambda r: r['machine'].update({field: value}))
    assert 'MISMATCH' in obs.resource_differences(normalize(p, g)).values()


def test_missing_machine_and_catalogue_remain_unknown():
    p, _ = receipts(lambda r: r.pop('machine'))
    result = normalize(p)
    assert all(not result['fields'][k]['verified'] for k in ['gpu_name', 'gpu_vram_gb', 'cloud_type', 'region'])


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(endpoint='https://fake.invalid'),
    lambda p: p.update(identity='other-pod'),
    lambda p: p.update(operation='mutation'),
    lambda p: p.update(query_sha256='f' * 64),
    lambda p: p.update(body_sha256='f' * 64),
    lambda p: p.update(received_at=(NOW + timedelta(seconds=1)).isoformat()),
    lambda p: p.update(started_at=(NOW - timedelta(seconds=601)).isoformat()),
    lambda p: p.update(started_at=(NOW - timedelta(seconds=40)).isoformat()),
    lambda p: p.update(started_at='2026-09-25T00:00:00'),
    lambda p: p['body']['pod'].update(id='other-pod'),
])
def test_bad_evidence_rejected(mutation):
    p, g = receipts()
    mutation(p)
    with pytest.raises(ValueError):
        normalize(p, g)


@pytest.mark.parametrize('catalogue', [[], [{'id': 'wrong', 'memoryInGb': 48}],
    [{'id': 'NVIDIA A40', 'memoryInGb': 48}] * 2])
def test_catalogue_join_must_be_unique_exact(catalogue):
    p, _ = receipts()
    g = envelope('gpu', 'NVIDIA A40', {'gpuTypes': catalogue})
    with pytest.raises(ValueError):
        normalize(p, g)


def test_catalogue_memory_not_pod_system_ram():
    p, _ = receipts(lambda r: r.update(memoryInGb=200))
    assert not normalize(p)['fields']['gpu_vram_gb']['verified']


def test_forged_ready_flag_cannot_authorize():
    value = normalize()
    value['ready_for_live_use'] = True
    for field in value['fields'].values():
        field['verified'] = True
    with pytest.raises(ValueError):
        obs.require_complete(value)


def test_private_receipt_exclusive(tmp_path):
    p = tmp_path / 'observation.json'
    obs.private_write(p, {'ready': False})
    assert p.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        obs.private_write(p, {})


def test_frozen_network_pins_and_all_baseline_sources():
    assert deploy.verify_protected() == 322
    assert obs.load_frozen('network_supervisor').MAX_INPUT_BYTES == 65536


def test_no_live_or_deploy_switch():
    policy = json.loads((STAGE / 'configs/preparation.json').read_text())
    assert all(v is False for k, v in policy.items() if k.endswith('_enabled'))
    assert policy['maximum_cumulative_usd'] == 1.5
    assert policy['maximum_outer_provider_seconds'] == 3600


def layer(name='opt/nvidia/nvidia_entrypoint.sh', content=b'exec "$@"', kind=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as t:
        info = tarfile.TarInfo(name)
        info.size = len(content) if kind is None else 0
        if kind:
            info.type, info.linkname = kind, '/tmp/unsafe'
        t.addfile(info, io.BytesIO(content) if kind is None else None)
    return stream.getvalue()


def inspect(raw):
    return boot.inspect_layer(raw, 'sha256:' + obs.sha(raw), len(raw))


def test_tar_inspection_hashes_no_extraction():
    result = inspect(layer())
    assert result['startup_artifacts'][0]['sha256'] == obs.sha(b'exec "$@"')


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'a/../../escape'])
def test_unsafe_archive_path_rejected(name):
    with pytest.raises(ValueError):
        inspect(layer(name))


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_startup_symlink_rejected(kind):
    with pytest.raises(ValueError):
        inspect(layer(kind=kind))


def test_whiteout_visible_not_resolved():
    result = inspect(layer('opt/nvidia/.wh.entrypoint.d'))
    assert result['hazards'][0]['kind'] == 'whiteout'


def test_hash_and_size_mismatch_rejected():
    raw = layer()
    with pytest.raises(ValueError):
        boot.inspect_layer(raw, 'sha256:' + '0' * 64, len(raw))
    with pytest.raises(ValueError):
        boot.inspect_layer(raw, 'sha256:' + obs.sha(raw), len(raw) + 1)


def test_oversize_startup_artifact_rejected():
    with pytest.raises(ValueError):
        inspect(layer(content=b'x' * 65537))


def test_bootstrap_does_not_claim_complete_with_skipped_layers(monkeypatch):
    config = json.dumps({'architecture': 'amd64', 'os': 'linux', 'config': {
        'Entrypoint': ['/opt/nvidia/nvidia_entrypoint.sh'], 'Cmd': ['/start.sh']}}).encode()
    manifest = json.dumps({'config': {'digest': 'sha256:' + obs.sha(config), 'size': len(config)},
        'layers': [{'digest': 'sha256:' + 'a' * 64, 'size': boot.LAYER_LIMIT + 1}]}).encode()
    monkeypatch.setattr(boot, 'MANIFEST', 'sha256:' + obs.sha(manifest))
    monkeypatch.setattr(boot, 'CONFIG', 'sha256:' + obs.sha(config))
    def fake(url, limit, token=None):
        if 'auth.docker.io' in url:
            return b'{"token":"synthetic-public-token"}'
        return manifest if '/manifests/' in url else config
    monkeypatch.setattr(boot, 'fetch', fake)
    result = boot.inspect()
    assert not result['all_layers_inspected']
    assert not result['live_command_mapping_verified']
    assert not result['no_automatic_model_activity_verified']


def test_prep_does_not_load_model_or_deploy():
    source = (STAGE / 'scripts/prepare_deployment.py').read_text()
    import ast
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
             for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
             and isinstance(n.func, (ast.Attribute, ast.Name))]
    assert not {'create_backend', 'create_encoder', 'load_model', 'Popen', 'run', 'system', 'exec'} & set(calls)


def test_worker_only_sends_fixed_read_query(monkeypatch, tmp_path):
    transport = obs.load_frozen('provider_transport')
    key = tmp_path / 'key'
    key.write_text('synthetic-key')
    key.chmod(0o600)
    requests = []
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return b'{"data":{"pod":{"id":"synthetic-pod"}}}'
    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return Response()
    monkeypatch.setattr(obs, 'load_frozen', lambda _: transport)
    monkeypatch.setattr(obs.urllib.request, 'build_opener', lambda *args: Opener())
    result = obs.http_worker({'kind': 'pod', 'identity': 'synthetic-pod', 'key_path': str(key),
        'query': 'mutation { forbidden }', 'url': 'https://wrong.invalid'})
    assert requests[0].full_url == obs.URL
    assert json.loads(requests[0].data)['query'] == obs.POD_QUERY
    assert 'mutation' not in obs.POD_QUERY
    assert result['operation'] == 'pod'
    assert 'synthetic-key' not in json.dumps(result)


@pytest.mark.parametrize('kind,identity', [('create', 'pod'), ('resume', 'pod'), ('stop', 'pod'),
    ('pod', '../unsafe'), ('pod', 'a" mutation'), ('gpu', '\nmutation')])
def test_worker_rejects_nonread_operations(kind, identity):
    with pytest.raises(ValueError):
        obs.http_worker({'kind': kind, 'identity': identity, 'key_path': '/must/not/read'})


def test_collect_uses_frozen_hard_supervisor_and_shared_deadline(monkeypatch):
    p, g = receipts()
    times = []
    class Network:
        def supervise(self, command, payload, *, timeout, deadline):
            times.append((timeout, deadline, payload['kind']))
            result = deepcopy(p if payload['kind'] == 'pod' else g)
            now = datetime.now(timezone.utc)
            result['started_at'] = (now - timedelta(seconds=2)).isoformat()
            result['received_at'] = (now - timedelta(seconds=1)).isoformat()
            return result
    monkeypatch.setattr(obs, 'load_frozen', lambda _: Network())
    assert not obs.collect('synthetic-pod', '/never-read')['ready_for_live_use']
    assert times[0][0] == times[1][0] == 14
    assert times[0][1] == times[1][1]
    assert [t[2] for t in times] == ['pod', 'gpu']


def test_partial_provider_error_does_not_retry(monkeypatch):
    calls = []
    class Network:
        def supervise(self, *args, **kwargs):
            calls.append(1)
            return {'ok': False}
    monkeypatch.setattr(obs, 'load_frozen', lambda _: Network())
    with pytest.raises(ValueError):
        obs.collect('synthetic-pod', '/never-read')
    assert len(calls) == 1


def test_pinned_create_contract_sleep_cmd_and_no_template():
    transport = obs.load_frozen('provider_transport')
    body = transport.create_body({**transport.FIXED, 'name': 'qwen-provision-' + '0' * 32})
    assert body['dockerStartCmd'] == ['/bin/sleep', 'infinity']
    assert 'dockerEntrypoint' not in body
    assert body['templateId'] is None and body['env'] == {} and body['ports'] == []
    assert body['supportPublicIp'] is False


def test_source_manifest_hashes():
    path = STAGE / 'configs/source_manifest.json'
    assert path.exists()
    manifest = json.loads(path.read_text())
    for name, expected in manifest['files'].items():
        assert obs.sha((STAGE / name).read_bytes()) == expected
