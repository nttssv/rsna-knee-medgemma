"""CPU-only two-replica lifecycle and parity checks.

All reports, IDs, scores, GPU facts and worker receipts in this file are
SYNTHETIC. Real subprocesses test CUDA visibility, cancellation and reaping;
no test loads Torch, a model, CUDA, labels, or a provider client.
"""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = ROOT.parent / "kaggle_image_t4_v1"
sys.path.insert(0, str(V8_ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
import replica_runner  # noqa: E402
from inference_core import ContractError, LABELS, sha256_file  # noqa: E402


IDS = ["SYNTHETIC-c", "SYNTHETIC-a", "SYNTHETIC-b"]


def predictions(ids=IDS):
    bases = {"SYNTHETIC-a": 0.1, "SYNTHETIC-b": 0.2, "SYNTHETIC-c": 0.3}
    return pd.DataFrame([
        {"StudyInstanceUID": uid, **{label: bases[uid] + i / 100 for i, label in enumerate(LABELS)}}
        for uid in ids
    ], columns=["StudyInstanceUID", *LABELS])


SYNTHETIC_WORKER = r'''
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

request_path, worker_id, worker_dir, events_path, scenario = sys.argv[1:]
worker_id = int(worker_id)
worker_dir = Path(worker_dir)
request = json.loads(Path(request_path).read_text())
with (Path(request['data_dir']) / 'test.csv').open() as stream:
    ids = sorted(row['StudyInstanceUID'] for row in csv.DictReader(stream))[worker_id::2]
event = {'kind': 'SYNTHETIC', 'worker_id': worker_id, 'pid': os.getpid(),
         'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'study_ids': ids}
with open(events_path, 'a') as stream:
    stream.write(json.dumps(event) + '\n')
    stream.flush()

if scenario == 'timeout':
    time.sleep(60)
    raise SystemExit(99)
if scenario == 'failed':
    # Both real worker processes must start before this one fails so sibling
    # cancellation is exercised rather than relying on a launch race.
    until = time.monotonic() + 3
    while len(Path(events_path).read_text().splitlines()) < 2 and time.monotonic() < until:
        time.sleep(0.01)
    (worker_dir / 'worker_result.json').write_text(json.dumps({
        'kind': 'SYNTHETIC', 'worker_id': worker_id, 'pid': os.getpid(),
        'status': 'failed', 'phase': 'inference', 'error': 'SYNTHETIC forward failure'}))
    raise SystemExit(1)
if scenario == 'missing_receipt':
    raise SystemExit(0)
if scenario == 'malformed_receipt':
    (worker_dir / 'worker_result.json').write_text('not json')
    raise SystemExit(0)

labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
bases = {'SYNTHETIC-a': 0.1, 'SYNTHETIC-b': 0.2, 'SYNTHETIC-c': 0.3}
rows = [{'StudyInstanceUID': uid, **{name: bases[uid] + j / 100 for j, name in enumerate(labels)}}
        for uid in ids]
if scenario == 'duplicate_rows':
    rows.append(rows[0])
elif scenario == 'missing_row':
    rows = rows[:-1]
elif scenario == 'foreign_id':
    rows[0]['StudyInstanceUID'] = 'SYNTHETIC-foreign'
elif scenario == 'nan_score':
    rows[0]['ACL'] = float('nan')
elif scenario == 'bad_parity':
    rows[0]['ACL'] += 0.01
elif scenario == 'bounds_score':
    rows[0]['ACL'] = 1.01

output = worker_dir / 'example' / 'submission.csv'
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['StudyInstanceUID', *labels])
    writer.writeheader()
    writer.writerows(rows)
receipt = {
    'kind': 'SYNTHETIC', 'worker_id': worker_id, 'pid': os.getpid(),
    'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_count': 1,
    'placement': 'single_gpu', 'status': 'complete', 'phase': 'inference',
    'study_ids': ids, 'submission_path': str(output),
    'submission_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'example_status': {'load_seconds': 0.01, 'total_seconds': 0.05,
        'observed_seconds_per_study': 0.05 / len(ids),
        'gpu_memory': [{'device': 0, 'peak_allocated_gib': 4.8, 'peak_reserved_gib': 5.7}]},
}
if scenario == 'wrong_pid':
    receipt['pid'] += 100000
elif scenario == 'wrong_worker':
    receipt['worker_id'] = 1 - worker_id
elif scenario == 'wrong_visibility':
    receipt['visible_devices'] = '0,1'
elif scenario == 'wrong_count':
    receipt['gpu_count'] = 2
elif scenario == 'wrong_placement':
    receipt['placement'] = 'two_gpu'
elif scenario == 'wrong_phase':
    receipt['phase'] = 'prepare'
elif scenario == 'wrong_status':
    receipt['status'] = 'running'
elif scenario == 'wrong_shard':
    receipt['study_ids'] = ['SYNTHETIC-wrong']
elif scenario == 'wrong_digest':
    receipt['submission_sha256'] = '0' * 64
elif scenario == 'wrong_path':
    receipt['submission_path'] = request['reference_csv']
    receipt['submission_sha256'] = request['reference_sha256']
elif scenario == 'tampered_csv':
    with output.open('a') as stream:
        stream.write('\n')
(worker_dir / 'worker_result.json').write_text(json.dumps(receipt))
raise SystemExit(1 if scenario == 'wrong_exit' else 0)
'''


def events_at(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


class FakeCuda:
    def __init__(self, count=2, free=(14.46, 14.46)):
        self.count = count
        self.free = free
        self.synchronized = []

    def device_count(self):
        return self.count

    def synchronize(self, index):
        self.synchronized.append(index)

    def mem_get_info(self, index):
        return int(self.free[index] * 2**30), int(14.56 * 2**30)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    test = pd.DataFrame({'StudyInstanceUID': IDS})
    test.to_csv(data / 'test.csv', index=False)
    predictions().to_csv(data / 'sample_submission.csv', index=False)
    pd.DataFrame([{
        'StudyInstanceUID': uid, 'SeriesInstanceUID': f'SYNTHETIC-series-{i}',
        'Fluid_Sensitive': 1, 'Fat_Suppression': 1, 'Anatomical_Plane': 'SAG',
    } for i, uid in enumerate(IDS)]).to_csv(data / 'test_series.csv', index=False)
    reference = tmp_path / 'SYNTHETIC_reference.csv'
    predictions().to_csv(reference, index=False)
    request = tmp_path / 'request.json'
    output = tmp_path / 'output'
    request.write_text(json.dumps({
        'config_path': str(V8_ROOT / 'configs' / 'inference.json'),
        'data_dir': str(data), 'base_dir': str(tmp_path / 'base'),
        'adapter_dir': str(tmp_path / 'adapter'), 'output_root': str(output),
        'reference_csv': str(reference), 'reference_sha256': sha256_file(reference),
        'asset_verification_seconds': 0.01, 'dependency_install_seconds': 0.01,
    }))
    worker = tmp_path / 'synthetic_worker.py'
    worker.write_text(SYNTHETIC_WORKER)
    events = tmp_path / 'events.jsonl'
    launched = []
    monkeypatch.setattr(replica_runner, 'verify_v8_sources', lambda *_args, **_kwargs: None)

    def install(scenarios=None):
        scenarios = scenarios or {0: 'complete', 1: 'complete'}

        def command(actual_request, worker_id, worker_dir):
            assert Path(actual_request) == request
            assert Path(worker_dir) == output / f'worker_{worker_id}'
            launched.append(worker_id)
            assert launched.count(worker_id) == 1, 'No retry or replacement replica is permitted'
            return [sys.executable, str(worker), str(actual_request), str(worker_id),
                    str(worker_dir), str(events), scenarios[worker_id]]

        monkeypatch.setattr(replica_runner, '_worker_command', command)

    install()
    return request, output, events, launched, install


def test_shards_are_sorted_alternating_and_independent_of_original_order():
    test = pd.DataFrame({'StudyInstanceUID': IDS})
    before = test.copy(deep=True)
    expected = [['SYNTHETIC-a', 'SYNTHETIC-c'], ['SYNTHETIC-b']]
    assert replica_runner.deterministic_shards(test) == expected
    assert replica_runner.deterministic_shards(test.iloc[::-1]) == expected
    pd.testing.assert_frame_equal(test, before)


@pytest.fixture
def frozen_bundle(tmp_path):
    bundle = tmp_path / 'byte_frozen_v8'
    bundle.mkdir()
    for name in replica_runner.V8_HASHES:
        source = V8_ROOT / ('configs/inference.json' if name == 'inference_config.json' else f'scripts/{name}')
        shutil.copyfile(source, bundle / name)
    return bundle


def test_real_frozen_v8_sources_pass_checksum_gate(frozen_bundle):
    replica_runner.verify_v8_sources(frozen_bundle / 'inference_config.json')


@pytest.mark.parametrize('name', [
    'inference_core.py', 'submission_runtime.py', 'numerical_runtime.py', 'inference_config.json',
])
def test_frozen_source_or_config_drift_is_blocked(frozen_bundle, name):
    target = frozen_bundle / name
    target.write_bytes(target.read_bytes() + b'\nSYNTHETIC source drift\n')
    with pytest.raises(ContractError, match='drift'):
        replica_runner.verify_v8_sources(frozen_bundle / 'inference_config.json')


@pytest.mark.parametrize('ids', [[], ['SYNTHETIC-a'], ['x', 'x'], ['x', None]])
def test_invalid_or_insufficient_ids_fail_sharding(ids):
    with pytest.raises(ContractError):
        replica_runner.deterministic_shards(pd.DataFrame({'StudyInstanceUID': ids}))


def test_two_real_processes_have_single_gpu_visibility_and_merge_original_order(harness, monkeypatch):
    request, output, events, launched, _install = harness
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    result = replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda())
    assert launched == [0, 1]
    child_events = sorted(events_at(events), key=lambda row: row['worker_id'])
    assert [row['visible_devices'] for row in child_events] == ['0', '1']
    assert [row['study_ids'] for row in child_events] == [['SYNTHETIC-a', 'SYNTHETIC-c'], ['SYNTHETIC-b']]
    assert len({row['pid'] for row in child_events}) == 2
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    for row in child_events:
        assert_reaped(row['pid'])
    final = pd.read_csv(result['final_submission_path'])
    assert Path(result['final_submission_path']) == output / 'submission.csv'
    assert final.StudyInstanceUID.tolist() == IDS
    pd.testing.assert_frame_equal(final, pd.read_csv(json.loads(request.read_text())['reference_csv']))
    assert result['parity']['passed'] is True
    assert result['parity']['compared_scores'] == 36
    assert result['parity']['max_absolute_difference'] == 0


@pytest.mark.parametrize('scenario', [
    'missing_receipt', 'malformed_receipt', 'wrong_pid', 'wrong_worker',
    'wrong_visibility', 'wrong_count', 'wrong_placement', 'wrong_phase',
    'wrong_status', 'wrong_shard', 'wrong_digest', 'wrong_path', 'tampered_csv',
    'wrong_exit', 'duplicate_rows', 'missing_row', 'foreign_id', 'nan_score',
    'bounds_score', 'bad_parity',
])
def test_invalid_worker_or_scores_never_publish_final_or_retry(harness, scenario):
    request, output, events, launched, install = harness
    install({0: scenario, 1: 'complete'})
    with pytest.raises(ContractError):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda())
    assert not (output / 'submission.csv').exists()
    assert len(launched) == len(set(launched)) == 2
    for row in events_at(events):
        assert_reaped(row['pid'])


def test_worker_failure_kills_and_reaps_sleeping_sibling(harness):
    request, output, events, launched, install = harness
    install({0: 'failed', 1: 'timeout'})
    started = time.monotonic()
    with pytest.raises(ContractError):
        replica_runner.run_replicas(request, started + 15, cuda=FakeCuda())
    assert time.monotonic() - started < 8
    assert launched == [0, 1]
    assert len(events_at(events)) == 2
    for row in events_at(events):
        assert_reaped(row['pid'])
    assert not (output / 'submission.csv').exists()


def test_hard_deadline_kills_and_reaps_both_workers(harness):
    request, output, events, launched, install = harness
    install({0: 'timeout', 1: 'timeout'})
    started = time.monotonic()
    with pytest.raises(ContractError):
        replica_runner.run_replicas(request, started + 0.8, cuda=FakeCuda())
    assert time.monotonic() - started < 8
    assert launched == [0, 1]
    assert len(events_at(events)) == 2
    for row in events_at(events):
        assert_reaped(row['pid'])
    assert not (output / 'submission.csv').exists()


def test_expired_deadline_starts_no_worker(harness):
    request, output, events, launched, _install = harness
    with pytest.raises(ContractError):
        replica_runner.run_replicas(request, time.monotonic() - 1, cuda=FakeCuda())
    assert launched == []
    assert events_at(events) == []
    assert not (output / 'submission.csv').exists()


def test_ambiguous_parent_mapping_cannot_silently_select_other_devices(harness, monkeypatch):
    request, _output, events, launched, _install = harness
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    with pytest.raises(ContractError, match='mapping'):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda())
    assert launched == []
    assert events_at(events) == []
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'


@pytest.mark.parametrize('count', [0, 1, 3])
def test_exactly_two_visible_parent_devices_required(harness, count):
    request, output, events, launched, _install = harness
    with pytest.raises(ContractError):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda(count=count))
    assert launched == []
    assert events_at(events) == []


@pytest.mark.parametrize('free', [(11.9, 14.46), (14.46, 11.9)])
def test_each_replica_needs_its_own_memory_headroom(harness, free):
    request, output, events, launched, _install = harness
    with pytest.raises(ContractError):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda(free=free))
    assert launched == []
    assert events_at(events) == []


def test_existing_outputs_remain_untouched(harness):
    request, output, events, launched, _install = harness
    output.mkdir()
    historical = output / 'submission.csv'
    historical.write_bytes(b'SYNTHETIC historical artifact\n')
    old_sha = sha256_file(historical)
    with pytest.raises((FileExistsError, ContractError)):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda())
    assert sha256_file(historical) == old_sha
    assert launched == []
    assert events_at(events) == []


def test_real_worker_dispatch_uses_one_visible_gpu_and_frozen_single_placement(harness, monkeypatch):
    """Exercise the real worker dispatch with synthetic model preparation/inference."""
    request, output, _events, _launched, _install = harness
    worker_dir = output / 'worker_1'
    worker_dir.mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    fake_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(major=7, minor=5, total_memory=int(14.56 * 2**30)),
        get_device_name=lambda index: 'Tesla T4',
    )
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=fake_cuda))
    calls = []
    teacher = SimpleNamespace(device_map={'': '0'}, dtype_report={'kind': 'SYNTHETIC'})

    def prepare(config, series, data, base, adapter, diagnostic_uid, log, *, only_placement):
        assert only_placement == 'single_gpu'
        assert config == json.loads((V8_ROOT / 'configs/inference.json').read_text())
        calls.append(('prepare', diagnostic_uid, str(base), str(adapter)))
        return teacher, 0.01, {'kind': 'SYNTHETIC'}, []

    def infer(config, test, series, sample, data, base, adapter, out, actual_teacher,
              load_seconds, asset_seconds, dependency_seconds, diagnostic):
        assert actual_teacher is teacher
        assert test.StudyInstanceUID.tolist() == ['SYNTHETIC-b']
        calls.append(('infer', test.StudyInstanceUID.tolist()))
        out.mkdir()
        prediction = predictions(test.StudyInstanceUID.tolist())
        prediction.to_csv(out / 'submission.csv', index=False)
        return prediction, {'kind': 'SYNTHETIC', 'total_seconds': 0.1}

    monkeypatch.setitem(sys.modules, 'submission_runtime', SimpleNamespace(
        prepare_t4_teacher=prepare, run_inference=infer))
    assert replica_runner.worker(request, 1, worker_dir) == 0
    assert [row[0] for row in calls] == ['prepare', 'infer']
    receipt = json.loads((worker_dir / 'worker_result.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['gpu_count'] == 1
    assert receipt['placement'] == 'single_gpu'
    assert receipt['visible_devices'] == '1'
    assert receipt['study_ids'] == ['SYNTHETIC-b']
    assert receipt['submission_sha256'] == sha256_file(worker_dir / 'example/submission.csv')


@pytest.mark.parametrize('visibility', ['', '0', '0,1', '2'])
def test_real_worker_rejects_wrong_visibility_before_model_import(harness, monkeypatch, visibility):
    request, output, _events, _launched, _install = harness
    worker_dir = output / 'worker_1'
    worker_dir.mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visibility)
    # A correct rejection happens before reading CUDA/model state.
    class ForbiddenTorch:
        def __getattr__(self, _name):
            raise AssertionError('Torch must not be touched for wrong worker visibility')
    monkeypatch.setitem(sys.modules, 'torch', ForbiddenTorch())
    assert replica_runner.worker(request, 1, worker_dir) == 1
    result = json.loads((worker_dir / 'worker_result.json').read_text())
    assert result['error_type'] == 'ContractError'
    assert result['status'] == 'failed'
    assert not (worker_dir / 'example/submission.csv').exists()


@pytest.fixture
def parity_input(tmp_path):
    reference = tmp_path / 'SYNTHETIC_reference.csv'
    predictions().to_csv(reference, index=False)
    return pd.read_csv(reference), pd.DataFrame({'StudyInstanceUID': IDS}), reference, sha256_file(reference)


def test_reference_parity_checks_every_score(parity_input):
    prediction, test, path, digest = parity_input
    result = replica_runner.compare_reference(prediction, test, path, digest)
    assert result['passed'] is True
    assert result['compared_scores'] == 36
    assert result['max_absolute_difference'] == 0


def test_small_machine_level_difference_is_recorded_not_hidden(parity_input):
    prediction, test, path, digest = parity_input
    prediction.loc[2, LABELS[-1]] += 5e-7
    result = replica_runner.compare_reference(prediction, test, path, digest)
    assert result['passed'] is True
    assert result['max_absolute_difference'] == pytest.approx(5e-7)


@pytest.mark.parametrize('row,column', [(0, LABELS[0]), (2, LABELS[-1])])
def test_material_difference_even_last_condition_rejects_parity(parity_input, row, column):
    prediction, test, path, digest = parity_input
    prediction.loc[row, column] += 1e-4
    with pytest.raises(ContractError):
        replica_runner.compare_reference(prediction, test, path, digest)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.01, 1.01])
def test_invalid_scores_cannot_pass_parity(parity_input, bad):
    prediction, test, path, digest = parity_input
    prediction.loc[0, LABELS[0]] = bad
    with pytest.raises(ContractError):
        replica_runner.compare_reference(prediction, test, path, digest)


def test_wrong_reference_hash_rejected(parity_input):
    prediction, test, path, _digest = parity_input
    with pytest.raises(ContractError):
        replica_runner.compare_reference(prediction, test, path, '0' * 64)


def test_reference_file_tampering_rejected(parity_input):
    prediction, test, path, digest = parity_input
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ContractError):
        replica_runner.compare_reference(prediction, test, path, digest)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'foreign', 'column_order'])
def test_parity_requires_complete_identical_schema_and_study_set(parity_input, mutation):
    prediction, test, path, digest = parity_input
    if mutation == 'missing':
        prediction = prediction.iloc[:-1]
    elif mutation == 'duplicate':
        prediction.loc[2, 'StudyInstanceUID'] = prediction.loc[0, 'StudyInstanceUID']
    elif mutation == 'foreign':
        prediction.loc[2, 'StudyInstanceUID'] = 'SYNTHETIC-foreign'
    elif mutation == 'column_order':
        prediction = prediction[['StudyInstanceUID', *LABELS[::-1]]]
    with pytest.raises(ContractError):
        replica_runner.compare_reference(prediction, test, path, digest)


@pytest.mark.parametrize('atol', [1e-3, -1.0, float('nan'), float('inf')])
def test_parity_tolerance_cannot_be_relaxed_or_nonfinite(parity_input, atol):
    prediction, test, path, digest = parity_input
    with pytest.raises(ContractError):
        replica_runner.compare_reference(prediction, test, path, digest, atol=atol)


def test_source_gate_failure_prevents_launch(harness, monkeypatch):
    request, _output, events, launched, _install = harness

    def drift(_path):
        raise ContractError('SYNTHETIC frozen source drift')

    monkeypatch.setattr(replica_runner, 'verify_v8_sources', drift)
    with pytest.raises(ContractError, match='source drift'):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda())
    assert launched == []
    assert events_at(events) == []


def test_reference_drift_prevents_any_worker_launch(harness):
    request, _output, events, launched, _install = harness
    reference = Path(json.loads(request.read_text())['reference_csv'])
    reference.write_text(reference.read_text() + '\nSYNTHETIC drift\n')
    with pytest.raises(ContractError, match='reference checksum'):
        replica_runner.run_replicas(request, time.monotonic() + 15, cuda=FakeCuda())
    assert launched == []
    assert events_at(events) == []


@pytest.fixture
def builder(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('synthetic_replica_builder', ROOT / 'scripts/build_notebook.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = tmp_path / 'SYNTHETIC_v8_reference.csv'
    predictions().to_csv(reference, index=False)
    # Portable synthetic fixture only. Production build stays bound to the real
    # downloaded Version 8 CSV digest, never changed by this test injection.
    monkeypatch.setattr(module, 'REFERENCE_SHA', sha256_file(reference))
    return module, reference, tmp_path / 'synthetic_notebook.ipynb'


def embedded_sources(notebook):
    embedded = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        compile(source, '<SYNTHETIC notebook cell>', 'exec')
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'write_text' and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == 'Path'
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                path = ast.literal_eval(node.func.value.args[0])
                embedded[Path(path).name] = ast.literal_eval(node.args[0])
    return embedded


def test_built_notebook_compiles_and_embeds_frozen_v8_bytes(builder):
    module, reference, output = builder
    result = module.build(output, reference)
    notebook = json.loads(output.read_text())
    assert result['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    embedded = embedded_sources(notebook)
    for name, digest in replica_runner.V8_HASHES.items():
        assert hashlib.sha256(embedded[name].encode()).hexdigest() == digest
    assert embedded['replica_runner.py'] == (ROOT / 'scripts/replica_runner.py').read_text()
    assert 'isolated_runner.py' not in embedded
    final = notebook['cells'][-1]['source']
    assert 'run_replicas(request_path,SESSION_DEADLINE_MONOTONIC)' in final
    assert 'run_isolated' not in final
    assert reference.read_text() in ast.literal_eval(next(
        node.args[0] for node in ast.walk(ast.parse(final))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'REFERENCE'
        and node.func.attr == 'write_text'
    ))
    assert notebook['metadata']['kaggle']['isInternetEnabled'] is False


def test_notebook_keeps_frozen_offline_install_and_two_gpu_preflight(builder):
    module, reference, output = builder
    module.build(output, reference)
    notebook = json.loads(output.read_text())
    cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    setup = next(cell for cell in cells if "os.environ['HF_HUB_OFFLINE']" in cell)
    assert "os.environ['HF_HUB_OFFLINE']='1'" in setup
    assert "os.environ['TRANSFORMERS_OFFLINE']='1'" in setup
    assert '115*60' in setup
    install = next(cell for cell in cells if 'subprocess.check_call' in cell)
    assert "'--no-index'" in install
    assert "'--no-deps'" in install
    preflight = next(cell for cell in cells if 'Exactly two T4 replicas required' in cell)
    assert 'assert len(devices)==2' in preflight
    assert 'free_gib[:1]' not in preflight


def test_notebook_builder_rejects_reference_drift(builder):
    module, reference, output = builder
    reference.write_text(reference.read_text() + '\nSYNTHETIC drift\n')
    with pytest.raises(ValueError, match='exact downloaded'):
        module.build(output, reference)
    assert not output.exists()


def test_notebook_builder_refuses_overwrite(builder):
    module, reference, output = builder
    output.write_bytes(b'SYNTHETIC historical notebook\n')
    before = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        module.build(output, reference)
    assert output.read_bytes() == before
