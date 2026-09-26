"""Synthetic CPU contracts for the arbitrary-size submission-only runner.

Worker subprocesses below use only the Python standard library. No competition
IDs, reference predictions, model weights, network, or GPU are used by tests.
"""
from __future__ import annotations

import importlib.util
from contextlib import nullcontext
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
V8 = ROOT.parent / 'kaggle_image_t4_v1'
V10 = ROOT.parent / 'kaggle_image_t4_vision_cache_v1'
sys.path.insert(0, str(V8 / 'scripts'))
from inference_core import ContractError, LABELS, sha256_file  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module('submission_contract_runner', ROOT / 'scripts/submission_runner.py')


def synthetic_ids(count):
    # Deliberately differs from the deterministic worker sorting order.
    return [f'SYNTHETIC-{index:03d}' for index in range(count - 1, -1, -1)]


def predictions(ids):
    return pd.DataFrame([
        {'StudyInstanceUID': uid,
         **{label: 0.1 + int(uid.rsplit('-', 1)[1]) / 100 + i / 1000
            for i, label in enumerate(LABELS)}}
        for uid in ids
    ], columns=['StudyInstanceUID', *LABELS]).astype({label: float for label in LABELS})


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
assert 'reference_csv' not in request
assert 'reference_sha256' not in request
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
    until = time.monotonic() + 3
    while len(Path(events_path).read_text().splitlines()) < 2 and time.monotonic() < until:
        time.sleep(0.01)
    (worker_dir / 'worker_result.json').write_text(json.dumps({
        'status': 'failed', 'phase': 'inference', 'error': 'SYNTHETIC worker failure'}))
    raise SystemExit(1)
if scenario == 'missing_receipt':
    raise SystemExit(0)
if scenario == 'malformed_receipt':
    (worker_dir / 'worker_result.json').write_text('not json')
    raise SystemExit(0)

labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
columns = ['StudyInstanceUID', *labels]
rows = [{'StudyInstanceUID': uid,
         **{label: 0.1 + int(uid.rsplit('-', 1)[1]) / 100 + j / 1000
            for j, label in enumerate(labels)}} for uid in ids]
if scenario == 'duplicate_rows':
    rows.append(dict(rows[0]))
elif scenario == 'missing_row':
    rows = rows[:-1]
elif scenario == 'foreign_id':
    rows[0]['StudyInstanceUID'] = 'SYNTHETIC-foreign'
elif scenario == 'nan_score':
    rows[0]['ACL'] = float('nan')
elif scenario == 'inf_score':
    rows[0]['ACL'] = float('inf')
elif scenario == 'bounds_score':
    rows[0]['ACL'] = 1.01
elif scenario == 'wrong_columns':
    columns = ['StudyInstanceUID', *labels[::-1]]
elif scenario == 'extra_column':
    columns.append('SYNTHETIC-extra')
    for row in rows:
        row['SYNTHETIC-extra'] = 0

output = worker_dir / 'inference' / 'submission.csv'
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
timings = [{'StudyInstanceUID': uid, 'study_seconds': 31.25} for uid in ids]
if scenario == 'timing_nan':
    timings[0]['study_seconds'] = float('nan')
elif scenario == 'timing_inf':
    timings[0]['study_seconds'] = float('inf')
elif scenario == 'timing_zero':
    timings[0]['study_seconds'] = 0
elif scenario == 'timing_negative':
    timings[0]['study_seconds'] = -1
elif scenario == 'timing_missing_row':
    timings = timings[:-1]
elif scenario == 'timing_wrong_ids':
    timings[0]['StudyInstanceUID'] = 'SYNTHETIC-foreign'
elif scenario == 'timing_wrong_order':
    timings = timings[::-1]
if scenario != 'timing_missing_file':
    with (output.parent / 'study_timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['StudyInstanceUID', 'study_seconds'])
        writer.writeheader()
        writer.writerows(timings)
cache = {
    'policy': 'same_study_exact_pixel_vision_cache_v1', 'misses': len(ids),
    'hits': 11 * len(ids), 'underlying_vision_calls': len(ids),
    'cache_released': True, 'active_study': None, 'decoder_forwards_per_study': 12,
    'projector_unchanged': True, 'arithmetic_unchanged': True,
    'studies': [{'study_id': uid, 'misses': 1, 'hits': 11} for uid in ids],
}
receipt = {
    'kind': 'SYNTHETIC', 'worker_id': worker_id, 'pid': os.getpid(),
    'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_count': 1,
    'placement': 'single_gpu', 'status': 'complete', 'phase': 'inference',
    'study_ids': ids, 'submission_path': str(output), 'model_loaded': bool(ids),
    'submission_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'score_record_count': 12 * len(ids), 'vision_cache': cache,
    'inference_status': {'status': 'complete', 'planned_studies': len(ids),
        'completed_studies': len(ids), 'load_seconds': 0.01 if ids else 0,
        'total_seconds': 0.05, 'observed_seconds_per_study': 0.05 / len(ids) if ids else None,
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
    receipt['submission_path'] = str(Path(request['data_dir']) / 'sample_submission.csv')
elif scenario == 'tampered_csv':
    with output.open('a') as stream:
        stream.write('\n')
elif scenario == 'cache_missing':
    del receipt['vision_cache']
elif scenario == 'cache_misses':
    cache['misses'] += 1
elif scenario == 'cache_hits':
    cache['hits'] -= 1
elif scenario == 'cache_calls':
    cache['underlying_vision_calls'] += 1
elif scenario == 'cache_unreleased':
    cache['cache_released'] = False
elif scenario == 'cache_active':
    cache['active_study'] = ids[0]
elif scenario == 'cache_decoder_count':
    cache['decoder_forwards_per_study'] = 11
elif scenario == 'cache_study_hits':
    cache['studies'][0]['hits'] = 10
elif scenario == 'cache_study_id':
    cache['studies'][0]['study_id'] = 'SYNTHETIC-foreign'
elif scenario == 'cache_study_missing':
    cache['studies'] = cache['studies'][:-1]
elif scenario == 'score_record_count':
    receipt['score_record_count'] -= 1
(worker_dir / 'worker_result.json').write_text(json.dumps(receipt))
raise SystemExit(1 if scenario == 'wrong_exit' else 0)
'''


class FakeCuda:
    def __init__(self, count=2, free=(14.46, 14.46)):
        self.count, self.free = count, free

    def device_count(self):
        return self.count

    def synchronize(self, index):
        assert 0 <= index < self.count

    def mem_get_info(self, index):
        return int(self.free[index] * 2**30), int(14.56 * 2**30)


def events_at(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


@pytest.fixture
def harness_factory(tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)

    def make(count=3):
        bundle = tmp_path / 'frozen_sources'
        bundle.mkdir()
        for name in runner.V8_HASHES:
            source = V8 / ('configs/inference.json' if name == 'inference_config.json' else f'scripts/{name}')
            shutil.copyfile(source, bundle / name)
        shutil.copyfile(V10 / 'scripts/vision_cache.py', bundle / 'vision_cache.py')
        data = tmp_path / 'synthetic_data'
        data.mkdir()
        ids = synthetic_ids(count)
        pd.DataFrame({'StudyInstanceUID': ids}).to_csv(data / 'test.csv', index=False)
        predictions(ids).to_csv(data / 'sample_submission.csv', index=False)
        pd.DataFrame([{'StudyInstanceUID': uid, 'SeriesInstanceUID': f'SYNTHETIC-series-{i}',
            'Fluid_Sensitive': 1, 'Fat_Suppression': 1, 'Anatomical_Plane': 'SAG'}
            for i, uid in enumerate(ids)]).to_csv(data / 'test_series.csv', index=False)
        request, output, events = tmp_path / 'request.json', tmp_path / 'output', tmp_path / 'events.jsonl'
        request.write_text(json.dumps({
            'config_path': str(bundle / 'inference_config.json'), 'data_dir': str(data),
            'base_dir': str(tmp_path / 'base'), 'adapter_dir': str(tmp_path / 'adapter'),
            'output_root': str(output), 'vision_cache_sha256': sha256_file(bundle / 'vision_cache.py'),
            'asset_verification_seconds': 0.01, 'dependency_install_seconds': 0.01,
        }))
        worker = tmp_path / 'synthetic_worker.py'
        worker.write_text(SYNTHETIC_WORKER)
        launched = []

        def install(scenarios=None):
            scenarios = scenarios or {0: 'complete', 1: 'complete'}

            def command(actual_request, worker_id, worker_dir):
                assert Path(actual_request) == request
                assert Path(worker_dir) == output / f'worker_{worker_id}'
                launched.append(worker_id)
                assert launched.count(worker_id) == 1, 'Retries are forbidden'
                return [sys.executable, str(worker), str(request), str(worker_id),
                        str(worker_dir), str(events), scenarios[worker_id]]

            monkeypatch.setattr(runner, '_worker_command', command)

        install()
        return SimpleNamespace(request=request, output=output, events=events, launched=launched,
                               install=install, bundle=bundle, ids=ids, data=data)

    return make


@pytest.fixture
def harness(harness_factory):
    return harness_factory()


def run(harness, deadline=None, cuda=None):
    return runner.run_replicas(harness.request,
        time.monotonic() + 15 if deadline is None else deadline,
        cuda=FakeCuda() if cuda is None else cuda)


def assert_failed_cleanly(harness):
    assert not (harness.output / 'submission.csv').exists()
    assert len(harness.launched) == len(set(harness.launched))
    for row in events_at(harness.events):
        assert_reaped(row['pid'])


@pytest.mark.parametrize('count', [3, 7, 1])
def test_arbitrary_count_balanced_deterministic_shards_and_exact_test_order_merge(count):
    ids = synthetic_ids(count)
    test = pd.DataFrame({'StudyInstanceUID': ids})
    before = test.copy(deep=True)
    expected = [sorted(ids)[0::2], sorted(ids)[1::2]]
    assert runner.deterministic_shards(test) == expected
    assert runner.deterministic_shards(test.iloc[::-1]) == expected
    assert abs(len(expected[0]) - len(expected[1])) <= 1
    actual = runner.merge_predictions([predictions(shard) for shard in expected], test)
    pd.testing.assert_frame_equal(actual, predictions(ids))
    pd.testing.assert_frame_equal(test, before)


@pytest.mark.parametrize('ids', [[], ['SYNTHETIC-a', 'SYNTHETIC-a'], ['SYNTHETIC-a', None], [''], ['  ']])
def test_empty_null_blank_or_duplicate_ids_fail_sharding(ids):
    with pytest.raises(ContractError):
        runner.deterministic_shards(pd.DataFrame({'StudyInstanceUID': ids}))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'foreign', 'nan', 'inf', 'columns', 'bounds'])
def test_merge_rejects_incomplete_invalid_or_misshapen_scores(mutation):
    test = pd.DataFrame({'StudyInstanceUID': synthetic_ids(7)})
    pieces = [predictions(shard) for shard in runner.deterministic_shards(test)]
    if mutation == 'missing':
        pieces[0] = pieces[0].iloc[:-1]
    elif mutation == 'duplicate':
        pieces[0] = pd.concat([pieces[0], pieces[0].iloc[:1]], ignore_index=True)
    elif mutation == 'foreign':
        pieces[0].loc[0, 'StudyInstanceUID'] = 'SYNTHETIC-foreign'
    elif mutation in ('nan', 'inf', 'bounds'):
        pieces[0].loc[0, LABELS[0]] = {'nan': float('nan'), 'inf': float('inf'), 'bounds': -0.1}[mutation]
    else:
        pieces = [piece[['StudyInstanceUID', *LABELS[::-1]]] for piece in pieces]
    with pytest.raises(ContractError):
        runner.merge_predictions(pieces, test)


@pytest.mark.parametrize('count', [3, 7, 1])
def test_real_cpu_processes_complete_without_reference_for_any_count(harness_factory, monkeypatch, count):
    harness = harness_factory(count)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    result = run(harness)
    assert result['status'] == 'complete'
    assert 'parity' not in result
    assert 'under40_target_met' not in result
    assert harness.launched == [0, 1]
    rows = sorted(events_at(harness.events), key=lambda row: row['worker_id'])
    assert [row['visible_devices'] for row in rows] == ['0', '1']
    assert [row['study_ids'] for row in rows] == runner.deterministic_shards(pd.DataFrame({'StudyInstanceUID': harness.ids}))
    assert len({row['pid'] for row in rows}) == 2
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    for row in rows:
        assert_reaped(row['pid'])
    actual = pd.read_csv(result['final_submission_path'])
    assert Path(result['final_submission_path']) == harness.output / 'submission.csv'
    assert result['submission_sha256'] == sha256_file(harness.output / 'submission.csv')
    assert actual.StudyInstanceUID.tolist() == harness.ids
    pd.testing.assert_frame_equal(actual, predictions(harness.ids))
    assert 'reference' not in harness.request.read_text().lower()


@pytest.mark.parametrize('scenario', [
    'missing_receipt', 'malformed_receipt', 'wrong_pid', 'wrong_worker',
    'wrong_visibility', 'wrong_count', 'wrong_placement', 'wrong_phase',
    'wrong_status', 'wrong_shard', 'wrong_digest', 'wrong_path', 'tampered_csv',
    'wrong_exit', 'duplicate_rows', 'missing_row', 'foreign_id', 'nan_score',
    'inf_score', 'bounds_score', 'wrong_columns', 'extra_column',
    'cache_missing', 'cache_misses', 'cache_hits', 'cache_calls', 'cache_unreleased',
    'cache_active', 'cache_decoder_count', 'cache_study_hits', 'cache_study_id',
    'cache_study_missing', 'score_record_count',
    'timing_missing_file', 'timing_missing_row', 'timing_wrong_ids', 'timing_wrong_order',
    'timing_nan', 'timing_inf', 'timing_zero', 'timing_negative',
])
def test_invalid_child_or_csv_never_publishes_final_or_retries(harness, scenario):
    harness.install({0: scenario, 1: 'complete'})
    with pytest.raises((ContractError, OSError, ValueError)):
        run(harness)
    assert harness.launched == [0, 1]
    assert_failed_cleanly(harness)


def test_worker_fault_kills_and_reaps_sleeping_sibling(harness):
    harness.install({0: 'failed', 1: 'timeout'})
    started = time.monotonic()
    with pytest.raises(ContractError):
        run(harness, deadline=started + 15)
    assert time.monotonic() - started < 8
    assert harness.launched == [0, 1]
    assert len(events_at(harness.events)) == 2
    assert_failed_cleanly(harness)


def test_live_deadline_kills_and_reaps_all_children(harness):
    harness.install({0: 'timeout', 1: 'timeout'})
    started = time.monotonic()
    with pytest.raises(ContractError):
        run(harness, deadline=started + 0.8)
    assert time.monotonic() - started < 8
    assert harness.launched == [0, 1]
    assert len(events_at(harness.events)) == 2
    assert_failed_cleanly(harness)


@pytest.mark.parametrize('deadline', [float('nan'), float('inf'), float('-inf'), 0.0])
def test_invalid_or_expired_deadline_starts_no_worker(harness, deadline):
    with pytest.raises(ContractError):
        run(harness, deadline=deadline)
    assert harness.launched == []
    assert events_at(harness.events) == []
    assert_failed_cleanly(harness)


def test_deadline_after_merge_prevents_publication(harness, monkeypatch):
    deadline = time.monotonic() + 15
    original = runner.merge_predictions

    def merge_then_expire(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: deadline + 1, sleep=time.sleep))
        return result

    monkeypatch.setattr(runner, 'merge_predictions', merge_then_expire)
    with pytest.raises(ContractError):
        run(harness, deadline=deadline)
    assert harness.launched == [0, 1]
    assert_failed_cleanly(harness)


def test_existing_run_is_never_overwritten(harness):
    harness.output.mkdir()
    final = harness.output / 'submission.csv'
    final.write_bytes(b'SYNTHETIC historical artifact\n')
    before = final.read_bytes()
    with pytest.raises((FileExistsError, ContractError)):
        run(harness)
    assert final.read_bytes() == before
    assert harness.launched == []
    assert events_at(harness.events) == []


def test_final_publication_cannot_overwrite_a_racing_existing_file(harness, monkeypatch):
    original = runner.merge_predictions
    final = harness.output / 'submission.csv'

    def merge_then_insert_file(*args, **kwargs):
        result = original(*args, **kwargs)
        final.write_bytes(b'SYNTHETIC concurrently published file\n')
        return result

    monkeypatch.setattr(runner, 'merge_predictions', merge_then_insert_file)
    with pytest.raises((FileExistsError, ContractError)):
        run(harness)
    assert final.read_bytes() == b'SYNTHETIC concurrently published file\n'
    for row in events_at(harness.events):
        assert_reaped(row['pid'])


@pytest.mark.parametrize('count', [0, 1, 3])
def test_parent_requires_two_visible_devices(harness, count):
    with pytest.raises(ContractError):
        run(harness, cuda=FakeCuda(count=count))
    assert harness.launched == []
    assert_failed_cleanly(harness)


@pytest.mark.parametrize('free', [(11.9, 14.46), (14.46, 11.9)])
def test_parent_checks_memory_on_each_gpu(harness, free):
    with pytest.raises(ContractError):
        run(harness, cuda=FakeCuda(free=free))
    assert harness.launched == []
    assert_failed_cleanly(harness)


def test_ambiguous_parent_device_mapping_is_rejected(harness, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    with pytest.raises(ContractError):
        run(harness)
    assert harness.launched == []
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'


@pytest.mark.parametrize('name', ['vision_cache.py', *runner.V8_HASHES])
def test_frozen_source_or_cache_drift_prevents_launch(harness, name):
    path = harness.bundle / name
    path.write_bytes(path.read_bytes() + b'\nSYNTHETIC source drift\n')
    with pytest.raises(ContractError):
        run(harness)
    assert harness.launched == []
    assert events_at(harness.events) == []


def test_request_cannot_rebind_cache_checksum_to_different_code(harness):
    path = harness.bundle / 'vision_cache.py'
    path.write_bytes(path.read_bytes() + b'\nSYNTHETIC source drift\n')
    request = json.loads(harness.request.read_text())
    request['vision_cache_sha256'] = sha256_file(path)
    harness.request.write_text(json.dumps(request))
    with pytest.raises(ContractError):
        run(harness)
    assert harness.launched == []


@pytest.mark.parametrize('name', ['vision_cache.py', 'submission_runtime.py'])
def test_real_worker_checks_source_before_importing_torch(harness, monkeypatch, name):
    path = harness.bundle / name
    path.write_bytes(path.read_bytes() + b'\nSYNTHETIC source drift\n')
    directory = harness.output / 'worker_0'
    directory.mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    class ForbiddenTorch:
        def __getattr__(self, _name):
            raise AssertionError('Model/CUDA import occurred after source drift')

    monkeypatch.setitem(sys.modules, 'torch', ForbiddenTorch())
    assert runner.worker(harness.request, 0, directory) == 1
    receipt = json.loads((directory / 'worker_result.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['error_type'] == 'ContractError'


@pytest.mark.parametrize('visibility', ['', '0', '0,1', '2'])
def test_real_worker_rejects_wrong_visibility_before_model_import(harness, monkeypatch, visibility):
    directory = harness.output / 'worker_1'
    directory.mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visibility)

    class ForbiddenTorch:
        def __getattr__(self, _name):
            raise AssertionError('Torch must remain untouched for wrong visibility')

    monkeypatch.setitem(sys.modules, 'torch', ForbiddenTorch())
    assert runner.worker(harness.request, 1, directory) == 1
    receipt = json.loads((directory / 'worker_result.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['error_type'] == 'ContractError'


@pytest.fixture
def synthetic_model(monkeypatch):
    """Mock only model construction/forwards; run frozen predict/run_inference."""
    calls = []
    state = SimpleNamespace(count=1, name='Tesla T4', major=7, minor=5,
                            total_gib=14.56, free_gib=14.46, dtype_error=False,
                            score_error=None, load_error=False)
    fake_cuda = SimpleNamespace(
        device_count=lambda: state.count,
        get_device_properties=lambda _index: SimpleNamespace(
            major=state.major, minor=state.minor, total_memory=int(state.total_gib * 2**30)),
        get_device_name=lambda _index: state.name,
        synchronize=lambda _index: None,
        mem_get_info=lambda _index: (state.free_gib * 2**30, state.total_gib * 2**30),
        reset_peak_memory_stats=lambda _index: None,
        max_memory_allocated=lambda _index: int(4.8 * 2**30),
        max_memory_reserved=lambda _index: int(5.7 * 2**30),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, inference_mode=nullcontext,
                                 nn=SimpleNamespace(Module=object))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers.modeling_outputs',
                        SimpleNamespace(BaseModelOutputWithPooling=object))
    frozen = load_module('synthetic_frozen_submission_runtime', V8 / 'scripts/submission_runtime.py')
    monkeypatch.setitem(sys.modules, 'submission_runtime', frozen)
    cache_module = load_module('synthetic_frozen_cache_composition', V10 / 'scripts/vision_cache.py')

    def initialize(self, config, series, data, base, adapter, placement):
        calls.append(('construct', placement))
        assert placement == 'single_gpu'
        if state.load_error:
            raise RuntimeError('SYNTHETIC construction failure')
        self.torch = fake_torch
        self.config, self.series, self.placement = config, series, placement
        self.model = SimpleNamespace(eval=lambda: None)
        self.images = {}
        self.device_map = {'': '0'}
        self.dtype_report = {'kind': 'SYNTHETIC'}
        self.numerical_policy = {'kind': 'SYNTHETIC'}
        self.vision_runtime = {'vision_microbatch_images': 1}
        self.imaging = SimpleNamespace(issues=[])
        self.preprocessing_seconds = self.inference_seconds = 0.0

    def score_logits(self, uid, label):
        calls.append(('score', uid, label))
        result = {'native_no_logit': 0.1, 'native_yes_logit': 0.2,
                  'yes_probability': predictions([uid]).iloc[0][label]}
        if state.score_error is not None:
            key, value = state.score_error
            result[key] = value
        return result

    def forbidden_diagnostic(*_args, **_kwargs):
        raise AssertionError('Repeated diagnostic must never execute in submission mode')

    def dtype_check(model):
        calls.append(('dtype',))
        if state.dtype_error:
            raise ContractError('SYNTHETIC NF4 compute dtype mismatch')
        return {'torch.float16': 1}

    class SyntheticCache:
        def __init__(self):
            self.uid, self.hits, self.misses, self.studies = None, 0, 0, []
            self.study_hits = 0

        def begin_study(self, uid):
            if uid != self.uid:
                self.clear()
                self.uid, self.study_hits = uid, 0
                self.misses += 1
            else:
                self.hits += 1
                self.study_hits += 1

        def clear(self):
            if self.uid is not None:
                self.studies.append({'study_id': self.uid, 'hits': self.study_hits, 'misses': 1})
            self.uid = None

        def summary(self, collect_timing=True):
            return {'policy': runner.CACHE_POLICY, 'misses': self.misses, 'hits': self.hits,
                    'underlying_vision_calls': self.misses, 'cache_released': self.uid is None,
                    'active_study': self.uid, 'decoder_forwards_per_study': 12,
                    'studies': list(self.studies)}

    def install(teacher):
        calls.append(('cache_install',))
        return cache_module.CachedImageTeacher(teacher, SyntheticCache())

    monkeypatch.setattr(frozen.ImageTeacher, '__init__', initialize)
    monkeypatch.setattr(frozen.ImageTeacher, 'score_logits', score_logits)
    monkeypatch.setattr(frozen.ImageTeacher, 'diagnostic', forbidden_diagnostic)
    monkeypatch.setattr(frozen, 'prepare_t4_teacher', forbidden_diagnostic)
    monkeypatch.setitem(sys.modules, 'numerical_runtime',
                        SimpleNamespace(verify_quantized_compute_dtype=dtype_check))
    monkeypatch.setitem(sys.modules, 'vision_cache', SimpleNamespace(install_cached_teacher=install))
    return SimpleNamespace(calls=calls, state=state, frozen=frozen, torch=fake_torch)


def invoke_real_worker(harness, monkeypatch, worker_id=0):
    directory = harness.output / f'worker_{worker_id}'
    directory.mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', str(worker_id))
    code = runner.worker(harness.request, worker_id, directory)
    return code, json.loads((directory / 'worker_result.json').read_text()), directory


def test_real_worker_uses_frozen_predict_single_gpu_and_no_repeated_diagnostic(harness, synthetic_model, monkeypatch):
    # The production worker, loader, frozen run_inference, frozen ImageTeacher.predict,
    # and frozen CachedImageTeacher composition all execute. Only hardware/model
    # creation, raw logits and vision storage are synthetic.
    code, receipt, directory = invoke_real_worker(harness, monkeypatch)
    shard = sorted(harness.ids)[0::2]
    assert code == 0
    assert receipt['status'] == 'complete'
    assert receipt['model_loaded'] is True
    assert receipt['placement'] == 'single_gpu'
    assert [row for row in synthetic_model.calls if row[0] == 'construct'] == [('construct', 'single_gpu')]
    assert [row for row in synthetic_model.calls if row[0] == 'score'] == [
        ('score', uid, label) for uid in shard for label in LABELS]
    assert len([row for row in synthetic_model.calls if row[0] == 'dtype']) == 2
    assert receipt['score_record_count'] == 12 * len(shard)
    assert receipt['vision_cache']['hits'] == 11 * len(shard)
    assert receipt['vision_cache']['cache_released'] is True
    assert receipt['inference_status']['pre_run_diagnostic'] is None
    metadata = json.loads((directory / 'frozen_runtime.json').read_text())
    assert metadata['diagnostic_performed'] is False
    assert metadata['dtype_report'] == {'kind': 'SYNTHETIC'}
    assert metadata['observed_quantized_compute_dtypes'] == {'torch.float16': 1}
    pd.testing.assert_frame_equal(pd.read_csv(directory / 'inference/submission.csv'), predictions(shard))


def test_empty_assigned_shard_validates_hardware_without_model_load(harness_factory, synthetic_model, monkeypatch):
    harness = harness_factory(1)
    code, receipt, directory = invoke_real_worker(harness, monkeypatch, worker_id=1)
    assert code == 0
    assert receipt['study_ids'] == []
    assert receipt['model_loaded'] is False
    assert receipt['gpu_count'] == 1
    assert receipt['score_record_count'] == 0
    assert receipt['vision_cache']['misses'] == receipt['vision_cache']['hits'] == 0
    assert synthetic_model.calls == []
    prediction = pd.read_csv(directory / 'inference/submission.csv')
    assert prediction.empty
    assert list(prediction.columns) == ['StudyInstanceUID', *LABELS]


@pytest.mark.parametrize('field,value', [('count', 2), ('name', 'SYNTHETIC Other GPU'),
                                       ('major', 8), ('minor', 0), ('total_gib', 13.9)])
def test_real_worker_rejects_hardware_mismatch_before_loading(harness, synthetic_model, monkeypatch, field, value):
    setattr(synthetic_model.state, field, value)
    code, receipt, _directory = invoke_real_worker(harness, monkeypatch)
    assert code == 1
    assert receipt['error_type'] == 'ContractError'
    assert synthetic_model.calls == []


@pytest.mark.parametrize('field,value', [('free_gib', 11.9), ('free_gib', float('nan')),
                                       ('free_gib', float('inf')), ('dtype_error', True),
                                       ('load_error', True)])
def test_real_loader_retains_finite_memory_dtype_and_single_attempt_guards(harness, synthetic_model, monkeypatch, field, value):
    setattr(synthetic_model.state, field, value)
    code, receipt, directory = invoke_real_worker(harness, monkeypatch)
    assert code == 1
    assert receipt['status'] == 'failed'
    assert len([row for row in synthetic_model.calls if row[0] == 'construct']) <= 1
    assert not [row for row in synthetic_model.calls if row[0] == 'score']
    assert not (directory / 'inference/submission.csv').exists()


@pytest.mark.parametrize('key,value', [
    ('native_no_logit', float('nan')), ('native_yes_logit', float('inf')),
    ('yes_probability', float('nan')), ('yes_probability', float('inf')),
    ('yes_probability', -0.01), ('yes_probability', 1.01),
])
def test_nonfinite_actual_forward_evidence_fails_real_worker(harness, synthetic_model, monkeypatch, key, value):
    synthetic_model.state.score_error = (key, value)
    code, receipt, directory = invoke_real_worker(harness, monkeypatch)
    assert code == 1
    assert receipt['error_type'] == 'ContractError'
    assert receipt['status'] == 'failed'
    assert not (harness.output / 'submission.csv').exists()
    cache = json.loads((directory / 'vision_cache_partial.json').read_text())
    assert cache['cache_released'] is True


def test_real_worker_preserves_existing_receipt(harness, synthetic_model, monkeypatch):
    directory = harness.output / 'worker_0'
    directory.mkdir(parents=True)
    receipt_path = directory / 'worker_result.json'
    receipt_path.write_text('{"kind":"SYNTHETIC historical receipt"}')
    before = receipt_path.read_bytes()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    assert runner.worker(harness.request, 0, directory) == 1
    assert receipt_path.read_bytes() == before
    assert synthetic_model.calls == []
