"""Synthetic CPU parent/worker and packaged-notebook cache integration tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT.parent / 'kaggle_image_t4_v1'
V9 = ROOT.parent / 'kaggle_image_t4_replicas_v1'
sys.path.insert(0, str(V8 / 'scripts'))
from inference_core import ContractError, LABELS, sha256_file  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module('cache_replica_test_target', ROOT / 'scripts/replica_runner.py')
old = load_module('v9_synthetic_test_utilities', V9 / 'tests/test_replica_runner.py')

EXTRA_WORKER = r'''
timings = [{'StudyInstanceUID': uid, 'study_seconds': 31.25} for uid in ids]
if scenario == 'exact_40':
    timings[0]['study_seconds'] = 40.0
elif scenario == 'just_under_40':
    timings[0]['study_seconds'] = 39.999999
elif scenario == 'above_40':
    timings[0]['study_seconds'] = 40.000001
elif scenario == 'timing_nan':
    timings[0]['study_seconds'] = float('nan')
elif scenario == 'timing_inf':
    timings[0]['study_seconds'] = float('inf')
elif scenario == 'timing_zero':
    timings[0]['study_seconds'] = 0.0
elif scenario == 'timing_negative':
    timings[0]['study_seconds'] = -1.0
elif scenario == 'timing_missing_row':
    timings = timings[:-1]
if scenario != 'timing_missing_file':
    with (output.parent / 'study_timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['StudyInstanceUID', 'study_seconds'])
        writer.writeheader()
        writer.writerows(timings)
receipt['vision_cache'] = {
    'policy': 'same_study_exact_pixel_vision_cache_v1', 'misses': len(ids),
    'hits': 11 * len(ids), 'underlying_vision_calls': len(ids),
    'cache_released': True, 'active_study': None,
    'decoder_forwards_per_study': 12,
    'studies': [{'study_id': uid, 'misses': 1, 'hits': 11} for uid in ids],
}
if scenario == 'cache_missing':
    del receipt['vision_cache']
elif scenario == 'cache_misses':
    receipt['vision_cache']['misses'] += 1
elif scenario == 'cache_hits':
    receipt['vision_cache']['hits'] -= 1
elif scenario == 'cache_calls':
    receipt['vision_cache']['underlying_vision_calls'] += 1
elif scenario == 'cache_unreleased':
    receipt['vision_cache']['cache_released'] = False
elif scenario == 'cache_decoder_count':
    receipt['vision_cache']['decoder_forwards_per_study'] = 11
'''
NEEDLE = "(worker_dir / 'worker_result.json').write_text(json.dumps(receipt))"
assert old.SYNTHETIC_WORKER.count(NEEDLE) == 1
SYNTHETIC_WORKER = old.SYNTHETIC_WORKER.replace(NEEDLE, EXTRA_WORKER + '\n' + NEEDLE)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    bundle = tmp_path / 'byte_frozen_sources'
    bundle.mkdir()
    for name in runner.V8_HASHES:
        path = V8 / ('configs/inference.json' if name == 'inference_config.json' else f'scripts/{name}')
        shutil.copyfile(path, bundle / name)
    for name in ['replica_runner.py', 'vision_cache.py']:
        shutil.copyfile(ROOT / 'scripts' / name, bundle / name)
    data = tmp_path / 'synthetic_test_data'
    data.mkdir()
    pd.DataFrame({'StudyInstanceUID': old.IDS}).to_csv(data / 'test.csv', index=False)
    old.predictions().to_csv(data / 'sample_submission.csv', index=False)
    pd.DataFrame([{'StudyInstanceUID': uid, 'SeriesInstanceUID': f'SYNTHETIC-series-{i}',
        'Fluid_Sensitive': 1, 'Fat_Suppression': 1, 'Anatomical_Plane': 'SAG'}
        for i, uid in enumerate(old.IDS)]).to_csv(data / 'test_series.csv', index=False)
    reference = tmp_path / 'SYNTHETIC_reference.csv'
    old.predictions().to_csv(reference, index=False)
    request = tmp_path / 'request.json'
    output = tmp_path / 'output'
    request.write_text(json.dumps({
        'config_path': str(bundle / 'inference_config.json'), 'data_dir': str(data),
        'base_dir': str(tmp_path / 'base'), 'adapter_dir': str(tmp_path / 'adapter'),
        'output_root': str(output), 'reference_csv': str(reference),
        'reference_sha256': sha256_file(reference),
        'vision_cache_sha256': sha256_file(bundle / 'vision_cache.py'),
        'asset_verification_seconds': 0.01, 'dependency_install_seconds': 0.01,
    }))
    worker = tmp_path / 'synthetic_worker.py'
    worker.write_text(SYNTHETIC_WORKER)
    events = tmp_path / 'events.jsonl'
    launched = []

    def install(scenarios=None):
        scenarios = scenarios or {0: 'complete', 1: 'complete'}

        def command(actual_request, worker_id, worker_dir):
            assert Path(actual_request) == request
            launched.append(worker_id)
            assert launched.count(worker_id) == 1, 'Retries are forbidden'
            return [sys.executable, str(worker), str(request), str(worker_id),
                    str(worker_dir), str(events), scenarios[worker_id]]

        monkeypatch.setattr(runner, '_worker_command', command)

    install()
    return request, output, events, launched, install, bundle


def run(harness):
    return runner.run_replicas(harness[0], time.monotonic() + 15, cuda=old.FakeCuda())


def test_real_parent_merges_three_studies_36_scores_with_exact_cache_accounting(harness):
    request, output, events, launched, _install, _bundle = harness
    result = run(harness)
    assert result['status'] == 'UNDER40_PARITY_PASS'
    assert result['under40_target_met'] is True
    assert result['parity']['compared_scores'] == 36
    assert result['parity']['max_absolute_difference'] == 0
    assert result['complete_study_seconds'] == [31.25, 31.25, 31.25]
    assert result['maximum_complete_study_seconds'] == 31.25
    actual = pd.read_csv(output / 'submission.csv')
    assert actual.StudyInstanceUID.tolist() == old.IDS
    pd.testing.assert_frame_equal(actual, pd.read_csv(json.loads(request.read_text())['reference_csv']))
    assert launched == [0, 1]
    for row in old.events_at(events):
        assert row['visible_devices'] == str(row['worker_id'])
        old.assert_reaped(row['pid'])


@pytest.mark.parametrize('scenario,expected', [
    ('just_under_40', True), ('exact_40', False), ('above_40', False),
])
def test_complete_study_threshold_is_strict_not_average_or_rounded(harness, scenario, expected):
    harness[4]({0: scenario, 1: 'complete'})
    result = run(harness)
    assert result['under40_target_met'] is expected
    assert result['status'] == ('UNDER40_PARITY_PASS' if expected else 'PARITY_PASS_TARGET_NOT_MET')
    assert result['parity']['passed'] is True


@pytest.mark.parametrize('scenario', [
    'timing_missing_file', 'timing_missing_row', 'timing_nan', 'timing_inf',
    'timing_zero', 'timing_negative',
])
def test_absent_or_invalid_complete_study_timing_never_publishes_final(harness, scenario):
    harness[4]({0: scenario, 1: 'complete'})
    with pytest.raises((ContractError, OSError, ValueError)):
        run(harness)
    assert not (harness[1] / 'submission.csv').exists()
    assert harness[3] == [0, 1]
    for row in old.events_at(harness[2]):
        old.assert_reaped(row['pid'])


@pytest.mark.parametrize('scenario', [
    'cache_missing', 'cache_misses', 'cache_hits', 'cache_calls',
    'cache_unreleased', 'cache_decoder_count',
])
def test_parent_rejects_bad_cache_accounting_even_with_valid_predictions(harness, scenario):
    harness[4]({0: scenario, 1: 'complete'})
    with pytest.raises(ContractError):
        run(harness)
    assert not (harness[1] / 'submission.csv').exists()
    assert harness[3] == [0, 1]
    for row in old.events_at(harness[2]):
        old.assert_reaped(row['pid'])


def test_material_score_drift_stops_despite_fast_valid_cache_accounting(harness):
    harness[4]({0: 'bad_parity', 1: 'complete'})
    with pytest.raises(ContractError, match='parity failed'):
        run(harness)
    assert not (harness[1] / 'submission.csv').exists()


@pytest.mark.parametrize('name', ['vision_cache.py', *runner.V8_HASHES])
def test_parent_rejects_cache_or_frozen_source_drift_before_launch(harness, name):
    target = harness[5] / name
    target.write_bytes(target.read_bytes() + b'\nSYNTHETIC source drift\n')
    with pytest.raises(ContractError):
        run(harness)
    assert harness[3] == []
    assert old.events_at(harness[2]) == []


@pytest.mark.parametrize('name', ['vision_cache.py', 'submission_runtime.py'])
def test_real_worker_checks_source_hash_before_model_loading(harness, monkeypatch, name):
    target = harness[5] / name
    target.write_bytes(target.read_bytes() + b'\nSYNTHETIC source drift\n')
    worker_dir = harness[1] / 'worker_0'
    worker_dir.mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    class ForbiddenTorch:
        def __getattr__(self, _name):
            raise AssertionError('CUDA/model may not load after source drift')

    monkeypatch.setitem(sys.modules, 'torch', ForbiddenTorch())
    assert runner.worker(harness[0], 0, worker_dir) == 1
    receipt = json.loads((worker_dir / 'worker_result.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['error_type'] == 'ContractError'
    assert 'checksum mismatch' in receipt['error'] or 'source drift' in receipt['error']


@pytest.fixture
def builder(tmp_path, monkeypatch):
    module = load_module('cache_builder_test_target', ROOT / 'scripts/build_notebook.py')
    reference = tmp_path / 'SYNTHETIC_v8_reference.csv'
    old.predictions().to_csv(reference, index=False)
    monkeypatch.setattr(module, 'REFERENCE_SHA', sha256_file(reference))
    return module, reference, tmp_path / 'synthetic_notebook.ipynb'


def test_notebook_cells_compile_and_embed_exact_frozen_sources_and_cache(builder):
    module, reference, output = builder
    module.build(output, reference)
    notebook = json.loads(output.read_text())
    embedded = old.embedded_sources(notebook)
    for name, digest in runner.V8_HASHES.items():
        assert hashlib.sha256(embedded[name].encode()).hexdigest() == digest
    assert embedded['vision_cache.py'] == (ROOT / 'scripts/vision_cache.py').read_text()
    assert embedded['replica_runner.py'] == (ROOT / 'scripts/replica_runner.py').read_text()
    assert 'isolated_runner.py' not in embedded
    final = notebook['cells'][-1]['source']
    assert sha256_file(ROOT / 'scripts/vision_cache.py') in final
    assert 'run_replicas(request_path,SESSION_DEADLINE_MONOTONIC)' in final
    assert notebook['metadata']['kaggle']['isInternetEnabled'] is False


def test_notebook_has_25_minute_inner_deadline_and_no_old_115_minute_limit(builder):
    module, reference, output = builder
    module.build(output, reference)
    notebook = json.loads(output.read_text())
    setup = next(cell['source'] for cell in notebook['cells'] if "os.environ['HF_HUB_OFFLINE']" in cell['source'])
    assert 'SESSION_DEADLINE_MONOTONIC=time.monotonic()+25*60' in setup
    assert '115*60' not in setup
    assert "os.environ['TRANSFORMERS_OFFLINE']='1'" in setup
    assert "os.environ['HF_HUB_OFFLINE']='1'" in setup


def test_notebook_builder_rejects_changed_reference_and_existing_output(builder):
    module, reference, output = builder
    module.build(output, reference)
    original = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        module.build(output, reference)
    assert output.read_bytes() == original
    reference.write_text(reference.read_text() + '\nSYNTHETIC drift\n')
    with pytest.raises(ValueError, match='exact downloaded'):
        module.build(output.parent / 'other.ipynb', reference)
