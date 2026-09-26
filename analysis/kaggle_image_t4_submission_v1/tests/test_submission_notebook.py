"""Offline notebook/replay packaging checks; all replay data is synthetic."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from test_submission_runner import (LABELS, ROOT, V8, V10, load_module,
                                    predictions, runner, sha256_file, synthetic_ids)


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
                    and node.func.value.func.id == 'Path' and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                path = ast.literal_eval(node.func.value.args[0])
                embedded[Path(path).name] = ast.literal_eval(node.args[0])
    return embedded


@pytest.fixture
def built_notebook(tmp_path):
    builder = load_module('submission_notebook_contract_builder', ROOT / 'scripts/build_notebook.py')
    output = tmp_path / 'synthetic_package.ipynb'
    assert list(inspect.signature(builder.build).parameters) == ['output']
    result = builder.build(output)
    return builder, output, result, json.loads(output.read_text())


def test_notebook_compiles_and_embeds_exact_frozen_scientific_sources(built_notebook):
    _builder, output, result, notebook = built_notebook
    embedded = embedded_sources(notebook)
    for name, digest in runner.V8_HASHES.items():
        assert hashlib.sha256(embedded[name].encode()).hexdigest() == digest
    assert embedded['vision_cache.py'] == (V10 / 'scripts/vision_cache.py').read_text()
    assert embedded['submission_runner.py'] == (ROOT / 'scripts/submission_runner.py').read_text()
    assert result['sha256'] == sha256_file(output)
    assert result['status'] == 'BUILT_NOT_EXECUTED'


def test_notebook_is_private_offline_and_contains_no_reference_or_id_payload(built_notebook):
    _builder, _output, _result, notebook = built_notebook
    metadata = notebook['metadata']['kaggle']
    assert metadata['isPrivate'] is True
    assert metadata['isInternetEnabled'] is False
    assert metadata['isGpuEnabled'] is True
    assert metadata['dataSources'] == []
    embedded = embedded_sources(notebook)
    assert not any(name.endswith('.csv') for name in embedded)
    assert 'isolated_runner.py' not in embedded
    assert 'replica_runner.py' not in embedded
    assert 'verify_v10_replay.py' not in embedded
    text = json.dumps(notebook)
    for forbidden in ['reference_csv', 'reference_sha256', 'version8_reference.csv',
                      'REFERENCE_SHA', 'SYNTHETIC-', '/Users/', 'KAGGLE_RUN_PASS']:
        assert forbidden not in text
    # No saved execution payload is allowed in a distributable notebook.
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == []
            assert cell['execution_count'] is None


def test_notebook_preserves_offline_install_asset_audit_and_two_gpu_preflight(built_notebook):
    builder, _output, _result, notebook = built_notebook
    cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    setup = next(cell for cell in cells if "os.environ['HF_HUB_OFFLINE']" in cell)
    assert "os.environ['HF_HUB_OFFLINE']='1'" in setup
    assert "os.environ['TRANSFORMERS_OFFLINE']='1'" in setup
    install = next(cell for cell in cells if 'subprocess.check_call' in cell)
    assert "'--no-index'" in install
    assert "'--no-deps'" in install
    assert builder.ASSET_MANIFEST_SHA in install
    assert "digest==row['sha256']" in install
    preflight = next(cell for cell in cells if 'Exactly two T4 replicas required' in cell)
    for required in ['validate_hardware(devices,CONFIG)', 'validate_runtime_versions(runtime,CONFIG)',
                     'audit_adapter(ADAPTER,CONFIG)', 'audit_base_model(BASE,CONFIG,ASSET)',
                     'assert len(devices)==2', 'validate_free_memory(free_gib,']:
        assert required in preflight
    assert 'free_gib[:1]' not in preflight


def test_notebook_future_runtime_budget_is_nine_hours_less_ten_minute_reserve(built_notebook):
    builder, _output, _result, notebook = built_notebook
    assert builder.SESSION_MINUTES == 9 * 60 - 10
    setup = next(cell['source'] for cell in notebook['cells']
                 if "os.environ['HF_HUB_OFFLINE']" in cell['source'])
    assert 'SESSION_DEADLINE_MONOTONIC=time.monotonic()+530*60' in setup
    assert '25*60' not in setup
    assert '115*60' not in setup
    markdown = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    assert 'authorized' in markdown
    assert 'not current compute approval' in markdown
    final = notebook['cells'][-1]['source']
    assert 'run_replicas(request_path,SESSION_DEADLINE_MONOTONIC)' in final
    assert "os.link(source,Path('/kaggle/working/submission.csv'))" in final
    assert final.index('time.monotonic()<SESSION_DEADLINE_MONOTONIC') < final.index('os.link(')
    assert 'sha256_file(source)' in final


def test_submission_dispatch_has_no_active_reference_or_diagnostic_calls(built_notebook):
    _builder, _output, _result, notebook = built_notebook
    source = embedded_sources(notebook)['submission_runner.py']
    called_names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.append(node.func.attr)
    assert not {'diagnostic', 'prepare_t4_teacher', 'compare_reference', 'ActivationTrace'} & set(called_names)
    assert 'ImageTeacher' in called_names
    assert 'install_cached_teacher' in called_names
    assert 'run_inference' in called_names
    assert 'verify_quantized_compute_dtype' in called_names


def test_notebook_builder_never_overwrites_existing_output(built_notebook):
    builder, output, _result, _notebook = built_notebook
    before = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        builder.build(output)
    assert output.read_bytes() == before


@pytest.fixture
def synthetic_replay(tmp_path, monkeypatch):
    module = load_module('submission_synthetic_replay', ROOT / 'scripts/verify_v10_replay.py')
    saved, output = tmp_path / 'synthetic_saved', tmp_path / 'replayed'
    saved.mkdir()
    ids = synthetic_ids(3)
    # Use decimal values that round-trip through the CSV parser, so this fixture
    # tests assembly bytes rather than floating-expression serialization noise.
    baseline = predictions(ids).round(6)
    baseline.to_csv(saved / 'submission.csv', index=False)
    monkeypatch.setattr(module, 'FROZEN_FINAL_SHA', sha256_file(saved / 'submission.csv'))
    for worker_id, shard in enumerate(runner.deterministic_shards(baseline[['StudyInstanceUID']])):
        directory = saved / f'worker_{worker_id}'
        (directory / 'example').mkdir(parents=True)
        path = directory / 'example/submission.csv'
        predictions(shard).round(6).to_csv(path, index=False)
        (directory / 'worker_result.json').write_text(json.dumps({
            'study_ids': shard, 'status': 'complete', 'submission_sha256': sha256_file(path)}))
        (directory / 'score_logits.json').write_text(json.dumps({'scores': [
            {'study_id': uid, 'label': label, 'native_no_logit': 0.1, 'native_yes_logit': 0.2,
             'yes_probability': float(predictions([uid]).iloc[0][label])}
            for uid in shard for label in LABELS]}))
    return module, saved, output


def test_saved_output_replay_is_separate_local_byte_exact_assembly(synthetic_replay):
    module, saved, output = synthetic_replay
    result = module.replay(saved, output)
    assert result['status'] == 'V10_SAVED_OUTPUT_REPLAY_PASS'
    assert result['mode'] == 'LOCAL_REPLAY_NOT_NEW_INFERENCE'
    assert result['byte_identical_final_csv'] is True
    assert result['model_loaded'] is result['gpu_used'] is result['runtime_reference_dependency'] is False
    assert result['scores'] == 36
    assert (output / 'replayed_submission.csv').read_bytes() == (saved / 'submission.csv').read_bytes()


@pytest.mark.parametrize('mutation', ['final', 'worker', 'score_order'])
def test_saved_output_replay_rejects_drift(synthetic_replay, mutation):
    module, saved, output = synthetic_replay
    if mutation in ('final', 'worker'):
        path = saved / ('submission.csv' if mutation == 'final' else 'worker_0/example/submission.csv')
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        path = saved / 'worker_0/score_logits.json'
        data = json.loads(path.read_text())
        data['scores'] = data['scores'][::-1]
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.replay(saved, output)
    assert not output.exists()
