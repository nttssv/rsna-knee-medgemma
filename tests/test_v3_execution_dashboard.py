"""Offline dashboard refuses altered evidence and preserves missing-work denominators."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / 'analysis/report_labeling_llm_v3_execution_v1/diagnostics/build_dashboard.py'
spec = importlib.util.spec_from_file_location('v3_execution_dashboard', PATH)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def measured(tmp_path, monkeypatch):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    report = 'Fictional fixture <script>alert(1)</script>.'
    plan = dict(synthetic=False, model=dict(model_id='fixture', revision='fixture'))
    source = dict(Report=report, StudyInstanceUID='fixture-id', report_sha256='fixture-hash', language='English')
    monkeypatch.setattr(d.rt, 'checked_plan', lambda *args: (plan, [source]))
    monkeypatch.setattr(d.rt, 'context', lambda *args: {'test': True})
    raw_output = json.dumps({c: dict(label='not_mentioned', evidence_text='', confidence=0) for c in d.rt.core.LABELS})
    generation = dict(raw_output=raw_output, generation_status='completed', runtime_seconds=1., output_tokens=100, input_tokens=100)
    record = dict(synthetic=False, case_index=0, StudyInstanceUID='fixture-id', report_sha256='fixture-hash',
        context={'test': True}, prompt=d.rt.candidate.make_prompt('control', report), rendered_prompt=report, generation=generation)
    response = d.rt.candidate.validate_v3(raw_output, report, 'completed')
    prediction = dict(case_index=0, raw_record_sha256=d.rt.digest(record), response=response)
    d.rt.write(prepared / 'plan.json', plan)
    (prepared / 'session/control-1').mkdir(parents=True)
    d.rt.write(prepared / 'session/result.json', dict(status='failed'))
    for name, row in [('raw', record), ('predictions', prediction)]:
        (prepared / f'session/control-1/{name}.jsonl').write_text(json.dumps(row) + '\n')
    inventory = tmp_path / 'inventory.json'
    def bind():
        inventory.write_text(json.dumps({str(p.relative_to(prepared)): d.rt.sha(p) for p in prepared.rglob('*') if p.is_file()}))
        return prepared, inventory, d.rt.sha(inventory)
    return bind, plan


def test_partial_run_keeps_all_planned_cells_and_no_semantic_claim(measured):
    bind, _ = measured
    plan, session, outputs = d.load_verified(*bind())
    summary = d.summarize(plan, session, outputs)
    assert summary['planned_cells'] == 240
    assert summary['recorded_cells'] == 12
    assert summary['attempted_generations_lower_bound'] == 1
    assert summary['runs'][0]['states'] == {'not_mentioned': 12}
    assert all(x['binary_decisions'] == 0 for x in summary['runs'])
    assert summary['runs'][1]['attempted'] == 0
    assert summary['semantic_accuracy'] is None
    assert summary['semantically_reviewed_cells'] == 0
    assert summary['expansion_allowed'] is False


def test_changed_bytes_rejected(measured):
    bind, _ = measured
    args = bind()
    (args[0] / 'session/control-1/raw.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='changed'):
        d.load_verified(*args)


def test_unexpected_file_rejected(measured):
    bind, _ = measured
    args = bind()
    (args[0] / 'unexpected.json').write_text('{}')
    with pytest.raises(ValueError, match='file set'):
        d.load_verified(*args)


def test_rebound_but_inconsistent_parsing_rejected(measured):
    bind, _ = measured
    args = bind()
    path = args[0] / 'session/control-1/predictions.jsonl'
    value = json.loads(path.read_text())
    value['response']['rows'][0]['label'] = 'positive'
    path.write_text(json.dumps(value) + '\n')
    with pytest.raises(ValueError, match='reparse'):
        d.load_verified(*bind())


def test_synthetic_is_never_measured(measured):
    bind, plan = measured
    plan['synthetic'] = True
    with pytest.raises(ValueError, match='real measured'):
        d.load_verified(*bind())


def test_private_viewer_escapes_report_and_refuses_overwrite(measured, tmp_path):
    bind, _ = measured
    output = tmp_path / 'viewer'
    d.build(*bind(), output)
    html = (output / 'index.html').read_text()
    assert '<script>alert(1)</script>' not in html
    assert '\\u003cscript\\u003e' in html
    assert 'fixture-id' not in html
    assert 'review_context' not in html
    with pytest.raises(ValueError, match='new private'):
        d.build(*bind(), output)
