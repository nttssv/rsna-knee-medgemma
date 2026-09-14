"""Synthetic tests only: no clinical reports, model downloads, GPU use or real identifiers."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'analysis/report_labeling_llm_v1/scripts'
sys.path.insert(0, str(SCRIPTS))
import benchmark_core as core
from analysis_metrics import checked_frame, metrics, agreement_frames, paired_bootstrap
from repeatability_check import compare as repeat_compare


def output(label='not_mentioned', evidence='', confidence=0):
    return {c: {'label': label, 'evidence_text': evidence, 'confidence': confidence} for c in core.LABELS}


def test_schema_four_states_and_exact_original_evidence():
    report = 'An invented sample: no fracture. Other condition uncertain.'
    obj = output()
    obj['Fracture'] = {'label': 'negative', 'evidence_text': 'no fracture.', 'confidence': .9}
    obj['ACL'] = {'label': 'uncertain', 'evidence_text': 'Other condition uncertain.', 'confidence': .5}
    rows = core.validate_response(json.dumps(obj), report)
    assert all(r['status'] == 'valid' for r in rows)
    for r in rows:
        if r['evidence_text']:
            assert report[r['evidence_start']:r['evidence_end']] == r['evidence_text']
    assert rows[-1]['extracted_label'] == 'negative'


@pytest.mark.parametrize('value', [True, -0.1, 1.1, '0.8', None, float('nan'), float('inf')])
def test_invalid_confidence_never_becomes_a_label(value):
    obj = output(); obj['ACL']['confidence'] = value
    row = core.validate_response(json.dumps(obj), 'synthetic')[0]
    assert row['status'] in {'schema_error', 'parse_error'}
    assert row['extracted_label'] is None


@pytest.mark.parametrize('label,evidence', [('positive', ''), ('negative', ''), ('uncertain', ''),
                                         ('positive', 'fabricated'), ('not_mentioned', 'sample')])
def test_unsupported_evidence_is_a_failure_not_a_negative(label, evidence):
    obj = output(); obj['ACL'] = {'label': label, 'evidence_text': evidence, 'confidence': .9}
    row = core.validate_response(json.dumps(obj), 'sample')[0]
    assert row['status'] == 'evidence_error' and row['extracted_label'] is None


def test_missing_duplicate_extra_conditions_and_fenced_json():
    obj = output(); del obj['ACL']
    assert core.validate_response(json.dumps(obj), '')[0]['status'] == 'missing_condition'
    obj['unexpected'] = {}; assert core.validate_response(json.dumps(obj), '')[0]['status'] == 'schema_error'
    assert core.validate_response('{"ACL":{},"ACL":{}}', '')[0]['status'] == 'duplicate_condition'
    assert core.validate_response('```json\n{}\n```', '')[0]['status'] == 'parse_error'


def test_repair_preserves_first_pass_and_has_a_one_call_limit():
    calls = []
    def generate(prompt):
        calls.append(prompt)
        return {'text': 'not JSON' if len(calls) == 1 else json.dumps(output()), 'status': 'completed'}
    result = core.extract_case(generate, 'instructions', 'sample')
    assert len(calls) == 2 and result['repair_attempted']
    assert result['first_pass'][0]['extracted_label'] is None
    assert result['repair_assisted'][0]['extracted_label'] == 'not_mentioned'
    assert result['repair_changed_labels'] == 12
    calls.clear()
    def bad(prompt):
        calls.append(prompt); return {'text': 'bad', 'status': 'completed'}
    assert core.extract_case(bad, '', '')['repair_assisted'][0]['status'] == 'parse_error'
    assert len(calls) == 2


@pytest.mark.parametrize('status', ['oom', 'runtime_error', 'timeout', 'context_overflow', 'generation_truncated'])
def test_infrastructure_failures_do_not_trigger_semantic_repair(status):
    calls = []
    def generate(prompt):
        calls.append(prompt); return {'text': '', 'status': status}
    result = core.extract_case(generate, '', '')
    assert len(calls) == 1 and not result['repair_attempted']
    assert all(r['extracted_label'] is None for r in result['first_pass'])


def test_prompt_equivalence_and_join_keys_not_injected():
    report = 'A synthetic report string.'
    assert core.prompt_for('medgemma', report) == core.prompt_for('qwen', report)
    prompt = core.prompt_for('medgemma', report)
    assert 'StudyInstanceUID' not in prompt and 'organizer_label' not in prompt
    assert json.dumps(report) in prompt
    cfg = core.config('qwen')
    assert cfg['enable_thinking'] is False and cfg['do_sample'] is False


def frame(labels, gold=None, statuses=None):
    rows = []
    for i, label in enumerate(labels):
        for c in core.LABELS:
            rows.append(dict(StudyInstanceUID=f'synthetic_{i}', condition=c,
                organizer_label=(gold or [0] * len(labels))[i], extracted_label=label,
                status=(statuses or ['valid'] * len(labels))[i], confidence=0., language='English', evidence_text=''))
    return checked_frame(pd.DataFrame(rows), [f'synthetic_{i}' for i in range(len(labels))])


def test_denominator_accounts_for_failures_and_medical_abstentions():
    f = frame(['positive', 'negative', 'uncertain', 'not_mentioned', None], [1, 1, 0, 1, 0],
              ['valid', 'valid', 'valid', 'valid', 'oom'])
    m = metrics(f).iloc[0]
    assert (m.TP, m.FN, m.technical_failures, m.uncertain, m.not_mentioned) == (1, 1, 1, 1, 1)
    assert m.coverage == .4 and m.all_study_correct_label_yield == .2
    assert m.accuracy == .5 and m.conditional_error_rate == .5


def test_no_decisions_produces_undefined_conditional_metrics():
    m = metrics(frame(['not_mentioned', None], statuses=['valid', 'oom'])).iloc[0]
    assert m.coverage == 0 and pd.isna(m.accuracy) and pd.isna(m.precision_PPV)
    assert m.all_study_correct_label_yield == 0


def test_agreement_never_accepts_shared_abstentions_or_shared_failures():
    f = frame(['not_mentioned', 'uncertain', None, 'positive'], [0, 1, 0, 1], ['valid', 'valid', 'oom', 'valid'])
    ensembles, diagnostic = agreement_frames({'rule_v1': f, 'medgemma': f.copy(), 'qwen': f.copy()})
    assert all(int(x.extracted_label.notna().sum()) == 12 for x in ensembles.values())
    assert all(metrics(x).iloc[0].coverage == .25 for x in ensembles.values())
    assert diagnostic[diagnostic.matching_state == 'not_mentioned']['count'].sum() == 36


def test_disagreement_and_reference_misalignment_rejected():
    a = frame(['positive', 'negative']); b = frame(['negative', 'positive'])
    ensembles, _ = agreement_frames({'rule_v1': a, 'medgemma': a.copy(), 'qwen': b})
    assert ensembles['medgemma_qwen_binary'].extracted_label.isna().all()
    bad = a.copy(); bad.loc[0, 'organizer_label'] = 1
    with pytest.raises(ValueError, match='aligned'):
        agreement_frames({'rule_v1': a, 'medgemma': bad, 'qwen': b})


def test_duplicates_and_missing_prediction_rows_rejected():
    f = frame(['positive'])
    with pytest.raises(ValueError, match='exactly one'):
        checked_frame(pd.concat([f, f.iloc[:1]]), ['synthetic_0'])
    with pytest.raises(ValueError, match='exactly one'):
        checked_frame(f.iloc[1:], ['synthetic_0'])


def test_paired_bootstrap_resamples_whole_studies_and_preserves_pairing():
    f = frame(['positive', 'negative', 'uncertain', 'negative'], [1, 0, 1, 1])
    overall, delta = paired_bootstrap({'rule_v1': f, 'medgemma': f.copy()}, replicates=100, seed=5)
    assert set(overall.studies) == {4}
    assert np.allclose(delta[['delta', 'ci_low', 'ci_high']].to_numpy(), 0)
    assert overall[(overall.model == 'rule_v1') & (overall.metric == 'all_study_correct_label_yield')].estimate.iloc[0] == .5


def test_repeatability_rejects_identical_failures():
    rows = [{'StudyInstanceUID': str(i), 'first_pass': core.failure('oom'), 'repair_assisted': core.failure('oom')} for i in range(5)]
    assert not repeat_compare(rows, rows)['passed']
    for r in rows:
        r['first_pass'] = core.validate_response(json.dumps(output()), '')
        r['repair_assisted'] = r['first_pass']
    assert repeat_compare(rows, rows)['passed']


@pytest.fixture
def private_dataset(tmp_path, monkeypatch):
    state = tmp_path / 'private_state'; (state / 'data').mkdir(parents=True)
    snapshot = state / 'runs/report-labeling-20260913-v1'; snapshot.mkdir(parents=True)
    source = state / 'data/train.csv'; splits = snapshot / 'splits.csv'; chars = snapshot / 'report_characteristics.csv'
    with source.open('w') as f, splits.open('w') as sf, chars.open('w') as cf:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID', 'Report'] + core.LABELS); writer.writeheader()
        split = csv.DictWriter(sf, fieldnames=['StudyInstanceUID', 'split', 'report_sha256']); split.writeheader()
        language = csv.DictWriter(cf, fieldnames=['StudyInstanceUID', 'language_heuristic']); language.writeheader()
        for i in range(4407):
            uid, report = f'fixture_{i:04d}', f'Synthetic report number {i}.'
            writer.writerow(dict(StudyInstanceUID=uid, Report=report, **{c: i % 2 if i < 58 else '' for c in core.LABELS}))
            if i < 58:
                split.writerow(dict(StudyInstanceUID=uid, split='development' if i < 40 else 'validation', report_sha256=core.text_sha(report.lower())))
                language.writerow(dict(StudyInstanceUID=uid, language_heuristic='English'))
    reviewed = snapshot / 'report_characteristics_reviewed.csv'
    with reviewed.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID', 'language_analyst_review']); writer.writeheader()
        for i in range(58): writer.writerow({'StudyInstanceUID': f'fixture_{i:04d}', 'language_analyst_review': 'English'})
    baseline = snapshot / 'long_extractions.csv'
    with baseline.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID', 'condition', 'organizer_label', 'split', 'confidence', 'extracted_label', 'evidence_text']); writer.writeheader()
        for i in range(58):
            for c in core.LABELS:
                writer.writerow(dict(StudyInstanceUID=f'fixture_{i:04d}', condition=c, organizer_label=i % 2,
                    split='development' if i < 40 else 'validation', confidence=0., extracted_label='not_mentioned', evidence_text=''))
    (snapshot / 'evaluation_manifest.json').write_text(json.dumps({'output_sha256': {'report_characteristics.csv': core.sha(chars), 'long_extractions.csv': core.sha(baseline)}}))
    old_config = core.config
    cfg = dict(old_config('benchmark'), source_sha256=core.sha(source), split_sha256=core.sha(splits), language_review_sha256=core.sha(reviewed))
    monkeypatch.setattr(core, 'config', lambda name: cfg if name == 'benchmark' else old_config(name))
    return state


def test_preparation_restricts_to_58_and_removes_all_gold(private_dataset):
    state = private_dataset; prepared = state / 'runs/llm/inputs'
    result = core.prepare(state, prepared)
    assert result['partition_sizes'] == {'development': 40, 'validation': 18}
    dev = core.load_inputs(state, prepared, 'development'); val = core.load_inputs(state, prepared, 'validation')
    assert len(dev) == 40 and len(val) == 18
    assert all(set(r) == core.INPUT_KEYS for r in dev + val)
    assert not set(r['StudyInstanceUID'] for r in dev) & set(r['StudyInstanceUID'] for r in val)
    with pytest.raises(FileExistsError): core.prepare(state, prepared)


def test_input_tampering_and_public_output_path_rejected(private_dataset):
    state = private_dataset; prepared = state / 'runs/llm/inputs'; core.prepare(state, prepared)
    with (prepared / 'validation.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='fingerprint'): core.load_inputs(state, prepared, 'validation')
    with pytest.raises(ValueError, match='private state'): core.private_path(state, core.ROOT / 'aggregate/data.csv')
    with pytest.raises(ValueError, match='ignored state'): core.state_dir(core.ROOT)


def test_source_or_split_edits_fail_closed(private_dataset):
    state = private_dataset
    (state / 'runs/report-labeling-20260913-v1/splits.csv').write_text('changed')
    with pytest.raises(ValueError, match='fingerprint'): core.checked_source(state)


def test_freeze_integrity_rejects_prompt_changes(tmp_path):
    prepared = tmp_path / 'inputs'; prepared.mkdir(); (prepared / 'inputs_manifest.json').write_text('{}')
    manifest = {'code_sha256': core.code_hashes(), 'benchmark': core.config('benchmark'),
                'inputs_manifest_sha256': core.sha(prepared / 'inputs_manifest.json'),
                'development_studies_per_model': {'medgemma': 40, 'qwen': 40}, 'repeatability': {'passed': True}}
    path = tmp_path / 'freeze.json'; path.write_text(json.dumps(manifest))
    assert core.verify_freeze(path, prepared)
    manifest['code_sha256']['invented_file.py'] = 'changed'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Frozen LLM'): core.verify_freeze(path, prepared)


def fake_run(state, prepared, name, partition, freeze_path=None):
    rows = core.load_inputs(state, prepared, partition)
    out = state / 'runs' / (name + '_' + partition); out.mkdir()
    predictions = []
    for row in rows:
        prediction = {k: row[k] for k in ['StudyInstanceUID', 'split', 'report_sha256', 'language']}
        parsed = core.validate_response(json.dumps(output()), row['Report'])
        prediction.update(first_pass=parsed, repair_assisted=parsed, attempts=[], repair_attempted=False, repair_changed_labels=0)
        predictions.append(prediction)
    core.write_jsonl(out / 'predictions.jsonl', predictions)
    core.write_json(out / 'run_manifest.json', dict(model_key=name, model_config=core.config(name), partition=partition,
        status='completed', studies_processed=len(rows), code_sha256=core.code_hashes(),
        freeze_sha256=core.sha(freeze_path) if freeze_path else None,
        output_sha256={'predictions.jsonl': core.sha(out / 'predictions.jsonl')}))
    return out


def test_full_synthetic_prepare_freeze_evaluate_path(private_dataset, monkeypatch):
    import evaluate_llm
    state = private_dataset; prepared = state / 'runs/llm/inputs'; core.prepare(state, prepared)
    med = fake_run(state, prepared, 'medgemma', 'development')
    qwen = fake_run(state, prepared, 'qwen', 'development')
    repeat = state / 'runs/repeatability.json'
    core.write_json(repeat, dict(code_sha256=core.code_hashes(), passed=True, models_checked=['medgemma', 'qwen'], studies_per_model=5))
    frozen = state / 'runs/freeze.json'; core.freeze(state, prepared, med, qwen, repeat, frozen)
    valmed = fake_run(state, prepared, 'medgemma', 'validation', frozen)
    valqwen = fake_run(state, prepared, 'qwen', 'validation', frozen)
    target = state / 'runs/evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate_llm.py', '--state-dir', str(state), '--prepared', str(prepared),
        '--medgemma-run', str(valmed), '--qwen-run', str(valqwen), '--freeze', str(frozen), '--output', str(target)])
    evaluate_llm.main()
    result = pd.read_csv(target / 'first_pass/model_comparison.csv')
    assert len(result) == 36 and set(result.studies) == {18}
    assert result.coverage.eq(0).all() and result.accuracy.isna().all()
    assert (target / 'first_pass/private_disagreements.csv').is_file()
    assert len(pd.read_csv(target / 'first_pass/private_disagreements.csv')) == 216
    assert json.loads((target / 'evaluation_manifest.json').read_text())['recommendation'].startswith('E_')


def test_tampered_run_is_rejected_before_evaluation(private_dataset):
    state = private_dataset; prepared = state / 'runs/llm/inputs'; core.prepare(state, prepared)
    run = fake_run(state, prepared, 'medgemma', 'development')
    (run / 'predictions.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='fingerprint'):
        core.verify_run(state, run, 'medgemma', 'development', prepared)


@pytest.mark.parametrize('overflow_model', ['medgemma', 'qwen'])
def test_preflight_blocks_either_models_overflow(tmp_path, overflow_model):
    from tokenizer_preflight import verify_preflight
    manifest = {'code_sha256': core.code_hashes(), 'files': {}, 'models': {
        name: {'studies': 58, 'overflow_studies': int(name == overflow_model)}
        for name in ['medgemma', 'qwen']}}
    path = tmp_path / 'preflight_manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='context overflow'):
        verify_preflight(tmp_path)
    manifest['models'][overflow_model]['overflow_studies'] = 0
    path.write_text(json.dumps(manifest))
    assert verify_preflight(tmp_path)['models'][overflow_model]['overflow_studies'] == 0


def test_medgemma_preserves_processor_inputs_without_bare_tokenization():
    from model_runtime import Tokenizer
    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': [{'type': 'text', 'text': 'synthetic'}]}]
            assert kwargs['add_generation_prompt'] is True
            if not kwargs['tokenize']:
                return 'official rendered synthetic'
            assert kwargs['return_dict'] and kwargs['return_tensors'] == 'pt'
            assert kwargs['truncation'] is False
            return {'input_ids': [[1, 2]], 'attention_mask': [[1, 1]], 'token_type_ids': [[0, 0]]}
    encoder = Tokenizer.__new__(Tokenizer)
    encoder.name, encoder.processor = 'medgemma', FakeProcessor()
    # No underlying tokenizer is attached: dropping to it would fail this test.
    assert encoder.render('synthetic') == 'official rendered synthetic'
    assert encoder.encode('synthetic')['token_type_ids'] == [[0, 0]]


def test_qwen_preserves_non_thinking_template_and_no_truncation():
    from model_runtime import Tokenizer
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'synthetic'}]
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return 'rendered synthetic'
        def __call__(self, text, **kwargs):
            assert text == 'rendered synthetic'
            assert kwargs == {'add_special_tokens': False, 'return_tensors': 'pt', 'truncation': False}
            return {'input_ids': [[1, 2]]}
    encoder = Tokenizer.__new__(Tokenizer)
    encoder.name = 'qwen'
    encoder.processor = encoder.tokenizer = FakeTokenizer()
    assert encoder.encode('synthetic')['input_ids'] == [[1, 2]]
