"""Offline technical dashboard; never unlocks semantic review or organizer comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import runtime_v3_execution as rt


def load_verified(prepared, inventory, inventory_sha256):
    prepared = rt.private(prepared)
    if rt.sha(inventory) != inventory_sha256:
        raise ValueError('Transferred inventory checksum mismatch')
    hashes = rt.read(inventory)
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Empty inventory')
    actual = {str(p.relative_to(prepared)) for p in prepared.rglob('*') if p.is_file()}
    if set(hashes) != actual:
        raise ValueError('Incomplete or unexpected transferred file set')
    for name, digest in hashes.items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared) or rt.sha(path) != digest:
            raise ValueError('Transferred artifact changed')
    plan, inputs = rt.checked_plan(prepared, rt.sha(prepared / 'plan.json'))
    if plan['synthetic']:
        raise ValueError('This dashboard requires real measured outputs')
    session = rt.read(prepared / 'session/result.json')
    if session['status'] not in ('completed', 'failed'):
        raise ValueError('Session must be finalized before analysis')
    if session['status'] == 'completed':
        rt.verify_session(prepared)
    outputs = []
    for key in rt.ORDER:
        folder = prepared / 'session' / key
        raw = rt.lines(folder / 'raw.jsonl') if (folder / 'raw.jsonl').exists() else []
        parsed = rt.lines(folder / 'predictions.jsonl') if (folder / 'predictions.jsonl').exists() else []
        if len(raw) > 5 or len(raw) != len(parsed):
            raise ValueError('Incomplete raw/parsed pairing')
        for i, (record, prediction) in enumerate(zip(raw, parsed)):
            source = inputs[i]
            gen = record['generation']
            expected = rt.candidate.validate_v3(gen['raw_output'], source['Report'], gen['generation_status'])
            if (record['synthetic'] or record['case_index'] != i or
                record['StudyInstanceUID'] != source['StudyInstanceUID'] or
                record['report_sha256'] != source['report_sha256'] or
                record['context'] != rt.context(prepared, key) or
                record['prompt'] != rt.candidate.make_prompt(key.split('-')[0], source['Report']) or
                prediction != dict(case_index=i, raw_record_sha256=rt.digest(record), response=expected)):
                raise ValueError('Response does not match exact source, prompt or reparse')
            outputs.append(dict(run=key, case_index=i, language=source['language'],
                report=source['Report'], prompt=record['rendered_prompt'], generation=gen,
                response=expected))
    if not outputs:
        raise ValueError('No measured response to display')
    return plan, session, outputs


def summarize(plan, session, outputs):
    runs, conditions, languages = [], [], []
    for key in rt.ORDER:
        selected = [x for x in outputs if x['run'] == key]
        cells = [c for x in selected for c in x['response']['rows']]
        statuses = Counter(c['status'] for c in cells)
        states = Counter(c['label'] for c in cells if c['status'] == 'valid')
        runs.append(dict(run=key, planned_generations=5, attempted=len(selected),
            completed=sum(x['generation']['generation_status'] == 'completed' for x in selected),
            planned_cells=60, attempted_cells=len(cells), technical_statuses=dict(statuses),
            valid=states.total(), states=dict(states), binary_decisions=states['positive'] + states['negative'],
            generation_seconds=sum(x['generation']['runtime_seconds'] for x in selected),
            output_tokens=sum(x['generation']['output_tokens'] for x in selected)))
        for condition in rt.core.LABELS:
            values = [c for c in cells if c['condition'] == condition]
            conditions.append(dict(run=key, condition=condition, planned=5, attempted=len(values),
                statuses=dict(Counter(c['status'] for c in values)),
                states=dict(Counter(c['label'] for c in values if c['status'] == 'valid'))))
        for language in sorted({x['language'] for x in outputs}):
            values = [c for x in selected if x['language'] == language for c in x['response']['rows']]
            languages.append(dict(run=key, language=language, attempted_cells=len(values),
                valid=sum(c['status'] == 'valid' for c in values)))
    return dict(status=session['status'], synthetic=False, independent_studies=5,
        observed_studies=len({x['case_index'] for x in outputs}), planned_generations=20,
        recorded_generations=len(outputs), attempted_generations_lower_bound=len(outputs), completed_generations=sum(r['completed'] for r in runs),
        planned_cells=240, recorded_cells=12 * len(outputs), counts_note='Counts reflect durable response records; a killed worker may have an additional unrecorded in-flight call.', runs=runs,
        per_condition=conditions, languages=languages, semantic_accuracy=None,
        semantically_reviewed_cells=0, organizer_comparison='NOT_PERFORMED_BEFORE_BLINDED_REVIEW',
        expansion_allowed=False, model_id=plan['model']['model_id'], revision=plan['model']['revision'])


def build(prepared, inventory, inventory_sha256, output):
    output = rt.private(output)
    if output.exists():
        raise ValueError('Choose a new private output directory')
    plan, session, outputs = load_verified(prepared, inventory, inventory_sha256)
    summary = summarize(plan, session, outputs)
    payload = json.dumps(dict(summary=summary, outputs=outputs), ensure_ascii=False, allow_nan=False)
    for char in '<>&':
        payload = payload.replace(char, '\\u%04x' % ord(char))
    output.mkdir(mode=0o700)
    (output / 'index.html').write_text((Path(__file__).with_name('dashboard.html')).read_text().replace('__PAYLOAD__', payload))
    (output / 'index.html').chmod(0o600)
    rt.write(output / 'technical_summary.json', summary)
    return {k: summary[k] for k in ('status', 'recorded_generations', 'completed_generations', 'semantically_reviewed_cells')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--inventory-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.prepared, a.inventory, a.inventory_sha256, a.output)))
