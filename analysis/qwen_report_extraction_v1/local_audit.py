"""Audit saved Qwen evidence and prepare prospective prompts; never run a model."""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V1 = REPO / 'analysis/report_labeling_llm_v1'
V2 = REPO / 'analysis/report_labeling_llm_v2'
sys.path.insert(0, str(V1 / 'scripts'))
import benchmark_core as old

spec = importlib.util.spec_from_file_location('qwen_frozen_v2_contract', V2 / 'scripts/core.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, digest):
    require(old.sha(path) == digest, 'Fingerprint mismatch: ' + Path(path).name)
    return path


def history():
    files = read(ROOT / 'protected_history.json')['files']
    for name, digest in files.items():
        checked(REPO / name, digest)
    return len(files)


def prompt(arm, report):
    require(arm in ('control', 'candidate'), 'Unknown arm')
    return ((ROOT / 'prompts' / (arm + '.txt')).read_text() + '\nTARGET DEFINITIONS:\n'
            + (ROOT / 'prompts' / (arm + '_definitions.txt')).read_text()
            + '\nREPORT_JSON_STRING:\n' + json.dumps(report, ensure_ascii=False))


def check_saved(record, source):
    """Reparse with original policy; lexical diagnostics never become predictions."""
    require(record['StudyInstanceUID'] == source['StudyInstanceUID']
            and record['report_sha256'] == source['report_sha256']
            and record['split'] == source['split'] == 'development'
            and record['language'] == source['language'], 'Case binding changed')
    require(not record['repair_attempted'] and len(record['attempts']) == 1,
            'Unexpected secondary generation')
    attempt = record['attempts'][0]
    require(attempt['status'] == 'completed', 'Historical Qwen was not complete')
    rows = old.validate_response(attempt['text'], source['Report'])
    require(rows == record['first_pass'] == record['repair_assisted'], 'Original reparse differs')
    require(old.text_sha(attempt['rendered_prompt']) == attempt['rendered_prompt_sha256'],
            'Rendered prompt hash differs')
    obj = json.loads(attempt['text'], object_pairs_hook=old.no_duplicates)
    failures = []
    for row in rows:
        if row['status'] != 'valid':
            evidence = obj[row['condition']]['evidence_text']
            match = contract.match_evidence(source['Report'], evidence)
            failures.append(dict(condition=row['condition'], original_status=row['status'],
                diagnostic_only=True, accepted_label=None, model_evidence=evidence,
                lexical_match=match))
    return rows, failures


def sensitivity_records(record, report, case_index=0, language='synthetic'):
    """Private status transitions only; never export replacement labels."""
    future = contract.validate_response(record['attempts'][0]['text'], report,
                                        record['attempts'][0]['status'])
    transitions = []
    for prior, diagnostic in zip(record['first_pass'], future['rows']):
        require(prior['condition'] == diagnostic['condition'], 'Parser condition order differs')
        transitions.append(dict(case_index=case_index,condition=prior['condition'],language=language,
            v1_status=prior['status'],v2_status=diagnostic['status'],
            changed=prior['status'] != diagnostic['status']))
        if diagnostic['status'] == 'valid' and diagnostic['source_span']:
            require(report[diagnostic['evidence_start']:diagnostic['evidence_end']] == diagnostic['source_span'],
                    'Diagnostic offset mismatch')
    return transitions


def parser_sensitivity(record, report):
    return dict(Counter(r['v1_status']+' -> '+r['v2_status'] for r in sensitivity_records(record,report)))


def summarize_transitions(records, review_keys):
    """Bind every private transition to the independently verified inherited queue."""
    require(len({(r['case_index'],r['condition']) for r in records}) == len(records), 'Duplicate diagnostic cell')
    for row in records:
        row['review_queue_overlap'] = (row['case_index'],row['condition']) in review_keys
    changed = [r for r in records if r['changed']]
    ambiguity = [r for r in changed if r['v2_status']=='ambiguous_evidence_error']
    return dict(changed_cells=len(changed),
        changed_cells_by_language={lang:sum(r['language']==lang for r in changed)
                                   for lang in sorted({r['language'] for r in records})},
        evidence_failures_becoming_diagnostic_valid=sum(r['v1_status']=='evidence_error' and r['v2_status']=='valid' for r in changed),
        new_ambiguity_flags=len(ambiguity),
        new_ambiguity_overlap_existing_review_queue=sum(r['review_queue_overlap'] for r in ambiguity))


def verify_inputs(state):
    anchors = contract.config('input_anchors')
    parent = state / 'runs/report-labeling-llm-smoke-20260914'
    snapshot = parent / 'remote_snapshot'
    # Source CSV is hashed as opaque bytes; no new organizer label evaluation.
    checked(state / 'data/train.csv', anchors['source_sha256'])
    split = checked(state / 'runs/report-labeling-20260913-v1/splits.csv', anchors['split_sha256'])
    with split.open() as f:
        assignments = list(csv.DictReader(f))
    require(len(assignments) == len({r['StudyInstanceUID'] for r in assignments}) == 58
            and Counter(r['split'] for r in assignments) == {'development': 40, 'validation': 18},
            'Original split changed')
    selected = old.read_jsonl(checked(snapshot / 'inputs/development.jsonl', anchors['development_sha256']))[:5]
    ids = [r['StudyInstanceUID'] for r in selected]
    dev = {r['StudyInstanceUID'] for r in assignments if r['split'] == 'development'}
    require(len(set(ids)) == 5 and set(ids) <= dev and all(set(r) == old.INPUT_KEYS
            and old.text_sha(r['Report']) == r['report_sha256'] for r in selected), 'Wrong five inputs')
    return anchors, parent, snapshot, selected


def audit(state, output, cache=None):
    state = old.state_dir(state)
    output = old.private_path(state, output)
    require(not output.exists(), 'Refuse to overwrite prior output')
    protected = history()
    cfg = read(ROOT / 'configs/experiment.json')
    require(cfg['execution_enabled'] is False, 'This package is local only')
    require(cfg['model_id'] == old.config('qwen')['model_id']
            and cfg['revision'] == old.config('qwen')['revision'], 'Checkpoint changed')
    anchors, parent, snapshot, inputs = verify_inputs(state)
    runs, details, provenance = [], [], {}
    for repeat in (1, 2):
        name = f'qwen-smoke{repeat}'
        folder = snapshot / name
        records = old.read_jsonl(checked(folder / 'predictions.jsonl', anchors['parent_predictions'][name]))
        manifest = read(folder / 'run_manifest.json')
        require(manifest['status'] == 'completed' and manifest['model_key'] == 'qwen'
                and manifest['partition'] == 'development' and manifest['studies_processed'] == len(records) == 5
                and manifest['model_config'] == old.config('qwen')
                and manifest['code_sha256'] == old.code_hashes()
                and manifest['output_sha256'] == {'predictions.jsonl': old.sha(folder / 'predictions.jsonl')},
                'Historical run manifest differs')
        counts, sensitivity, failures, transitions = Counter(), Counter(), [], []
        for index, (record, source) in enumerate(zip(records, inputs)):
            require(read(folder / f'case_{index:03d}.json') == record, 'Case sidecar differs')
            rows, found = check_saved(record, source)
            counts.update(r['status'] for r in rows)
            sensitivity.update(parser_sensitivity(record, source['Report']))
            transitions.extend(sensitivity_records(record,source['Report'],index,source['language']))
            for item in found:
                failures.append(dict(case_index=index, **item))
        runs.append(dict(repeat=repeat, studies=5, condition_cells=60, statuses=dict(counts),
            mean_generation_seconds=sum(r['attempts'][0]['runtime_seconds'] for r in records)/5,
            generation_tokens=sum(r['attempts'][0]['output_tokens'] for r in records),
            parser_sensitivity_transitions=dict(sensitivity)))
        details.append(dict(repeat=repeat, failures=failures, records=records,parser_sensitivity=transitions))
        provenance[name] = {p.name: old.sha(p) for p in folder.iterdir() if p.is_file()}
    first, second = [d['records'] for d in details]
    repeats = dict(raw_identical=sum(a['attempts'][0]['text'] == b['attempts'][0]['text'] for a,b in zip(first,second)),
        valid_identical=sum(x == y and x['status'] == 'valid' for a,b in zip(first,second)
                            for x,y in zip(a['first_pass'],b['first_pass'])))
    # Preserve existing analyst queue; it is not new clinical adjudication.
    review_path = checked(parent / 'smoke_review_v2/qwen_error_review.csv', anchors['error_review_sha256'])
    with review_path.open() as f:
        review = list(csv.DictReader(f))
    require(len(review) == 12 and len({(r['case'],r['condition']) for r in review}) == 12, 'Review queue changed')
    for row in review:
        i = int(row['case']) - 1
        require(0 <= i < 5 and row['report'] == inputs[i]['Report'], 'Review source differs')
        pred = next(r for r in first[i]['first_pass'] if r['condition'] == row['condition'])
        require((row['qwen_accepted_label'] or None) == pred['extracted_label']
                and row['status'] == pred['status']
                and row['review_status'] == 'analyst_hypothesis_not_clinically_adjudicated', 'Review status differs')
    review_keys={(int(r['case'])-1,r['condition']) for r in review}
    for run, detail in zip(runs,details):
        run['parser_sensitivity_details'] = summarize_transitions(detail['parser_sensitivity'],review_keys)
    prompts = {arm: [dict(case_index=i, prompt=prompt(arm,r['Report'])) for i,r in enumerate(inputs)]
               for arm in ('control','candidate')}
    require(all(p['prompt'] == old.prompt_for('qwen',r['Report']) for p,r in zip(prompts['control'],inputs)), 'Control differs')
    require(all(p['prompt'] == contract.prompt_for('qwen',r['Report']) for p,r in zip(prompts['candidate'],inputs)), 'Candidate differs')
    tokens, token_details = tokenize(cache, cfg, prompts, details) if cache else ({'status':'NOT_PERFORMED'}, None)
    unique_failures = details[0]['failures']
    summary = dict(status='SAVED_OUTPUT_AUDIT_AND_LOCAL_PREPARATION', model=cfg['model_id'], revision=cfg['revision'],
        runs=runs, repeatability=repeats, languages=dict(Counter(r['language'] for r in inputs)),
        historical_v1_unique_evidence_failures=len(unique_failures),
        unique_whitespace_diagnostic_matches=sum(f['lexical_match'].get('evidence_normalization_method') == 'ascii_whitespace'
                                                and f['lexical_match']['status'] == 'valid' for f in unique_failures),
        diagnostic_matches_are_predictions=False, inherited_review_items=len(review),
        parser_sensitivity_is_new_inference=False,
        inherited_review_categories=dict(Counter(r['tentative_category'] for r in review)),
        semantic_adjudications_completed=0, new_organizer_evaluation=False,
        validation_reports_loaded=0, new_generations=0, model_weights_loaded=False, provider_calls=0,
        protected_files=protected, tokenizer=tokens, source_sha256=old.sha(__file__),
        protocol_sha256=old.sha(ROOT/'PROTOCOL.md'),
        proposal_sha256=old.sha(ROOT/'configs/experiment.json'))
    require(history() == protected, 'History changed during audit')
    verify_inputs(state)
    for name, files in provenance.items():
        for file, digest in files.items(): checked(snapshot/name/file, digest)
    output = old.new_directory(state, output)
    old.write_json(output/'summary.json', summary)
    old.write_json(output/'provenance.json', provenance)
    old.write_json(output/'details.json', dict(inputs=inputs, runs=details, inherited_review=review,
                                              clinical_adjudication=False))
    for arm, rows in prompts.items(): old.write_jsonl(output/(arm+'_prompts.jsonl'), rows)
    if token_details is not None: old.write_json(output/'tokenized.json', token_details)
    old.write_json(output/'receipt.json', dict(files={p.name:old.sha(p) for p in output.iterdir()},
        protected_manifest_sha256=old.sha(ROOT/'protected_history.json'),
        package_files={str(p.relative_to(ROOT)):old.sha(p) for p in ROOT.rglob('*')
                       if p.is_file() and p.suffix in ('.py','.txt','.json','.md') and '__pycache__' not in p.parts
                       and 'aggregate' not in p.parts}))
    return summary


def tokenize(cache, cfg, prompts, details):
    """Existing offline tokenizer only; no output IDs were saved in historical v1."""
    sys.path.insert(0, str(V2/'scripts'))
    from v2_runtime import HFEncoder, check_versions
    versions = check_versions()
    assets = read(V2/'aggregate/tokenizer_preflight.json')['qwen']['assets']
    snapshot = Path(cache)/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
    for asset in assets:
        checked(snapshot/asset['file'], asset['sha256'])
    encoder = HFEncoder('qwen', cache)
    context = read(snapshot/'config.json')['max_position_embeddings']
    encoded, summary = {}, {}
    for arm, rows in prompts.items():
        entries = [encoder.encode(r['prompt']) for r in rows]
        lengths = [e.input_tokens for e in entries]
        require(all(0 < n <= cfg['max_input_tokens'] and n+cfg['max_new_tokens'] <= context for n in lengths), 'Context overflow')
        if arm == 'control':
            for run in details:
                for e,r in zip(entries,run['records']):
                    a=r['attempts'][0]
                    require(e.rendered_prompt == a['rendered_prompt'] and e.input_tokens == a['input_tokens'],
                            'Historical rendered prompt/input count differs')
        encoded[arm] = [dict(rendered_prompt=e.rendered_prompt,input_ids=e.input_ids,input_tokens=e.input_tokens) for e in entries]
        summary[arm] = dict(studies=5,min_input_tokens=min(lengths),max_input_tokens=max(lengths),
                           total_input_tokens=sum(lengths),max_input_plus_output=max(lengths)+cfg['max_new_tokens'])
    for asset in assets: checked(snapshot/asset['file'],asset['sha256'])
    return dict(status='TOKENIZED_OFFLINE_NO_INFERENCE',arms=summary,context_limit=context,
        original_rendered_prompt_and_count_matches=10,original_token_ids_available=False,
        assets=assets,versions=versions,metadata=encoder.metadata), encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',type=Path,default=REPO/'state')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.state_dir,args.output,args.cache),indent=2))
