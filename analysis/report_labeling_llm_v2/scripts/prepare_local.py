"""Prepare only the five historical development inputs and unresolved review queue."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from core import REPO, ROOT, LABELS, config, prompt_for, sha, verify_protected, code_hashes


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError(f'Input fingerprint mismatch: {path.name}')
    return path


def prepare(state, output):
    state, output = Path(state).resolve(), Path(output).resolve()
    if state.is_relative_to(REPO) and not state.is_relative_to(REPO / 'state'):
        raise ValueError('Private state in this checkout must be under state/')
    if output == state or not output.is_relative_to(state) or output.exists():
        raise ValueError('Output must be a new directory under private state')
    verify_protected()
    anchors = config('input_anchors')
    parent = state / 'runs/report-labeling-llm-smoke-20260914'
    snapshot = parent / 'remote_snapshot'
    source = checked(state / 'data/train.csv', anchors['source_sha256'])
    split = checked(state / 'runs/report-labeling-20260913-v1/splits.csv', anchors['split_sha256'])
    with split.open() as stream:
        assignments = list(csv.DictReader(stream))
    if len(assignments) != 58 or len({r['StudyInstanceUID'] for r in assignments}) != 58:
        raise ValueError('Expected original58 unique split assignments')
    if [sum(r['split']==p for r in assignments) for p in ('development','validation')] != [40,18]:
        raise ValueError('Original40/18 split required')
    development_ids = {r['StudyInstanceUID'] for r in assignments if r['split']=='development'}
    # Do not open the validation JSONL or load its reports/labels.
    development = load_jsonl(checked(snapshot/'inputs/development.jsonl', anchors['development_sha256']))
    selected = development[:5]
    ids = [r['StudyInstanceUID'] for r in selected]
    if len(set(ids)) != 5 or not set(ids) <= development_ids:
        raise ValueError('Expected five unique development cases')
    parents = {}
    for name, digest in anchors['parent_predictions'].items():
        records = load_jsonl(checked(snapshot/name/'predictions.jsonl', digest))
        if [r['StudyInstanceUID'] for r in records] != ids:
            raise ValueError('Historical model repeats differ from the selected five')
        parents[name] = records
    gold = {}
    with source.open() as stream:
        for row in csv.DictReader(stream):
            uid = row['StudyInstanceUID']
            if uid not in ids:
                continue
            item = selected[ids.index(uid)]
            if row['Report'] != item['Report'] or item['split'] != 'development':
                raise ValueError('Selected development report changed')
            gold[uid] = {c:int(row[c]) for c in LABELS}
    if len(gold)!=5 or any(v not in (0,1) for labels in gold.values() for v in labels.values()):
        raise ValueError('Expected complete binary labels for selected development cases only')
    review_path = checked(parent/'smoke_review_v2/qwen_error_review.csv', anchors['error_review_sha256'])
    with review_path.open() as stream:
        reviewed = list(csv.DictReader(stream))
    baseline_path=checked(state/'runs/report-labeling-20260913-v1/long_extractions.csv', anchors['baseline_extractions_sha256'])
    with baseline_path.open() as stream:
        baseline=[r for r in csv.DictReader(stream) if r['StudyInstanceUID'] in ids]
    if len(baseline)!=60: raise ValueError('Expected60 frozen rule comparison cells')
    context=[dict(case_index=i,StudyInstanceUID=r['StudyInstanceUID'],report=r['Report'],language=r['language'],
                  gold=gold[r['StudyInstanceUID']],rule=[x for x in baseline if x['StudyInstanceUID']==r['StudyInstanceUID']]) for i,r in enumerate(selected)]
    candidates = []
    for row in reviewed:
        case = int(row['case'])-1
        item = selected[case]
        uid, condition = item['StudyInstanceUID'], row['condition']
        prediction = next(r for r in parents['qwen-smoke1'][case]['first_pass'] if r['condition']==condition)
        if row['report'] != item['Report'] or int(row['organizer']) != gold[uid][condition]:
            raise ValueError('Development review does not match original report/reference')
        if (row['qwen_accepted_label'] or None) != prediction['extracted_label']:
            raise ValueError('Review prediction differs from preserved v1')
        candidates.append({'StudyInstanceUID':uid,'condition':condition,'organizer_label':gold[uid][condition],
                           'report_evidence':row['evidence'],'report_text':item['Report'],
                           'rule_v1_prediction':row['rule_v1'],'qwen_v1_prediction':prediction['extracted_label'],
                           'qwen_v1_status':prediction['status'],
                           'medgemma_v1_diagnostic_interpretation':'Rejected v1 output; refer to preserved raw response. Not a benchmark prediction.',
                           'reason_for_review':row['explanation'],'tentative_category':row['tentative_category'],
                           'adjudication_status':'unresolved','adjudicator':'','adjudication_note':''})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.v2-prepare-') as temp:
        tmp = Path(temp)
        (tmp/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        (tmp/'review_context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
        for model in ('medgemma','qwen'):
            (tmp/f'{model}_prompts.jsonl').write_text(''.join(json.dumps({'case_index':i,'prompt':prompt_for(model,r['Report'])},ensure_ascii=False)+'\n' for i,r in enumerate(selected)))
        with (tmp/'v2_adjudication_candidates.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(candidates[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(candidates)
        manifest={'status':'NOT_RUN','studies':5,'partition':'development','model_calls':0,
                  'validation_reports_loaded':0,'validation_labels_evaluated':0,
                  'reference_labels_used_only_for_private_review':True,'source_sha256':anchors['source_sha256'],
                  'split_sha256':anchors['split_sha256'],'v1_predictions_sha256':anchors['parent_predictions'],
                  'files':{p.name:sha(p) for p in tmp.iterdir()},
                  'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
                  'candidate_code_sha256':code_hashes()}
        (tmp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for p in tmp.iterdir(): p.chmod(0o600)
        tmp.rename(output)
    output.chmod(0o700)
    return {'status':'NOT_RUN','development_reports':5,'unresolved_review_candidates':len(candidates),'model_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir',type=Path,default=Path(os.environ.get('RSNA_STATE_DIR',REPO/'state')))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.state_dir,args.output),indent=2))


if __name__=='__main__':
    main()
