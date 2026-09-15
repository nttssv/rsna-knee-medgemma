"""Blinded report-entailment collection, followed by separate organizer comparison."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets

import runtime_v3 as rt


def proposal(response,condition):
    """Retain a structurally readable proposal even when its evidence failed validation."""
    try:
        obj=json.loads(response['normalized_output'],object_pairs_hook=rt.core.no_duplicates)
        value=obj.get(condition)
        return value if isinstance(value,dict) else None
    except (ValueError,AttributeError):return None


def coded_case(code,report,record,prediction):
    response=prediction['response']
    return dict(review_id=code,report=report,
        output_sha256=hashlib.sha256(record['generation']['raw_output'].encode()).hexdigest(),
        conditions=[dict(condition=row['condition'],technical_status=row['status'],
            accepted_label=row['label'],evidence=row['model_evidence'],proposal=proposal(response,row['condition']))
            for row in response['rows']])


def make_bundle(prepared,output,allow_synthetic=False):
    plan,inputs,runs=rt.verify_session(prepared,allow_synthetic)
    output=rt.private(output);output.mkdir(mode=0o700)
    blinded=output/'blinded';blinded.mkdir(mode=0o700)
    operator=output/'operator';operator.mkdir(mode=0o700)
    cases=[];mapping=[]
    for key in rt.ORDER:
        _,raw,parsed=runs[key]
        for i,(record,pred) in enumerate(zip(raw,parsed)):
            code=secrets.token_hex(12)
            cases.append(coded_case(code,inputs[i]['Report'],record,pred))
            mapping.append(dict(review_id=code,key=key,case_index=i,raw_record_sha256=rt.digest(record)))
    secrets.SystemRandom().shuffle(cases)
    rt.write(blinded/'cases.json',dict(synthetic=plan['synthetic'],cases=cases))
    rows=[dict(review_id=c['review_id'],output_sha256=c['output_sha256'],condition=v['condition'],
        reviewer='',reviewed_at='',categories=[],notes='') for c in cases for v in c['conditions']]
    rt.write(blinded/'review_template.json',rows)
    definitions=(rt.V3/'prompts/target_definitions.txt').read_text()
    payload=dict(synthetic=plan['synthetic'],cases=cases,rows=rows,definitions=definitions,
        categories=[c for c in rt.candidate.REVIEW_CATEGORIES if c!='reference_discordance'])
    template=(rt.ROOT/'scripts/reviewer.html').read_text()
    (blinded/'index.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
    (blinded/'index.html').chmod(0o600)
    rt.write(operator/'mapping.json',dict(synthetic=plan['synthetic'],plan_sha256=rt.sha(Path(prepared)/'plan.json'),
        session_sha256=rt.sha(Path(prepared)/'session/result.json'),mapping=mapping,
        blinded_files={p.name:rt.sha(p) for p in blinded.iterdir()}))
    return dict(status='BLANK_REVIEW_READY',synthetic=plan['synthetic'],outputs=20,unreviewed_cells=240,
        model_calls=0,serve_only='blinded/',expansion_allowed=False)


def verify_bundle(prepared,bundle,allow_synthetic=False):
    plan,inputs,runs=rt.verify_session(prepared,allow_synthetic)
    bundle=rt.private(bundle);mapping=rt.read(bundle/'operator/mapping.json')
    if (mapping['synthetic']!=plan['synthetic'] or mapping['plan_sha256']!=rt.sha(Path(prepared)/'plan.json') or
        mapping['session_sha256']!=rt.sha(Path(prepared)/'session/result.json')):
        raise ValueError('Review belongs to another experiment/session')
    if set(mapping['blinded_files'])!={'cases.json','review_template.json','index.html'}:raise ValueError('Wrong blinded files')
    for name,expected in mapping['blinded_files'].items():
        if rt.sha(bundle/'blinded'/name)!=expected:raise ValueError('Blinded source changed')
    coded=rt.read(bundle/'blinded/cases.json')
    if coded['synthetic']!=plan['synthetic']:raise ValueError('Synthetic marker changed')
    expected={}
    if len(mapping['mapping'])!=20 or len(coded['cases'])!=20:raise ValueError('Incomplete review mapping')
    pairs=set();codes=set()
    for m in mapping['mapping']:
        key,i,code=m['key'],m['case_index'],m['review_id']
        if key not in rt.ORDER or type(i) is not int or not 0<=i<5 or (key,i) in pairs or code in codes:
            raise ValueError('Duplicate or invalid review mapping')
        pairs.add((key,i));codes.add(code)
        record,pred=runs[key][1][i],runs[key][2][i]
        if m['raw_record_sha256']!=rt.digest(record):raise ValueError('Stale output binding')
        expected[code]=coded_case(code,inputs[i]['Report'],record,pred)
    actual={c['review_id']:c for c in coded['cases']}
    if len(actual)!=20 or actual!=expected:raise ValueError('Blinded cases differ from original outputs')
    return plan,inputs,runs,mapping,expected


def validate_review(prepared,bundle,review_path,allow_synthetic=False):
    plan,inputs,runs,mapping,cases=verify_bundle(prepared,bundle,allow_synthetic)
    rows=rt.read(review_path)
    if not isinstance(rows,list) or len(rows)!=240:raise ValueError('Exactly 240 reviewed cells required')
    seen=set();identities=set()
    allowed=set(rt.candidate.REVIEW_CATEGORIES)-{'reference_discordance'}
    for row in rows:
        if set(row)!={'review_id','output_sha256','condition','reviewer','reviewed_at','categories','notes'}:
            raise ValueError('Unexpected review fields')
        code,condition=row['review_id'],row['condition'];pair=(code,condition)
        if code not in cases or condition not in rt.core.LABELS or pair in seen or row['output_sha256']!=cases[code]['output_sha256']:
            raise ValueError('Duplicate, missing, swapped or stale review')
        seen.add(pair)
        if not isinstance(row['reviewer'],str) or not row['reviewer'].strip():raise ValueError('Reviewer required')
        timestamp=datetime.fromisoformat(row['reviewed_at'])
        if timestamp.tzinfo is None:raise ValueError('Timezone-aware review time required')
        categories=row['categories']
        if not isinstance(categories,list) or not categories or any(not isinstance(c,str) for c in categories):
            raise ValueError('Explicit review categories required')
        if len(set(categories))!=len(categories) or not set(categories)<=allowed:
            raise ValueError('Invalid, repeated, or premature reference category')
        if 'no_issue_identified' in categories and len(categories)!=1:raise ValueError('No-issue is exclusive')
        status=next(c['technical_status'] for c in cases[code]['conditions'] if c['condition']==condition)
        if status!='valid' and categories==['no_issue_identified']:
            raise ValueError('Technically failed cells cannot be approved')
        if not isinstance(row['notes'],str) or (categories!=['no_issue_identified'] and not row['notes'].strip()):
            raise ValueError('Explain each issue or insufficient-information judgment')
        identities.add(row['reviewer'].strip())
    return dict(plan=plan,inputs=inputs,runs=runs,mapping=mapping,cases=cases,rows=rows,
                reviewers=sorted(identities),review_sha256=rt.sha(review_path))


def technical_summary(prepared,allow_synthetic=False):
    plan,inputs,runs=rt.verify_session(prepared,allow_synthetic)
    results=[]
    for key in rt.ORDER:
        _,raw,parsed=runs[key]
        for condition in rt.core.LABELS:
            cells=[next(c for c in p['response']['rows'] if c['condition']==condition) for p in parsed]
            results.append(dict(run=key,condition=condition,planned=5,attempted=5,
                statuses=dict(Counter(c['status'] for c in cells)),states=dict(Counter(c['label'] for c in cells if c['status']=='valid')),
                binary_decisions=sum(c['label'] in ('positive','negative') for c in cells),semantically_reviewed=0))
    repeat=[]
    for arm in ('control','candidate'):
        first=runs[arm+'-1'][2];second=runs[arm+'-2'][2]
        for i,(a,b) in enumerate(zip(first,second)):
            fields=('label','model_evidence','evidence_start','evidence_end','status')
            differences=[x['condition'] for x,y in zip(a['response']['rows'],b['response']['rows']) if any(x[f]!=y[f] for f in fields)]
            repeat.append(dict(arm=arm,case_index=i,condition_differences=differences))
    return dict(status='SYNTHETIC_SOFTWARE_ONLY' if plan['synthetic'] else 'TECHNICAL_ONLY',
        synthetic=plan['synthetic'],independent_studies=5,planned_generations=20,planned_cells=240,
        condition_results=results,repeat_consistency=repeat,semantic_accuracy=None,expansion_allowed=False)


def finalize(prepared,bundle,review_path,output,allow_synthetic=False):
    data=validate_review(prepared,bundle,review_path,allow_synthetic)
    output=rt.private(output);output.mkdir(mode=0o700)
    reviews={(r['review_id'],r['condition']):r for r in data['rows']}
    # Organizer references are opened only after all first-stage reviews validate.
    context=None if data['plan']['synthetic'] else rt.read(Path(prepared)/'review_context.json')
    if context is not None:
        if len(context)!=5:raise ValueError('Expected five anchored reference cases')
        for i,(c,source) in enumerate(zip(context,data['inputs'])):
            if c['case_index']!=i or c['StudyInstanceUID']!=source['StudyInstanceUID'] or c['report']!=source['Report'] or set(c['gold'])!=set(rt.core.LABELS) or any(type(v) is not int or v not in (0,1) for v in c['gold'].values()):
                raise ValueError('Organizer comparison is not joined to exact source study')
    comparisons=[];counts={}
    for m in data['mapping']['mapping']:
        key,i,code=m['key'],m['case_index'],m['review_id']
        for cell in data['cases'][code]['conditions']:
            condition=cell['condition'];r=reviews[code,condition];label=cell['accepted_label']
            binary=cell['technical_status']=='valid' and label in ('positive','negative')
            gold=None if context is None else context[i]['gold'][condition]
            row=dict(review_id=code,run=key,case_index=i,condition=condition,accepted_label=label,
                organizer=gold,agreement=None if not binary or gold is None else int(label=='positive')==gold,
                entailment_categories=r['categories'],reference_review_categories=[],reference_review_notes='')
            comparisons.append(row)
            c=counts.setdefault((key,condition),Counter(planned=5))
            c.update(reviewed=1,technically_accepted=int(cell['technical_status']=='valid'),binary_decisions=int(binary),
                reviewer_no_issue_yield=int(cell['technical_status']=='valid' and r['categories']==['no_issue_identified']),
                reviewer_no_issue_binary_yield=int(binary and r['categories']==['no_issue_identified']),
                organizer_comparisons=int(row['agreement'] is not None),organizer_matches=int(row['agreement'] is True))
            c.update(r['categories'])
    rt.write(output/'organizer_comparison.json',dict(synthetic=data['plan']['synthetic'],review_sha256=data['review_sha256'],
        note='Stage two: numeric disagreement is not adjudication of report or organizer correctness.',rows=comparisons))
    rt.write(output/'review_results.json',dict(synthetic=data['plan']['synthetic'],review_sha256=data['review_sha256'],
        reviewers=data['reviewers'],reviewer_credentials_independently_verified=False,
        condition_results=[dict(run=k,condition=c,**v) for (k,c),v in counts.items()],expansion_allowed=False))
    return dict(status='SYNTHETIC_REVIEW_TEST' if data['plan']['synthetic'] else 'REVIEW_COLLECTED_NOT_CLINICALLY_CERTIFIED',
        reviewed_cells=240,expansion_allowed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['bundle','summary','finalize'])
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path);p.add_argument('--review',type=Path)
    p.add_argument('--allow-synthetic-demo',action='store_true')
    a=p.parse_args()
    if a.action=='bundle':result=make_bundle(a.prepared,a.output,a.allow_synthetic_demo)
    elif a.action=='summary':
        result=technical_summary(a.prepared,a.allow_synthetic_demo);rt.write(rt.private(a.output),result)
        result={k:v for k,v in result.items() if k not in ('condition_results','repeat_consistency')}
    else:result=finalize(a.prepared,a.bundle,a.review,a.output,a.allow_synthetic_demo)
    print(json.dumps(result))


if __name__=='__main__':main()
