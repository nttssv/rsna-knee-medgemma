"""Verify and summarize the approved five-study smoke; no validation-label evaluation."""
import argparse,collections,csv,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--state-dir');p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
a=p.parse_args();repo=Path(a.repo).resolve();runs=Path(a.runs).resolve();out=Path(a.output).resolve()
sys.path.insert(0,str(repo/'analysis/report_labeling_llm_v1/scripts'))
import benchmark_core as core
state=core.state_dir(a.state_dir);prepared=state/'runs/report-labeling-llm-v1/inputs'
inputs=core.load_inputs(state,prepared,'development')[:5];ids=[r['StudyInstanceUID'] for r in inputs]
assert len(set(ids))==5 and all(r['split']=='development' for r in inputs)
with (state/'data/train.csv').open() as f:
 gold={r['StudyInstanceUID']:{c:int(float(r[c])) for c in core.LABELS} for r in csv.DictReader(f) if r['StudyInstanceUID'] in ids}
assert set(gold)==set(ids)
baseline_file=state/'runs/report-labeling-20260913-v1/long_extractions.csv'
manifest=json.loads((baseline_file.parent/'evaluation_manifest.json').read_text())
assert core.sha(baseline_file)==manifest['output_sha256'][baseline_file.name]
with baseline_file.open() as f: baseline=[r for r in csv.DictReader(f) if r['StudyInstanceUID'] in ids]
assert len(baseline)==60
summaries=[];case_models={uid:{} for uid in ids};condition_rows=[];language_rows=[]
languages={r['StudyInstanceUID']:r['language'] for r in inputs}
def score(rows):
 counts=collections.Counter({k:0 for k in ['checks','valid','positive','negative','uncertain','not_mentioned','semantic_abstention','technical_failure','decided','correct','incorrect']});confusion=collections.Counter({k:0 for k in ['TP','FP','TN','FN']})
 for uid,row in rows:
  status=row.get('status','valid');label=row['extracted_label'];truth=gold[uid][row['condition']]
  counts['checks']+=1
  if status!='valid':counts['technical_failure']+=1;counts['failure_'+status]+=1;continue
  counts['valid']+=1;counts[label]+=1
  if label not in core.BINARY:counts['semantic_abstention']+=1;continue
  pred=int(label=='positive');counts['decided']+=1;counts['correct' if pred==truth else 'incorrect']+=1
  confusion[('T' if pred==truth else 'F')+('P' if pred else 'N')]+=1
 n=counts['checks'];d=counts['decided']
 return dict(counts,**confusion,correct_yield=counts['correct']/n,coverage=d/n,conditional_accuracy=counts['correct']/d if d else None)
base_rows=[(r['StudyInstanceUID'],r) for r in baseline]
summaries.append({'model':'rule_v1','repeat':0,'variant':'first_pass',**score(base_rows)})
for uid in ids:case_models[uid]['rule_v1']=[r for u,r in base_rows if u==uid]
for model in ['medgemma','qwen']:
 for repeat in [1,2]:
  path=runs/f'{model}-smoke{repeat}'
  m,records=core.verify_run(state,path,model,'development',prepared,allow_subset=True)
  assert m['studies_processed']==5
  for variant in ['first_pass','repair_assisted']:
   rows=[(r['StudyInstanceUID'],x) for r in records for x in r[variant]]
   attempts=[x for r in records for x in r['attempts']]
   summaries.append({'model':model,'repeat':repeat,'variant':variant,**score(rows),'repair_attempts':sum(r['repair_attempted'] for r in records),'generation_seconds':sum(x['runtime_seconds'] for x in attempts),'generation_output_tokens':sum(x['output_tokens'] for x in attempts),'first_pass_seconds':sum(r['attempts'][0]['runtime_seconds'] for r in records),'first_pass_output_tokens':sum(r['attempts'][0]['output_tokens'] for r in records),'run_seconds':m['runtime_seconds'],'peak_gpu_allocated_gib':m['peak_gpu_allocated_gib']})
   for condition in core.LABELS:condition_rows.append({'model':model,'repeat':repeat,'variant':variant,'condition':condition,**score([(uid,r) for uid,r in rows if r['condition']==condition])})
   for language in sorted(set(languages.values())):language_rows.append({'model':model,'repeat':repeat,'variant':variant,'language':language,'studies':sum(x==language for x in languages.values()),**score([(uid,r) for uid,r in rows if languages[uid]==language])})
  for r in records:case_models[r['StudyInstanceUID']][f'{model}_{repeat}']=r
for condition in core.LABELS:condition_rows.append({'model':'rule_v1','repeat':0,'variant':'first_pass','condition':condition,**score([(uid,r) for uid,r in base_rows if r['condition']==condition])})
for language in sorted(set(languages.values())):language_rows.append({'model':'rule_v1','repeat':0,'variant':'first_pass','language':language,'studies':sum(x==language for x in languages.values()),**score([(uid,r) for uid,r in base_rows if languages[uid]==language])})
agreement_rows=[]
for variant in ['first_pass','repair_assisted']:
 for members in [('medgemma','qwen'),('rule_v1','qwen'),('rule_v1','medgemma'),('rule_v1','medgemma','qwen')]:
  selected=[];both_decided=0;binary_conflicts=0
  for uid in ids:
   for condition in core.LABELS:
    predictions=[next(x for x in (case_models[uid][m] if m=='rule_v1' else case_models[uid][m+'_1'][variant]) if x['condition']==condition) for m in members]
    binary=all(x.get('status','valid')=='valid' and x['extracted_label'] in core.BINARY for x in predictions)
    both_decided+=binary
    match=binary and len({x['extracted_label'] for x in predictions})==1
    binary_conflicts+=binary and not match
    if match:selected.append((uid,predictions[0]))
  result=score(selected) if selected else {'checks':0,'decided':0,'correct':0,'incorrect':0,'conditional_accuracy':None}
  agreement_rows.append({'members':'+'.join(members),'variant':variant,'studies':5,'total_checks':60,'all_members_binary':both_decided,'binary_conflicts':binary_conflicts,'accepted':len(selected),'correct':result['correct'],'incorrect':result['incorrect'],'coverage':len(selected)/60,'correct_yield':result['correct']/60,'conditional_accuracy':result['conditional_accuracy']})
out.mkdir(parents=True,exist_ok=False)
(out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
for name,rows in [('summary.csv',summaries),('per_condition.csv',condition_rows),('language.csv',language_rows),('agreement.csv',agreement_rows)]:
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
cases=[{'case':i+1,'report':r['Report'],'language':r['language'],'gold':gold[r['StudyInstanceUID']],'models':case_models[r['StudyInstanceUID']]} for i,r in enumerate(inputs)]
(out/'private_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
(out/'analysis_provenance.json').write_text(json.dumps({'scope':'five development studies only','source_sha256':core.sha(state/'data/train.csv'),'split_sha256':core.config('benchmark')['split_sha256'],'script_sha256':core.sha(__file__),'study_count':5,'checks_per_run':60,'validation_evaluated':False},indent=2))
print(json.dumps(summaries,indent=2))
