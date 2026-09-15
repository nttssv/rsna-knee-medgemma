"""Render NOT RUN preview or strictly verified future v2 five-case outputs, locally."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

from core import LABELS, ROOT, code_hashes, config, sha, validate_response
from gates import row_digest

KEYS=['medgemma-1','medgemma-2','qwen-1','qwen-2']


def csv_write(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def check_nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def load(prepared,run_root=None):
    manifest=json.loads((prepared/'manifest.json').read_text())
    if manifest['candidate_code_sha256']!=code_hashes(): raise ValueError('Prepared candidate fingerprints are stale')
    for name,digest in manifest['files'].items():
        if sha(prepared/name)!=digest:raise ValueError('Prepared file changed')
    cases=json.loads((prepared/'review_context.json').read_text())
    inputs=[json.loads(line) for line in (prepared/'inputs.jsonl').read_text().splitlines()]
    if len(cases)!=5 or len(inputs)!=5 or len({c['StudyInstanceUID'] for c in cases})!=5 or any(c['StudyInstanceUID']!=r['StudyInstanceUID'] or c['report']!=r['Report'] for c,r in zip(cases,inputs)):
        raise ValueError('Expected the fixed five inputs')
    for c in cases:c['runs']={}
    if run_root is None:return cases,[],[]
    metrics=[];diagnostics=[]
    for key in KEYS:
        model=key.rsplit('-',1)[0];directory=run_root/key
        m=json.loads((directory/'run_manifest.json').read_text())
        if m.get('synthetic'):
            raise ValueError('Synthetic backend artifacts are not model results')
        if m.get('runtime_adapter_version') == 1:
            for name,digest in m['artifact_sha256'].items():
                if Path(name).name != name or sha(directory/name) != digest:
                    raise ValueError('Runtime artifact hash mismatch')
        if (m['status']!='completed' or m['model_key']!=model or m['model_revision']!=config(model)['revision']
            or m['prepared_manifest_sha256']!=sha(prepared/'manifest.json') or m['candidate_code_sha256']!=code_hashes()):
            raise ValueError('Future v2 run provenance does not match reviewed candidate')
        if sha(directory/'predictions.jsonl')!=m['predictions_sha256']:raise ValueError('Run output hash mismatch')
        records=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        if [r['StudyInstanceUID'] for r in records]!=[c['StudyInstanceUID'] for c in cases]:raise ValueError('Run is not the exact five-case smoke')
        counts=collections.Counter(dict(checks=0,valid=0,positive=0,negative=0,uncertain=0,not_mentioned=0,technical_failure=0,decided=0,correct=0,incorrect=0,syntax_normalized_reports=0))
        seconds=inputs_tokens=outputs_tokens=0
        for c,r in zip(cases,records):
            if r.get('synthetic'):
                raise ValueError('Synthetic backend artifacts are not model results')
            if r['generation_status'] not in ('completed','generation_truncated','timeout','oom','context_overflow','runtime_error'):
                raise ValueError('Unknown generation completion status')
            response=validate_response(r['raw_output'],c['report'],r['generation_status'])
            if r['response']!=response:raise ValueError('Saved acceptance differs from current v2 contract')
            if not all(check_nonnegative(r[k]) for k in ('runtime_seconds','input_tokens','output_tokens')) or not isinstance(r['rendered_prompt'],str):
                raise ValueError('Missing generation measurements')
            c['runs'][key]=r;counts['syntax_normalized_reports']+=response['normalization_applied']
            seconds+=r['runtime_seconds'];inputs_tokens+=r['input_tokens'];outputs_tokens+=r['output_tokens']
            diagnostics.append(dict(run=key,case_index=c['case_index'],generation_status=r['generation_status'],normalization_type=response['normalization_type'],parser_status=response['parser_status']))
            for row in response['rows']:
                counts['checks']+=1
                if row['status']!='valid':counts['technical_failure']+=1;counts[row['status']]+=1;continue
                counts['valid']+=1;counts[row['label']]+=1
                if row['label'] in ('positive','negative'):
                    counts['decided']+=1;counts['correct' if int(row['label']=='positive')==c['gold'][row['condition']] else 'incorrect']+=1
        peak=m.get('peak_gpu_allocated_gib')
        if peak is not None and not check_nonnegative(peak):raise ValueError('Invalid peak allocation')
        metrics.append(dict(run=key,**counts,coverage=counts['decided']/60,correct_yield=counts['correct']/60,mean_seconds=seconds/5,mean_input_tokens=inputs_tokens/5,mean_output_tokens=outputs_tokens/5,peak_gpu_allocated_gib=peak))
    return cases,metrics,diagnostics


def build(prepared,output,run_root=None):
    prepared,output=Path(prepared).resolve(),Path(output).resolve()
    if not output.is_relative_to(prepared) or output.exists():raise ValueError('New review directory must be inside private prepared state')
    cases,metrics,diagnostics=load(prepared,run_root)
    output.mkdir(mode=0o700)
    status='NOT RUN' if run_root is None else 'DEVELOPMENT SMOKE ONLY'
    payload=json.dumps({'status':status,'cases':cases},ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(ROOT/'smoke_review_tools/case_viewer.html').read_text()
    (output/'case_viewer.html').write_text(template.replace('__PAYLOAD__',payload))
    comparison=[];semantic=[]
    for c in cases:
        for condition in LABELS:
            baseline=next(r for r in c['rule'] if r['condition']==condition)
            row=dict(case_index=c['case_index'],condition=condition,organizer=c['gold'][condition],rule_v1=baseline['extracted_label'])
            for key in KEYS:
                if not c['runs']:row[key+'_label']='NOT RUN';continue
                r=next(r for r in c['runs'][key]['response']['rows'] if r['condition']==condition)
                for field in ('label','status','model_evidence','source_span'):row[key+'_'+field]=r[field]
                semantic.append(dict(run=key,case_id=c['StudyInstanceUID'],condition=condition,technical_status=r['status'],semantic_review_status='not_reviewed',reviewer='',reviewed_at='',review_category='',notes='',response_sha256=row_digest(r)))
            comparison.append(row)
    csv_write(output/'smoke_case_comparison.csv',comparison)
    csv_write(output/'smoke_metrics.csv',metrics or [dict(status='NOT RUN')])
    csv_write(output/'smoke_format_diagnostics.csv',diagnostics or [dict(status='NOT RUN')])
    csv_write(output/'smoke_semantic_review.csv',semantic or [dict(status='NOT RUN')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,2,figsize=(13,17),layout='constrained')
    titles=['Technically valid cells /60','Medical states','Technical failure categories','Outer-fence normalization /5','Valid and identical paired cells /60','Binary coverage','Correct binary-label yield','Generation seconds/report','Peak allocated GPU GiB','Mean input/output tokens']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,loc='left')
        if not metrics:ax.text(.5,.5,'NOT RUN',ha='center',va='center',transform=ax.transAxes,color='#66717e',fontsize=19);ax.set_xticks([]);ax.set_yticks([])
    if metrics:
        import numpy as np
        axs=list(axes.flat);x=np.arange(4)
        for index,field in [(0,'valid'),(3,'syntax_normalized_reports'),(5,'coverage'),(6,'correct_yield'),(7,'mean_seconds')]:
            axs[index].bar(x,[r[field] for r in metrics],color='#327ca8');axs[index].set_xticks(x,KEYS,rotation=20)
        for index,fields in [(1,['positive','negative','uncertain','not_mentioned']),(2,sorted({row['status'] for c in cases for r in c['runs'].values() for row in r['response']['rows'] if row['status']!='valid'}))]:
            bottom=np.zeros(4)
            for field in fields:
                vals=[r.get(field,0) for r in metrics];axs[index].bar(x,vals,bottom=bottom,label=field);bottom+=vals
            axs[index].set_xticks(x,KEYS,rotation=20)
            if fields:axs[index].legend(fontsize=7)
        pairs=[]
        for model in ('medgemma','qwen'):
            total=0
            for c in cases:
                aa=c['runs'][model+'-1']['response']['rows'];bb=c['runs'][model+'-2']['response']['rows']
                for a,b in zip(aa,bb):total+=a['status']==b['status']=='valid' and all(a[k]==b[k] for k in ('label','model_evidence','source_span','evidence_start','evidence_end'))
            pairs.append(total)
        axs[4].bar(['MedGemma','Qwen'],pairs,color='#327ca8')
        for i,m in enumerate(metrics):
            if m['peak_gpu_allocated_gib'] is None:axs[8].text(i,0,'NOT MEASURED',rotation=90)
            else:axs[8].bar(i,m['peak_gpu_allocated_gib'],color='#327ca8')
        axs[8].set_xticks(x,KEYS,rotation=20)
        axs[9].bar(x-.15,[m['mean_input_tokens'] for m in metrics],width=.3,label='Input')
        axs[9].bar(x+.15,[m['mean_output_tokens'] for m in metrics],width=.3,label='Output');axs[9].set_xticks(x,KEYS,rotation=20);axs[9].legend()
    fig.suptitle('V2 report extraction • '+status+'\nFive development studies; repeats are not independent validation',fontsize=15)
    fig.savefig(output/'smoke_dashboard.png',dpi=140);fig.savefig(output/'smoke_dashboard.pdf');plt.close(fig)
    (output/'SMOKE_SUMMARY.md').write_text('# V2 smoke review\n\n**'+status+'**\n\nThis package compares original reports and organizer references. Technical acceptance is not semantic correctness. Missing model results are NOT RUN, never zero accuracy.\n\n[Case viewer](case_viewer.html) · [Dashboard](smoke_dashboard.png) · [PDF](smoke_dashboard.pdf)\n')
    for p in output.iterdir():p.chmod(0o600)
    return {'status':status,'studies':5,'measured_model_runs':len(metrics)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=Path)
    a=p.parse_args();print(json.dumps(build(a.prepared,a.output,a.runs)))
