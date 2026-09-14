"""Build private human-readable review from verified smoke summaries."""
import argparse,csv,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--template',required=True);a=p.parse_args()
b=Path(a.review);cases=json.loads((b/'private_cases.json').read_text());summaries=json.loads((b/'summary.json').read_text())
keys=['medgemma_1','medgemma_2','qwen_1','qwen_2'];names=['MedGemma 1','MedGemma 2','Qwen 1','Qwen 2']
rows=[next(r for r in summaries if r['model']==key.rsplit('_',1)[0] and r['repeat']==int(key[-1]) and r['variant']=='first_pass') for key in keys]
second=[next(r for r in summaries if r['model']==key.rsplit('_',1)[0] and r['repeat']==int(key[-1]) and r['variant']=='repair_assisted') for key in keys]
assert len(cases)==5 and all(r['checks']==60 for r in rows)
repeat={}
for model in ['medgemma','qwen']:
 all_equal=valid_equal=raw_equal=0
 for c in cases:
  r1,r2=c['models'][model+'_1'],c['models'][model+'_2']
  raw_equal+=r1['attempts'][0]['text']==r2['attempts'][0]['text']
  for x,y in zip(r1['first_pass'],r2['first_pass']):
   same=all(x[k]==y[k] for k in ['condition','extracted_label','evidence_text','status'])
   all_equal+=same;valid_equal+=same and x['status']=='valid' and y['status']=='valid'
 repeat[model]={'identical_including_failures':all_equal,'valid_and_identical':valid_equal,'cells':60,'identical_raw_first_responses':raw_equal,'reports':5,'passed':all_equal==60 and valid_equal==60}
(b/'repeatability_detail.json').write_text(json.dumps(repeat,indent=2))
comparison=[]
for c in cases:
 for variant in ['first_pass','repair_assisted']:
  for condition,truth in c['gold'].items():
   row={'case':c['case'],'variant':variant,'condition':condition,'organizer_label':truth}
   rule=next(r for r in c['models']['rule_v1'] if r['condition']==condition)
   for field in ['extracted_label','evidence_text']:row['rule_v1_'+field]=rule[field]
   for key in keys:
    prediction=next(r for r in c['models'][key][variant] if r['condition']==condition)
    for field in ['extracted_label','status','evidence_text']:row[key+'_'+field]=prediction[field]
   row['repeatability_disagreement']=any(any(row[m+'_1_'+f]!=row[m+'_2_'+f] for f in ['extracted_label','status','evidence_text']) for m in ['medgemma','qwen'])
   row['different_technical_status']=row['medgemma_1_status']!=row['qwen_1_status']
   row['model_state_disagreement']=row['medgemma_1_status']==row['qwen_1_status']=='valid' and row['medgemma_1_extracted_label']!=row['qwen_1_extracted_label']
   comparison.append(row)
with (b/'smoke_case_comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
(b/'smoke_metrics.csv').write_bytes((b/'summary.csv').read_bytes())
payload=json.dumps(cases,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
(b/'case_viewer.html').write_text(Path(a.template).read_text().replace('__PAYLOAD__',payload))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(4,2,figsize=(15,18),layout='constrained');ax=axes.flat
palette=['#3178ac','#ed9d36','#8874bc','#7d8b99','#bc4b51'];x=np.arange(4)
def bars(axis,values,title,ylabel,ceiling=None):
 axis.bar(x,values,color=['#3178ac','#78a7c8','#8874bc','#b0a3d0']);axis.set_xticks(x,names,rotation=12);axis.set_title(title,loc='left',fontweight='bold');axis.set_ylabel(ylabel);axis.set_ylim(0,ceiling if ceiling else max(max(values)*1.22,1));axis.bar_label(axis.containers[0],fmt='%.1f',padding=3)
bars(ax[0],[r.get('valid',0) for r in rows],'1. Valid first-pass cells','Cells / 60',66)
bottom=np.zeros(4)
for state,color in zip(['positive','negative','uncertain','not_mentioned','technical_failure'],palette):
 vals=[r.get(state,0) for r in rows];ax[1].bar(x,vals,bottom=bottom,label=state,color=color);bottom+=vals
ax[1].set_xticks(x,names,rotation=12);ax[1].set_ylabel('Cells / 60');ax[1].set_title('2. First-pass state distribution',loc='left',fontweight='bold');ax[1].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2)
failure_keys=sorted({k for r in rows for k in r if k.startswith('failure_')});bottom=np.zeros(4)
for i,key in enumerate(failure_keys):
 vals=[r.get(key,0) for r in rows];ax[2].bar(x,vals,bottom=bottom,label=key.removeprefix('failure_'),color=palette[i%5]);bottom+=vals
ax[2].set_xticks(x,names,rotation=12);ax[2].set_ylabel('Cells / 60');ax[2].set_ylim(0,66);ax[2].set_title('3. First-pass technical failures',loc='left',fontweight='bold')
if failure_keys:ax[2].legend(fontsize=9)
y=np.arange(2);ax[3].bar(y-.17,[repeat[m]['identical_including_failures'] for m in ['medgemma','qwen']],width=.34,label='Identical, including failures',color='#aeb7c1');ax[3].bar(y+.17,[repeat[m]['valid_and_identical'] for m in ['medgemma','qwen']],width=.34,label='Valid AND identical',color='#3178ac');ax[3].set_xticks(y,['MedGemma','Qwen']);ax[3].set_ylabel('Paired cells / 60');ax[3].set_ylim(0,70);ax[3].set_title('4. Repeatability of label + evidence + status',loc='left',fontweight='bold');ax[3].legend(fontsize=9,loc='lower center');[ax[3].bar_label(c,padding=2) for c in ax[3].containers]
first=np.array([r['first_pass_seconds']/5 for r in rows]);total=np.array([r['generation_seconds']/5 for r in rows]);ax[4].bar(x,first,label='First pass',color='#3178ac');ax[4].bar(x,total-first,bottom=first,label='Extra second pass',color='#ed9d36');ax[4].set_xticks(x,names,rotation=12);ax[4].set_ylabel('Generation seconds / report');ax[4].set_title('5. Mean generation time (loading excluded)',loc='left',fontweight='bold');ax[4].legend(fontsize=9)
bars(ax[5],[r['peak_gpu_allocated_gib'] for r in rows],'6. Peak PyTorch GPU allocation','GiB (not total device usage)')
inputs=[statistics.mean(c['models'][k]['attempts'][0]['input_tokens'] for c in cases) for k in keys];outputs=[r['first_pass_output_tokens']/5 for r in rows];ax[6].bar(x-.17,inputs,width=.34,label='Input',color='#3178ac');ax[6].bar(x+.17,outputs,width=.34,label='Output',color='#ed9d36');ax[6].set_xticks(x,names,rotation=12);ax[6].set_ylabel('Tokens / report');ax[6].set_title('7. Mean first-pass token counts',loc='left',fontweight='bold');ax[6].legend(fontsize=9)
base=summaries[0];rr=[base]+rows;xx=np.arange(5);bottom=np.zeros(5)
for field,color in [('correct','#3178ac'),('incorrect','#bc4b51'),('abstained','#aeb7c1')]:
 vals=[r.get(field,0) if field!='abstained' else 60-r.get('decided',0) for r in rr];ax[7].bar(xx,vals,bottom=bottom,label=field,color=color);bottom+=vals
ax[7].set_xticks(xx,['Rule v1']+names,rotation=18);ax[7].set_ylabel('Checks / 60');ax[7].set_title('8. Descriptive organizer-label comparison',loc='left',fontweight='bold');ax[7].legend(fontsize=9)
fig.suptitle('Five-report development smoke • measured outputs\nTechnical feasibility only; repeated runs are not independent validation',fontsize=18,fontweight='bold')
fig.savefig(b/'smoke_dashboard.png',dpi=160);fig.savefig(b/'smoke_dashboard.pdf');plt.close(fig)
lines=['# Five-report smoke summary','','**OBJECTIVE:** Test reproducible, technically valid report-to-label extraction. This is not an accuracy-validation study and cannot authorize bulk labeling.','','**INPUT:** Exactly five development reports, same cases for both models, two runs each. Prompts contain report text, instructions and condition definitions; no MRI images or organizer labels.','','**OUTPUT:** Twelve condition states and verbatim evidence, or separate technical failures. Raw first and second responses, exact rendered inputs, timing and token counts are preserved.','','## Measured first-pass results','','| Run | Valid / 60 | Correct / 60 | Incorrect / 60 | Technical failures / 60 | Second passes | Mean first-pass seconds/report | Peak allocation GiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
for n,r in zip(names,rows):lines.append(f"| {n} | {r.get('valid',0)} | {r.get('correct',0)} | {r.get('incorrect',0)} | {r.get('technical_failure',0)} | {r['repair_attempts']} | {r['first_pass_seconds']/5:.1f} | {r['peak_gpu_allocated_gib']:.2f} |")
lines+=['','## Repeatability','']
for m,r in repeat.items():lines.append(f"- {m}: {r['identical_including_failures']}/60 identical including failures; {r['valid_and_identical']}/60 valid and identical. Gate passed: {r['passed']}. Raw first responses identical: {r['identical_raw_first_responses']}/5.")
lines+=['','## Interpretation','','Reference comparisons are descriptive checks on five development cases, not a ranking of validated clinical accuracy. Parsing failures reject the whole response, so low correct yield can reflect formatting failures. No Markdown fences were removed, no text was silently extracted from prose, and no failure or not_mentioned state was converted to negative.','','The accepted output still needs semantic review: an exact quotation only establishes that the quoted words exist. It does not prove they support the target diagnosis or severity.','','The one permitted second pass is a new inference that may change content and adds runtime. See smoke_metrics.csv for separate secondary results. Mean first-pass timing excludes loading, downloads and setup; peak memory is PyTorch allocated memory, not total device memory.','','Open [case_viewer.html](case_viewer.html) for reports, evidence, organizer comparisons, and exact raw inputs/outputs. See [smoke_dashboard.png](smoke_dashboard.png) and [smoke_case_comparison.csv](smoke_case_comparison.csv).','','Keep this package private pending review. No validation inference, full 40-study development inference, MRI training or bulk labeling was performed by this smoke stage.']
(b/'SMOKE_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('Private case viewer, metrics, comparison CSV, summary and dashboard created.')
