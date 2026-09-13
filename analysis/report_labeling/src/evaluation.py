"""Metrics with explicit abstention accounting; gold is used here, not by the extractor."""
import math
import pandas as pd

def ratio(a,b):return float(a/b) if b else None

def wilson(success,n):
    if not n:return None,None
    z=1.959963984540054;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-delta),min(1,center+delta)

def condition_metrics(d,threshold=0.0):
    y=d.organizer_label.astype(int)
    decided=d.extracted_label.isin(['positive','negative']) & d.confidence.ge(threshold)
    p=d.extracted_label.eq('positive')
    tp=int((decided & p & y.eq(1)).sum());fp=int((decided & p & y.eq(0)).sum())
    tn=int((decided & ~p & y.eq(0)).sum());fn=int((decided & ~p & y.eq(1)).sum())
    n=int(decided.sum());total=len(d)
    sens=ratio(tp,tp+fn);spec=ratio(tn,tn+fp);ppv=ratio(tp,tp+fp);npv=ratio(tn,tn+fn)
    accuracy=ratio(tp+tn,n)
    expected=ratio((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp),n*n)
    meaningful=len(set(y[decided]))==2 and len(set(p[decided]))==2
    kappa=ratio(accuracy-expected,1-expected) if meaningful and expected is not None else None
    nm=d.extracted_label.eq('not_mentioned');nmneg=int((nm&y.eq(0)).sum());nmpos=int((nm&y.eq(1)).sum())
    r=dict(studies=total,decided=n,coverage=ratio(n,total),TP=tp,FP=fp,TN=tn,FN=fn,
        sensitivity_recall=sens,specificity=spec,precision_PPV=ppv,NPV=npv,F1=ratio(2*tp,2*tp+fp+fn),
        accuracy=accuracy,balanced_accuracy=(sens+spec)/2 if sens is not None and spec is not None else None,
        cohens_kappa=kappa,kappa_meaningful=meaningful,organizer_positive=int(y.sum()),organizer_negative=int(y.eq(0).sum()),
        prevalence=float(y.mean()),decided_prevalence=ratio(tp+fn,n),uncertain=int(d.extracted_label.eq('uncertain').sum()),
        not_mentioned=int(nm.sum()),below_confidence_threshold=int((d.extracted_label.isin(['positive','negative'])&~decided).sum()),
        abstained_reference_positive=int((~decided&y.eq(1)).sum()),abstained_reference_negative=int((~decided&y.eq(0)).sum()),
        all_study_correct_label_yield=ratio(tp+tn,total),all_positive_capture=ratio(tp,int(y.sum())),
        all_negative_capture=ratio(tn,int(y.eq(0).sum())),not_mentioned_gold_positive=nmpos,not_mentioned_gold_negative=nmneg,
        not_mentioned_negative_agreement=ratio(nmneg,int(nm.sum())),confidence_threshold=threshold)
    for key,succ,den in [('sensitivity_recall',tp,tp+fn),('specificity',tn,tn+fp),('precision_PPV',tp,tp+fp),('NPV',tn,tn+fn),('coverage',n,total),('not_mentioned_negative_agreement',nmneg,int(nm.sum()))]:
        r[key+'_ci_low'],r[key+'_ci_high']=wilson(succ,den)
    return r

def metrics_table(long,threshold=0):
    return pd.DataFrame([dict(condition=c,**condition_metrics(d,threshold)) for c,d in long.groupby('condition',sort=False)])

def macro_table(metrics):
    fields=['coverage','sensitivity_recall','specificity','precision_PPV','NPV','F1','accuracy','balanced_accuracy','cohens_kappa','prevalence','all_study_correct_label_yield','all_positive_capture','all_negative_capture']
    return pd.DataFrame([dict(metric=f,macro_average=float(metrics[f].mean()) if metrics[f].notna().any() else None,conditions_with_defined_metric=int(metrics[f].notna().sum()),conditions_total=len(metrics)) for f in fields])

def error_reason(r):
    rule=r.rule_id
    if 'combined_reports' in rule or 'contradictory' in rule:return 'anatomy_mismatch','Combined examinations or conflicting statements; manual anatomy/laterality review required.'
    if r.extracted_label=='not_mentioned':return 'report_omission_or_synonym_miss','No supported mention matched. A human must distinguish actual omission from missing terminology.'
    if 'postoperative' in rule or 'remote' in rule or 'temporality' in rule:return 'historical_postoperative_finding','Historical/postoperative status or timing may not match the organizer target.'
    if 'uncertain_statement' in rule:return 'uncertainty','The report hedges the diagnosis; the extractor preserved uncertainty.'
    if any(x in rule for x in ['grade','threshold','size','extent','cartilage_depth']):return 'severity_mismatch','Reported severity is missing, borderline, or differs from the organizer binary reference.'
    if 'negation' in rule or 'normal' in rule:return 'negation_failure_or_reference_discordance','Explicit negative/normal extraction disagrees; inspect negation scope, anatomy and report/reference consistency.'
    if r.extracted_label=='uncertain':return 'uncertainty','Insufficient explicit evidence to map the report finding to the target definition.'
    return 'organizer_label_ambiguity_or_extraction_scope','Affirmative report extraction disagrees; anatomy, scope, report fidelity or reference interpretation needs review.'

def disagreement_table(long,reports):
    rows=[]
    for _,r in long.iterrows():
        pred={'positive':1,'negative':0}.get(r.extracted_label)
        if pred is not None and pred==r.organizer_label:continue
        category,reason=error_reason(r);raw=reports[r.StudyInstanceUID]
        if r.evidence_text:
            start=raw.find(r.evidence_text);snippet=raw[max(0,start-100):start+len(r.evidence_text)+100] if start>=0 else r.evidence_text
        else:snippet=raw[:600]
        rows.append(dict(StudyInstanceUID=r.StudyInstanceUID,case_id=r.case_id,split=r.split,condition=r.condition,
            organizer_label=int(r.organizer_label),extracted_label=r.extracted_label,confidence=r.confidence,evidence_text=r.evidence_text,
            report_snippet=snippet,error_category=category,likely_reason=reason,review_status='provisional_rule_based_triage_not_clinical_adjudication',
            disagreement_type='abstention' if pred is None else 'binary_disagreement',rule_id=r.rule_id))
    return pd.DataFrame(rows)

def yield_table(high):
    rows=[]
    for _,r in high.iterrows():
        lo,hi=wilson(int(r.decided),int(r.studies))
        passed=bool(r.TP+r.FP>=5 and r.TN+r.FN>=5 and r.precision_PPV_ci_low>=0.8 and r.NPV_ci_low>=0.8 and r.coverage>=0.5)
        rows.append(dict(condition=r.condition,validation_accepted=int(r.decided),validation_studies=int(r.studies),
            estimated_candidate_labels=round(r.coverage*4349),candidate_count_ci_low=round(lo*4349),candidate_count_ci_high=round(hi*4349),
            estimated_positive_candidates=round((r.TP+r.FP)/r.studies*4349),estimated_negative_candidates=round((r.TN+r.FN)/r.studies*4349),
            passes_predeclared_scaling_gate=passed,quality_approved_usable_count='not_established' if not passed else 'pilot_gate_only_requires_external_validation',
            assumption='58-study labeled subset representative of 4349 unlabeled studies; not verified',unlabeled_studies_processed=0))
    return pd.DataFrame(rows)
