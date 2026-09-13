"""Develop -> freeze -> one held-out evaluation. No bulk-labeling command is provided."""
import argparse,json,platform,re
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from common import LABELS,fingerprint,report_hash,load_gold,write_json
from extractor import Extractor,detect_language,units
from evaluation import metrics_table,macro_table,disagreement_table,yield_table

def checked_data(source,experiment):
    manifest=json.loads((experiment/'split_manifest.json').read_text())
    assert fingerprint(source)==manifest['source_sha256'],'Source fingerprint changed.'
    assert fingerprint(experiment/'splits.csv')==manifest['split_sha256'],'Split fingerprint changed.'
    d=load_gold(source);split=pd.read_csv(experiment/'splits.csv',dtype=str)
    d=d.merge(split,on='StudyInstanceUID',validate='one_to_one')
    assert len(d)==58 and set(d.split)=={'development','validation'}
    assert (d.Report.map(report_hash)==d.report_sha256).all(),'Report fingerprints differ from original split.'
    return d

def frozen_files(experiment):
    paths=sorted((experiment/'src').glob('*.py'))+sorted((experiment/'config').glob('*.json'))+sorted((experiment/'tests').glob('*.py'))
    paths += [experiment/'PROTOCOL.md',experiment/'splits.csv',experiment/'split_manifest.json',experiment/'requirements.txt']
    return {str(p.relative_to(experiment)):fingerprint(p) for p in paths}

def verify_freeze(experiment,source):
    m=json.loads((experiment/'freeze_manifest.json').read_text())
    assert m['files_sha256']==frozen_files(experiment),'Frozen code/config/split changed; do not silently reuse the holdout.'
    assert m['source_sha256']==fingerprint(source)
    return m

def extract_rows(d,extractor):
    rows=[];wide=[];characteristics=[]
    for _,r in d.iterrows():
        findings=extractor.extract(r.Report)
        base=dict(StudyInstanceUID=r.StudyInstanceUID,case_id=r.case_id,split=r.split,report_sha256=r.report_sha256,
                  extraction_method=extractor.config['method'],extraction_version=extractor.config['version'])
        w=dict(base)
        characteristics.append(dict(**base,language_heuristic=detect_language(r.Report),characters=len(r.Report),
            paragraphs=len(r.Report.split('\n\n')),parsed_units=len(units(r.Report)),combined_reports='bilateral note' in r.Report.lower(),
            structured_headings=bool(re.search(r'(?i)findings:|hallazgos:|bevindingen:|bulgular:',r.Report))))
        for c,x in findings.items():
            rows.append(dict(**base,condition=c,organizer_label=int(r[c]),extracted_label=x['state'],confidence=x['confidence'],
                evidence_text=x['evidence_text'],normalized_finding=x['normalized_finding'],rule_id=x['rule_id'],
                section=x['section'],section_context=x['section_context'],historical_or_postoperative=x['historical_or_postoperative'],
                severity_terms=json.dumps(x['severity_terms']),uncertainty_detected=x['uncertainty_detected'],negation_detected=x['negation_detected'],
                evidence_start=x['evidence_start'],evidence_end=x['evidence_end'],candidates_json=json.dumps(x['candidates'],ensure_ascii=False)))
            for key,value in [('organizer',int(r[c])),('extracted',x['state']),('confidence',x['confidence']),('evidence',x['evidence_text']),('normalized_finding',x['normalized_finding']),('rule_id',x['rule_id']),('section',x['section'])]:w[key+'__'+c]=value
        wide.append(w)
    return pd.DataFrame(rows),pd.DataFrame(wide),pd.DataFrame(characteristics)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['develop','freeze','evaluate'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args();experiment=a.experiment.resolve();d=checked_data(a.source,experiment)
    if a.stage=='freeze':
        target=experiment/'freeze_manifest.json';assert not target.exists(),'Freeze already exists.'
        write_json(target,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),files_sha256=frozen_files(experiment),
            source_sha256=fingerprint(a.source),validation_studies=18,development_studies=40,validation_outcomes_inspected_before_freeze=False,
            python=platform.python_version(),pandas=pd.__version__,method='multilingual_rules',version='1.0.0'))
        print('Extractor, config, evaluation, tests and split frozen. No held-out extraction has run.');return
    assert a.out,'--out is required for develop/evaluate'
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'evaluation_manifest.json').exists(),'Existing result must not be overwritten.'
    if a.stage=='develop':
        assert not (experiment/'freeze_manifest.json').exists(),'Development is closed after freeze.'
        d=d[d.split=='development'].copy()
    else:verify_freeze(experiment,a.source)
    extractor=Extractor(experiment/'config/rules.json')
    long,wide,characteristics=extract_rows(d,extractor)
    assert len(long)==len(d)*12 and wide.StudyInstanceUID.is_unique
    wide.to_csv(out/('gold58_extractions.csv' if a.stage=='evaluate' else 'development_extractions.csv'),index=False)
    long.to_csv(out/'long_extractions.csv',index=False)
    characteristics.to_csv(out/'report_characteristics.csv',index=False)
    for split_name,s in long.groupby('split',sort=False):
        m=metrics_table(s);m.to_csv(out/(split_name+'_metrics.csv'),index=False)
        macro_table(m).to_csv(out/(split_name+'_macro_metrics.csv'),index=False)
        high=metrics_table(s,extractor.config['high_confidence_threshold']);high.to_csv(out/(split_name+'_high_confidence_metrics.csv'),index=False)
        if split_name=='validation':yield_table(high).to_csv(out/'label_yield_estimates.csv',index=False)
    disagreement_table(long,d.set_index('StudyInstanceUID').Report.to_dict()).to_csv(out/'disagreements.csv',index=False)
    assert fingerprint(a.source)==json.loads((experiment/'split_manifest.json').read_text())['source_sha256']
    if a.stage=='evaluate':verify_freeze(experiment,a.source)
    write_json(out/'evaluation_manifest.json',dict(stage=a.stage,completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprint(a.source),
        split_sha256=fingerprint(experiment/'splits.csv'),freeze_sha256=fingerprint(experiment/'freeze_manifest.json') if a.stage=='evaluate' else None,
        studies_processed=len(d),unlabeled_studies_processed=0,source_modified=False,organizer_labels_overwritten=False,image_predictions_used=False,
        llm_used=False,training_started=False,output_sha256={p.name:fingerprint(p) for p in sorted(out.glob('*.csv'))}))
    print('Completed',a.stage,'on',len(d),'gold-standard studies. No unlabeled studies processed.')
    metrics=pd.read_csv(out/('validation_macro_metrics.csv' if a.stage=='evaluate' else 'development_macro_metrics.csv'))
    print(metrics.to_string(index=False))

if __name__=='__main__':main()
