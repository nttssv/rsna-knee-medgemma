"""Create a fixed split using labels only for prevalence balance, never extractor outcomes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common import LABELS,load_gold,fingerprint,report_hash,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260913)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'splits.csv').exists(), 'Existing split must not be overwritten.'
    d=load_gold(a.source)
    hashes=d.Report.map(report_hash)
    assert hashes.nunique()==len(d), 'Duplicate report groups require a grouped split.'
    y=d[LABELS].to_numpy(); target=y.sum(axis=0)*18/58
    rng=np.random.default_rng(a.seed)
    best=None
    for _ in range(50000):
        idx=np.sort(rng.choice(len(d),18,replace=False));counts=y[idx].sum(axis=0)
        score=float(np.sum((counts-target)**2/np.maximum(target,1)))
        score+=float(np.sum((counts<2)|(18-counts<2)))*1000
        if best is None or score<best[0]:best=(score,idx)
    d['split']='development';d.loc[best[1],'split']='validation'
    d['report_sha256']=hashes
    d['case_id']=['G%02d'%(i+1) for i in range(len(d))]
    d[['StudyInstanceUID','case_id','split','report_sha256']].to_csv(a.out/'splits.csv',index=False)
    dev=d[d.split=='development']
    dev.to_csv(a.out/'development_gold.csv',index=False)
    packet=[]
    for _,r in dev.iterrows():
        packet += [r.case_id, 'ORGANIZER POSITIVES: '+', '.join(c for c in LABELS if r[c]==1),r.Report,'']
    (a.out/'development_reports.txt').write_text('\n'.join(packet))
    write_json(a.out/'split_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=a.seed,
        selection='Best prevalence balance among 50000 seeded candidate 18-study subsets; objective uses gold prevalence only, never report content or extraction outcomes.',
        source_sha256=fingerprint(a.source),split_sha256=fingerprint(a.out/'splits.csv'),
        development_studies=40,validation_studies=18,normalized_duplicate_gold_reports=0,
        balance_score=best[0],patient_level_independence_verified=False))
    print('Fixed 40 development / 18 validation studies. Only development text and labels exported for inspection.')

if __name__=='__main__':main()
