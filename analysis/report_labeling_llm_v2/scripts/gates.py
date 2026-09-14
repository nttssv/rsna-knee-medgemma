"""Fail-closed engineering review gate; never provisions or authorizes compute."""
from datetime import datetime
import hashlib
import json
from core import LABELS, code_hashes


def row_digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def assess_smoke(first, repeat, expected_ids, semantic_review=None):
    reasons=[]
    expected_ids=list(expected_ids)
    if len(expected_ids)!=5 or len(set(expected_ids))!=5:
        reasons.append('Expected exactly five fixed unique development IDs')
    for name, records in [('first',first),('repeat',repeat)]:
        if len(records)!=5 or [r.get('case_id') for r in records]!=expected_ids:
            reasons.append(f'{name}: fixed case identity/order mismatch')
        for record in records:
            response=record.get('response',{})
            if response.get('generation_status')!='completed':reasons.append(f'{name}: noncompleted generation')
            rows=response.get('rows',[])
            if [r.get('condition') for r in rows]!=list(LABELS):reasons.append(f'{name}: incomplete condition set')
            if len(rows)!=12 or any(r.get('status')!='valid' for r in rows):reasons.append(f'{name}: technical failure')
    signature=lambda records:[[(r.get('condition'),r.get('label'),r.get('model_evidence'),r.get('source_span'),r.get('evidence_start'),r.get('evidence_end'),r.get('status')) for r in record.get('response',{}).get('rows',[])] for record in records]
    if signature(first)!=signature(repeat):reasons.append('Label/evidence/status repeatability mismatch')
    review=semantic_review or {}
    if review.get('candidate_code_sha256')!=code_hashes():
        reasons.append('Human review is not bound to the current experiment fingerprints')
    reviewed=review.get('rows',[])
    expected={(record.get('case_id'),row.get('condition')):row for record in first for row in record.get('response',{}).get('rows',[])}
    by_key={(r.get('case_id'),r.get('condition')):r for r in reviewed}
    if review.get('review_source')!='human' or len(reviewed)!=60 or len(by_key)!=60 or set(by_key)!=set(expected):
        reasons.append('Missing complete human review of the first-run 60 cells')
    else:
        for key,row in expected.items():
            r=by_key[key]
            try:
                dated=datetime.fromisoformat(r.get('reviewed_at','')).tzinfo is not None
            except (TypeError,ValueError):dated=False
            documented=all(isinstance(r.get(k),str) and r[k].strip() for k in ('reviewer','review_category','notes'))
            if (r.get('semantic_review_status')!='acceptable' or r.get('technical_status')!=row['status']
                or r.get('response_sha256')!=row_digest(row) or not dated or not documented):
                reasons.append('Incomplete, stale or non-acceptable human review')
                break
    return {'engineering_gate_passed':not reasons,'reasons':reasons,
            'clinical_validation_established':False,'compute_authorized':False}
