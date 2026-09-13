"""Shared, provider-independent utilities. Never writes source data."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def norm(text):
    text = unicodedata.normalize('NFKD', str(text).lower())
    return ''.join(c for c in text if not unicodedata.combining(c)).replace('ı','i').replace('ß','ss')

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def report_hash(text):
    # Keep the normalization recorded by the original split stable.
    t=unicodedata.normalize('NFKD',str(text).lower())
    t=''.join(c for c in t if not unicodedata.combining(c))
    return hashlib.sha256(' '.join(t.split()).encode()).hexdigest()

def load_gold(path):
    import pandas as pd
    data = pd.read_csv(path, dtype={'StudyInstanceUID':str})
    assert data.StudyInstanceUID.is_unique
    gold = data.loc[data[LABELS].notna().any(axis=1)].copy()
    assert len(data)==4407 and len(gold)==58, 'This experiment is restricted to the audited 58/4407 dataset.'
    assert gold[LABELS].notna().all().all()
    assert gold[LABELS].isin([0,1]).all().all()
    gold[LABELS] = gold[LABELS].astype(int)
    return gold.sort_values('StudyInstanceUID').reset_index(drop=True)

def write_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
