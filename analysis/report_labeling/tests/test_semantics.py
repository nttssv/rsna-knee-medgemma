import sys
import json
from pathlib import Path
import pytest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from extractor import Extractor
from evaluation import condition_metrics,wilson

@pytest.fixture
def extractor():return Extractor(ROOT/'config/rules.json')

@pytest.mark.parametrize('report,condition,expected',[
 ('Findings: ACL intact.','ACL','negative'),
 ('Findings: Complete ACL tear.','ACL','positive'),
 ('Findings: Suspected complete ACL tear.','ACL','uncertain'),
 ('Findings: Low-grade partial tear of the MCL.','MCL','negative'),
 ('Findings: Complete tear of the MCL.','MCL','positive'),
 ('Findings: Normal PCL.','ACL','not_mentioned'),
 ('Findings: Small joint effusion.','Effusion','negative'),
 ('Findings: Moderate joint effusion.','Effusion','positive'),
 ('Findings: Joint effusion.','Effusion','uncertain'),
 ('Findings: Mild to moderate joint effusion.','Effusion','uncertain'),
 ('Findings: No Baker cyst.',"Baker's",'negative'),
 ('Findings: Baker cyst measuring 3 x 2 cm.',"Baker's",'uncertain'),
 ('Findings: Medial meniscus is not torn.','Medial Meniscus','negative'),
 ('Findings: Lateral meniscal tear.','Medial Meniscus','not_mentioned'),
 ('Findings: Full-thickness cartilage loss in the medial compartment, measuring 2 x 1 cm.','Medial OA','positive'),
 ('Findings: High-grade medial compartment cartilage loss.','Medial OA','uncertain'),
 ('Findings: Bone marrow edema.','Contusion','uncertain'),
 ('Findings: No acute fracture or bone contusion.','Contusion','negative'),
 ('Clinical history: Suspected medial meniscus tear. Findings: Medial meniscus intact.','Medial Meniscus','negative'),
 ('Findings: Previous partial medial meniscectomy.','Medial Meniscus','uncertain'),
 ('Findings: No acute fracture. Acute tibial fracture.','Fracture','uncertain'),
 ('[BILATERAL NOTE: two reports filed under this study] Findings: ACL intact.','ACL','uncertain'),
 ('Hallazgos: Menisco medial sin signos de rotura.','Medial Meniscus','negative'),
 ('Bulgular: Ön çapraz bağ komplet yırtık.','ACL','positive'),
 ('Findings: Patellar tendon is normal.','PF OA','not_mentioned'),
])
def test_statement_semantics(extractor,report,condition,expected):
    assert extractor.extract(report)[condition]['state']==expected

def test_abstentions_not_silently_negative():
    d=pd.DataFrame(dict(organizer_label=[1,0,1,0],extracted_label=['positive','negative','uncertain','not_mentioned'],confidence=[.95,.95,0,0]))
    m=condition_metrics(d)
    assert m['TP']==1 and m['TN']==1 and m['FP']==0 and m['FN']==0
    assert m['coverage']==.5 and m['accuracy']==1 and m['all_study_correct_label_yield']==.5
    assert m['abstained_reference_positive']==1 and m['not_mentioned_gold_negative']==1

def test_undefined_metrics_and_intervals():
    d=pd.DataFrame(dict(organizer_label=[1,0],extracted_label=['uncertain','not_mentioned'],confidence=[0,0]))
    m=condition_metrics(d)
    assert m['accuracy'] is None and m['precision_PPV'] is None and m['cohens_kappa'] is None
    assert wilson(0,0)==(None,None)

def test_output_serialization_and_evidence(extractor):
    report='Findings: Complete ACL tear. No fracture.'
    out=extractor.extract(report)
    json.dumps(out)
    for x in out.values():
        for c in x['candidates']:
            assert report[c['evidence_start']:c['evidence_end']]==c['evidence_text']

def test_anatomic_header_does_not_leak(extractor):
    report='Findings:\nMedial collateral ligament: Complete tear.\nLateral collateral ligament complex: Normal.'
    assert extractor.extract(report)['MCL']['state']=='positive'

def test_measurement_punctuation_preserves_anatomy(extractor):
    report='Findings: Full thickness cartilage defect, 1.2x1.5cm. at lateral trochlea.'
    assert extractor.extract(report)['PF OA']['state']=='positive'

def test_severity_does_not_cross_conditions(extractor):
    r=extractor.extract('Findings: Large Baker cyst and small joint effusion.')
    assert r["Baker's"]['state']=='positive' and r['Effusion']['state']=='negative'
    r=extractor.extract('Findings: Complete ACL tear and low-grade MCL sprain.')
    assert r['ACL']['state']=='positive' and r['MCL']['state']=='negative'

def test_negation_ends_at_independent_finding(extractor):
    r=extractor.extract('Findings: No medial meniscal tear but a lateral meniscal tear.')
    assert r['Medial Meniscus']['state']=='negative' and r['Lateral Meniscus']['state']=='positive'

def test_shared_normal_list_kept(extractor):
    r=extractor.extract('Findings: ACL and MCL intact.')
    assert r['ACL']['state']=='negative' and r['MCL']['state']=='negative'
