"""Interpretable multilingual baseline. Accepts report text only, never gold labels or images."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import re
from common import LABELS,norm

HEADER = re.compile(r'(?i)\b(?:imaging findings|findings|hallazgos|resultados|bevindingen|bulgular|ευρηματα|ευρήματα|impression|impresion|impresión|conclusion|conclusión|conclusie|besluit|sonuc|sonuç|συμπερασμα|συμπέρασμα|imp|clinical history|antecedentes clinicos|antecedentes clínicos|indication|comparison|technique|técnica|tecnica|procedure|tetkik protokolü|diagnostische vraagstelling|klinische inlichtingen|exam type|exam date and time|(?:medial|lateral|patellofemoral|anterior|intercondylar) compartment(?:\s*\([^\n:]*\))?|(?:medial|lateral|patellofemoral) compartment cartilage|(?:medial|lateral) meniscus|anterior cruciate ligament(?:\s*\(ACL\))?|posterior cruciate ligament(?:\s*\(PCL\))?|medial collateral ligament(?:\s*\(MCL\))?|joint effusion|baker cyst|osseous structures|bone marrow signal|bones|joint space|fluid|degenerative changes|muscles(?: and neurovascular structures)?|extensor tendons|soft tissues)\s*:')

@dataclass
class Unit:
    text:str
    section:str
    context:str
    start:int
    end:int

def section_type(header):
    h=norm(header)
    if re.search(r'impress|impres|conclus|besluit|sonuc|συμπερ|^imp\b',h):return 'impression'
    if re.search(r'clinical|antecedent|indication|comparison|vraagstelling|inlichtingen|exam type|exam date',h):return 'history'
    if re.search(r'technique|tecnica|procedure|protokol',h):return 'technique'
    return 'findings'

def units(text):
    """Keep evidence as exact source slices; normalize only for matching."""
    matches=list(HEADER.finditer(text))
    # Other structured field headings end inherited anatomy context too (e.g. LCL after MCL).
    for m in re.finditer(r'(?m)^[ \t]*[A-Za-z][A-Za-z /&()\'-]{0,75}:',text):
        if not any(m.start()<x.end() and x.start()<m.end() for x in matches):matches.append(m)
    matches.sort(key=lambda m:m.start());segments=[]
    last=0;section='findings';context=''
    for m in matches:
        if m.start()>last:segments.append((last,m.start(),section,context))
        h=m.group(); section=section_type(h)
        context=h if section=='findings' and not re.search(r'findings|hallazgos|resultados|bevindingen|bulgular|ευρη',norm(h)) else ''
        last=m.end()
    if last<len(text):segments.append((last,len(text),section,context))
    result=[]
    for start,end,section,context in segments:
        segment=text[start:end]
        # Preserve decimal measurements; split sentence ends, paragraphs and bullet lines.
        boundaries=[0]+[m.end() for m in re.finditer(r'(?<!\d)(?<!mm)(?<!cm)[.!?;](?!\d)|\n\s*\n|\n\s*(?:[*>-]|\d+\.)\s+',segment,re.I)]+[len(segment)]
        for i in range(len(boundaries)-1):
            a,b=boundaries[i:i+2];s=segment[a:b]
            if not s.strip():continue
            left=len(s)-len(s.lstrip());right=len(s.rstrip())
            a+=left;b=boundaries[i]+right
            actual=segment[a:b]
            st=section
            # Short technique-prefixed reports sometimes put all findings after the first sentence.
            if st=='technique' and i>0:st='findings'
            result.append(Unit(actual,st,context,start+a,start+b))
    return result

def detect_language(text):
    t=norm(text)
    patterns={
      'Spanish':r'\b(rodilla|hallazgos|menisco|rotura|derrame|condropatia)\b',
      'Turkish':r'\b(diz|izlenmistir|meniskus\w*|capraz|bag\w*|yirt\w*|bulgular)\b',
      'Dutch':r'\b(knie|bevindingen|besluit|kruisband|kraakbeen\w*|voorkomen)\b',
      'Croatian':r'\b(kostan\w*|hrskavic\w*|krizn\w*|ligament\w*|izljev\w*|menisk\w*|prikaz\w*)\b',
      'German':r'\b(vkb|hkb|gelenkerguss|innenmenisk\w*|aussenmenisk\w*|knorpel\w*|der|und)\b',
      'English':r'\b(the|is|are|tear|normal|findings|knee|cartilage|meniscus)\b'}
    if len(re.findall(r'[α-ω]',t))>30:return 'Greek'
    scores={k:len(re.findall(v,t)) for k,v in patterns.items()}
    return max(scores,key=scores.get) if max(scores.values())>=2 else 'unknown'

class Extractor:
    def __init__(self,config):
        self.config=json.loads(Path(config).read_text())
        self.patterns={k:re.compile(v,re.I) for k,v in self.config.items() if isinstance(v,str) and k in {'uncertainty','history','remote','postoperative','recurrent','negative_before','normal','low','high','complete','low_ligament','tear','cartilage','high_cartilage','broad_cartilage','low_cartilage'}}
        for c in LABELS:
            for key in ['anatomy','context','generic']:
                if key in self.config['conditions'][c]:re.compile(self.config['conditions'][c][key])

    def has(self,name,t):return bool(self.patterns[name].search(t))

    def refine_units(self,initial):
        """Separate coordinated independent findings, retaining shared anatomy lists."""
        out=[]
        def concepts(s):
            t=norm(s)
            return {c for c in LABELS if re.search(self.config['conditions'][c]['anatomy'],t)}
        def predicate(s):
            t=norm(s)
            return any(self.has(k,t) for k in ['tear','normal','low','high','uncertainty']) or bool(re.search(r'\b(no|sin|without|bez)\b',t))
        for u in initial:
            split_at=None
            for m in re.finditer(r'\b(?:and|but|pero|maar|aber|ancak|ali)\b',u.text,re.I):
                left=u.text[:m.start()];right=u.text[m.end():]
                a,b=concepts(left),concepts(right)
                if a and b and a.isdisjoint(b) and predicate(left) and predicate(right):
                    split_at=m;break
            if split_at is None:out.append(u);continue
            m=split_at
            left=u.text[:m.start()].rstrip();raw_right=u.text[m.end():]
            trim=len(raw_right)-len(raw_right.lstrip());right=raw_right.lstrip()
            out.extend(self.refine_units([Unit(left,u.section,u.context,u.start,u.start+len(left)),
                Unit(right,u.section,'',u.start+m.end()+trim,u.end)]))
        return out

    def candidate(self,condition,u):
        if u.section=='history':return None
        t=norm(u.text);ctx=norm(u.context);c=self.config['conditions'][condition];kind=c['kind']
        if u.section=='technique':return None
        combined=(ctx+' '+t).strip()
        anatomy=bool(re.search(c['anatomy'],combined))
        if kind=='meniscus' and not anatomy:
            anatomy=bool(re.search(c.get('context','(?!)'),ctx) and re.search(r'\bmenisc\w*\b',t))
            if not anatomy and re.search(c.get('generic','(?!)'),t):
                # A generic plural normal/no-tear statement covers both menisci.
                anatomy=self.has('normal',t) or bool(re.search(r'no\s+(?:evidence\s+of\s+)?(?:meniscal\s+)?tears?|no\s+fracture',t))
        if kind=='cartilage':
            cartilage=self.has('cartilage',combined)
            generic_normal=bool(re.search(r'\b(?:cartilages?|articular cartilage|kraakbeen|eklem kikirdaklari)\b',t)) and self.has('normal',t) and not re.search(r'\b(?:medial|lateral|patell|trochl)\w*',t)
            global_tf=bool(re.search(r'\b(?:tibiofemoral|femorotibial\w*)\b',t)) and not re.search(r'\b(?:medial|lateral)\b',t) and condition!='PF OA'
            anatomy=cartilage and (anatomy or bool(re.search(c.get('context','(?!)'),ctx)) or generic_normal or global_tf or 'tricompartmental' in t)
        if kind in {'contusion','fracture'} and ctx in {'bones:','osseous structures:'} and self.has('normal',t):
            anatomy=True
        if not anatomy:return None
        anchors=list(re.finditer(c['anatomy'],t))
        if kind in {'ligament','meniscus'}:anchors+=list(self.patterns['tear'].finditer(t))
        if kind=='cartilage':anchors+=list(self.patterns['cartilage'].finditer(t))
        negated=any(self.patterns['negative_before'].search(t[max(0,m.start()-100):m.start()]) for m in anchors)
        negated=negated or bool(re.search(r'^\s*(?:none|absent|negative|normal)\b',t))
        negated=negated or bool(re.search(r'\b(?:not\s+torn|bez\s+(?:znakova|evidentne)\s+rupture|yirtigi\s+eslik\s+etmiyor|yoktur)\b',t))
        uncertain=self.has('uncertainty',t)
        history=self.has('history',t)
        postoperative=self.has('postoperative',t)
        severity=[k for k in ['low','high','complete','low_ligament','high_cartilage','broad_cartilage','low_cartilage'] if self.has(k,t)]
        metadata=dict(evidence_text=u.text,section=u.section,section_context=u.context,evidence_start=u.start,evidence_end=u.end,
                      historical_or_postoperative=history,severity_terms=severity,uncertainty_detected=uncertain,negation_detected=negated)
        def result(state,confidence,reason):
            return dict(state=state,confidence=confidence,normalized_finding=condition+': '+reason,rule_id=kind+'.'+reason,**metadata)
        if uncertain:return result('uncertain',0.35,'uncertain_statement')
        # Explicit negative/normal is distinct from absent mention.
        if negated:return result('negative',0.95,'explicit_negation')
        normal=self.has('normal',t) and not re.search(r'\bnot\s+(?:normal|intact)\b',t)
        tear=self.has('tear',t)
        if kind in {'ligament','meniscus','cartilage','contusion','fracture'} and normal and not tear:
            return result('negative',0.95,'explicit_normal')
        if kind=='ligament':
            if postoperative and not self.has('recurrent',t):return result('uncertain',0.3,'postoperative_status')
            if condition=='MCL' and self.has('remote',t):return result('negative',0.9,'remote_injury_excluded')
            if self.has('complete',t) and tear:return result('positive',0.93,'high_grade_or_complete_tear')
            if self.has('low_ligament',t) and not self.has('complete',t):return result('negative',0.9,'low_grade_or_degeneration_excluded')
            if tear:return result('uncertain',0.4,'tear_grade_unspecified')
            return result('uncertain',0.25,'anatomy_without_definite_status')
        if kind=='meniscus':
            if postoperative and not self.has('recurrent',t):return result('uncertain',0.3,'postoperative_status')
            if tear:return result('positive',0.92,'definite_tear_or_morphology')
            if re.search(r'intrasubstance|intramenisc|degenera|dejenera|εκφυλι',t):return result('negative',0.85,'degeneration_without_reported_tear')
            return result('uncertain',0.25,'anatomy_without_definite_status')
        if kind=='cartilage':
            high=self.has('high_cartilage',t)
            # Numeric size can establish area extent, but not depth by itself.
            dimensions=re.findall(r'(\d+(?:\.\d+)?)\s*(?:x|\*|by)\s*(\d+(?:\.\d+)?)\s*(mm|cm)',t)
            broad=self.has('broad_cartilage',t) or any(max(float(a),float(b))*(10 if unit=='cm' else 1)>=10 for a,b,unit in dimensions)
            if high and broad:return result('positive',0.92,'high_grade_with_sufficient_extent')
            if high:return result('uncertain',0.45,'cartilage_extent_unspecified')
            if self.has('low_cartilage',t):return result('negative',0.85,'below_cartilage_threshold')
            return result('uncertain',0.3,'cartilage_depth_or_extent_unspecified')
        if kind=='sized':
            if re.search(r'(?:mild|small|low|leve)\s*(?:to|a|[-–])\s*(?:moderate|large|moderad)',t):return result('uncertain',0.4,'borderline_size')
            if self.has('high',t):return result('positive',0.93,'moderate_or_large')
            if self.has('low',t):return result('negative',0.9,'small_or_trace_excluded')
            return result('uncertain',0.4,'size_unspecified')
        if kind=='direct':return result('positive',0.92,'explicit_synovitis_or_thickening')
        if kind=='fracture':
            if self.has('remote',t):return result('uncertain',0.3,'fracture_temporality_unclear')
            return result('positive',0.9,'current_fracture_statement')
        if kind=='contusion':
            explicit=bool(re.search(r'contus|bruise|botcontus|kontuz|μωλωπ',t))
            fracture=bool(re.search(self.config['conditions']['Fracture']['anatomy'],t))
            if fracture and not re.search(r'(?:no|without|without\s+(?:evidence\s+of\s+)?associated|sin)\s+(?:a\s+)?(?:discrete\s+)?fracture',t):
                return result('uncertain',0.35,'marrow_injury_with_fracture')
            if explicit:return result('positive',0.9,'explicit_bone_contusion')
            return result('uncertain',0.3,'marrow_edema_cause_unspecified')
        raise AssertionError(kind)

    def extract(self,report):
        report=str(report);us=self.refine_units(units(report));combined_report='bilateral note' in norm(report) or 'two reports filed' in norm(report)
        out={}
        for condition in LABELS:
            candidates=[x for u in us if (x:=self.candidate(condition,u)) is not None]
            # Anatomy-only statements cannot overrule a definite finding elsewhere.
            substantive=[x for x in candidates if not x['rule_id'].endswith('anatomy_without_definite_status')]
            pool=substantive or candidates
            positives=[x for x in pool if x['state']=='positive'];negatives=[x for x in pool if x['state']=='negative']
            if combined_report:
                chosen=dict(state='uncertain',confidence=0.0,normalized_finding=condition+': combined reports require laterality review',rule_id='document.combined_reports',evidence_text='[BILATERAL NOTE: two reports filed under this study]',section='document',section_context='',evidence_start=None,evidence_end=None,historical_or_postoperative=False,severity_terms=[],uncertainty_detected=True,negation_detected=False)
            elif positives and negatives:
                chosen=dict(positives[0],state='uncertain',confidence=0.2,normalized_finding=condition+': conflicting positive and negative statements',rule_id='aggregation.contradictory_evidence')
            elif positives:chosen=max(positives,key=lambda x:(x['section']=='impression',x['confidence']))
            elif any(x['state']=='uncertain' for x in pool):chosen=max((x for x in pool if x['state']=='uncertain'),key=lambda x:(x['section']=='impression',x['confidence']))
            elif negatives:chosen=max(negatives,key=lambda x:(x['section']=='impression',x['confidence']))
            else:chosen=dict(state='not_mentioned',confidence=0.0,normalized_finding=condition+': no supported mention found',rule_id='document.not_mentioned',evidence_text='',section='',section_context='',evidence_start=None,evidence_end=None,historical_or_postoperative=False,severity_terms=[],uncertainty_detected=False,negation_detected=False)
            chosen=dict(chosen)
            chosen.update(condition=condition,method=self.config['method'],version=self.config['version'],candidates=candidates)
            out[condition]=chosen
        return out
