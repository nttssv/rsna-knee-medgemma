"""Offline syntax inspection only; never promotes rejected responses to predictions."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--review', required=True, type=Path)
a = p.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_core as core

cases = json.loads((a.review / 'private_cases.json').read_text())
decoder = json.JSONDecoder(object_pairs_hook=core.no_duplicates)
rows = []
for c in cases:
    for repeat in (1, 2):
        attempt = c['models'][f'medgemma_{repeat}']['attempts'][0]
        raw = attempt['text'].strip()
        fence = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', raw)
        candidate = fence.group(1) if fence else raw
        checked = core.validate_response(candidate, c['report'])
        strict_valid = all(x['status'] == 'valid' for x in checked)
        # Diagnostic search reports presence only. No object is extracted or accepted.
        complete_objects = 0
        for match in re.finditer(r'\{', raw):
            try:
                obj, _ = decoder.raw_decode(raw[match.start():])
                complete_objects += isinstance(obj, dict) and set(obj) == set(core.LABELS)
            except (ValueError, TypeError):
                pass
        try:
            json.loads(candidate, object_pairs_hook=core.no_duplicates)
            syntax_valid = True
        except (ValueError, TypeError):
            syntax_valid = False
        if fence and syntax_valid:
            category = 'json_in_outer_fence_only'
        elif syntax_valid:
            category = 'json_only'
        elif complete_objects:
            category = 'complete_condition_object_with_extra_text'
        elif attempt['status'] == 'generation_truncated':
            category = 'truncated_incomplete_condition_object'
        else:
            category = 'malformed_or_other'
        rows.append(dict(case=c['case'], repeat=repeat, generation_status=attempt['status'],
                         category=category, outer_fence_only=bool(fence),
                         complete_condition_objects=complete_objects,
                         diagnostic_json_syntax_valid=syntax_valid,
                         diagnostic_all_schema_and_evidence_valid=strict_valid,
                         diagnostic_valid_cells=sum(x['status']=='valid' for x in checked),
                         recorded_accepted_cells=0))
with (a.review/'medgemma_format_diagnostic.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
aggregate=dict(category_counts_per_repeat=dict(collections.Counter(r['category'] for r in rows if r['repeat']==1)),
               outer_fence_only_reports_per_repeat=sum(r['outer_fence_only'] for r in rows if r['repeat']==1),
               fence_only_all_schema_and_evidence_valid_reports_per_repeat=sum(r['outer_fence_only'] and r['diagnostic_all_schema_and_evidence_valid'] for r in rows if r['repeat']==1),
               note='Post hoc syntax diagnostic only. No original smoke prediction or metric changed.')
(a.review/'format_diagnostic_aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
print(json.dumps(aggregate,indent=2))
