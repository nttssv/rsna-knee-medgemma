"""Fictitious parent/child metadata for artifact integrity tests, not model output."""
import json
from core import sha
from session_contract import ORDER, context_for


def attach_session(prepared,root):
    manifest=json.loads((prepared/'manifest.json').read_text())
    plan=dict(prepared_manifest_sha256=sha(prepared/'manifest.json'),
              candidate_code_sha256=manifest['candidate_code_sha256'],run_order=ORDER)
    (prepared/'fixture_plan.json').write_text(json.dumps(plan))
    start=dict(session_id='synthetic-metadata-test',reviewed_plan_sha256=sha(prepared/'fixture_plan.json'),
               plan_file='fixture_plan.json',run_order=ORDER,parent_pid=0)
    (root/'session_start.json').write_text(json.dumps(start));start_sha=sha(root/'session_start.json')
    results=[]
    for index,key in enumerate(ORDER):
        dispatch=dict(session_id=start['session_id'],reviewed_plan_sha256=start['reviewed_plan_sha256'],
            parent_session_start_sha256=start_sha,run_order_index=index,run_key=key,nonce_sha256='synthetic-hash')
        path=root/f'dispatch-{index}.json';path.write_text(json.dumps(dispatch))
        mpath=root/key/'run_manifest.json';m=json.loads(mpath.read_text())
        m.update(runtime_adapter_version=1,synthetic=False,**context_for(start,start_sha,dispatch,sha(path)))
        m['artifact_sha256']={'predictions.jsonl':sha(root/key/'predictions.jsonl')}
        mpath.write_text(json.dumps(m))
        results.append(dict(run_key=key,run_order_index=index,status='completed',run_manifest_sha256=sha(mpath)))
    (root/'session_manifest.json').write_text(json.dumps(dict(status='completed',session_id=start['session_id'],
        reviewed_plan_sha256=start['reviewed_plan_sha256'],parent_session_start_sha256=start_sha,results=results)))
