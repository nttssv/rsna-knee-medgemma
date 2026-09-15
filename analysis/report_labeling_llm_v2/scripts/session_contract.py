"""Bind measured runs to one complete reviewed parent session."""
import json
from pathlib import Path
from core import sha

ORDER = ['medgemma-1','medgemma-2','qwen-1','qwen-2']


def context_for(start, start_sha, dispatch, dispatch_sha):
    return dict(reviewed_plan_sha256=start['reviewed_plan_sha256'],
                session_id=start['session_id'],run_order_index=dispatch['run_order_index'],
                expected_run_key=dispatch['run_key'],parent_session_start_sha256=start_sha,
                dispatch_sha256=dispatch_sha)


def verify_child(root, key, index, start, start_sha):
    dispatch_path=root/f'dispatch-{index}.json'
    dispatch=json.loads(dispatch_path.read_text())
    if (dispatch['run_key']!=key or dispatch['run_order_index']!=index or
            dispatch['session_id']!=start['session_id'] or
            dispatch['reviewed_plan_sha256']!=start['reviewed_plan_sha256'] or
            dispatch['parent_session_start_sha256']!=start_sha):
        raise ValueError('Child dispatch does not match the parent session')
    expected=context_for(start,start_sha,dispatch,sha(dispatch_path))
    manifest=json.loads((root/key/'run_manifest.json').read_text())
    if (manifest.get('status')!='completed' or manifest.get('synthetic') is not False or
            manifest.get('runtime_adapter_version')!=1 or
            any(manifest.get(k)!=v for k,v in expected.items())):
        raise ValueError('Child is not a completed run of the reviewed parent session')
    return manifest


def verify_session(prepared, root):
    prepared,root=Path(prepared).resolve(),Path(root).resolve()
    if root!=prepared/'adapter_runs':
        raise ValueError('Measured runs must belong to the prepared parent session')
    start_path=root/'session_start.json';start=json.loads(start_path.read_text());start_sha=sha(start_path)
    final=json.loads((root/'session_manifest.json').read_text())
    if (start['run_order']!=ORDER or final['status']!='completed' or
            final['session_id']!=start['session_id'] or
            final['reviewed_plan_sha256']!=start['reviewed_plan_sha256'] or
            final['parent_session_start_sha256']!=start_sha):
        raise ValueError('Missing completed reviewed four-run parent session')
    plan_name=start['plan_file']
    if Path(plan_name).name!=plan_name or sha(prepared/plan_name)!=start['reviewed_plan_sha256']:
        raise ValueError('Parent reviewed plan hash mismatch')
    plan=json.loads((prepared/plan_name).read_text())
    manifest=json.loads((prepared/'manifest.json').read_text())
    if (plan['prepared_manifest_sha256']!=sha(prepared/'manifest.json') or
            plan['candidate_code_sha256']!=manifest['candidate_code_sha256'] or
            plan['run_order']!=ORDER):
        raise ValueError('Parent plan does not match prepared experiment')
    results=final['results']
    if len(results)!=4 or [r['run_key'] for r in results]!=ORDER:
        raise ValueError('Parent session has an incomplete or reordered run list')
    for index,key in enumerate(ORDER):
        result=results[index]
        if (result['status']!='completed' or result['run_order_index']!=index or
                result['run_manifest_sha256']!=sha(root/key/'run_manifest.json')):
            raise ValueError('Parent receipt differs from child result')
        verify_child(root,key,index,start,start_sha)
    return final
