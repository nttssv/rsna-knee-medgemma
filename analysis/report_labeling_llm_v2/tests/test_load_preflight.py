"""CPU tests for cache provenance and bounded load-only dispatch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import load_preflight as lp


def cache_fixture(tmp_path,monkeypatch):
    snap=tmp_path/'models--Test--Model'/'snapshots'/'pinned';snap.mkdir(parents=True)
    blobs={'part.safetensors':b'fake weights','model.safetensors.index.json':json.dumps({'weight_map':{'x':'part.safetensors'}}).encode()}
    files=[]
    for name,data in blobs.items():
        (snap/name).write_bytes(data)
        files.append(dict(file=name,bytes=len(data),algorithm='sha256',digest=hashlib.sha256(data).hexdigest()))
    cfg={'model_id':'Test/Model','revision':'pinned'}
    manifest=dict(cfg,files=files)
    monkeypatch.setattr(lp,'config',lambda name:{'medgemma':manifest} if name=='weight_manifest' else cfg)
    return snap,manifest


def test_complete_cache_and_same_size_corruption(tmp_path,monkeypatch):
    snap,_=cache_fixture(tmp_path,monkeypatch)
    assert lp.audit_cache(tmp_path,'medgemma')['complete']
    (snap/'part.safetensors').write_bytes(b'FAKE weights')
    assert not lp.audit_cache(tmp_path,'medgemma')['complete']


def test_missing_shard_fails(tmp_path,monkeypatch):
    snap,_=cache_fixture(tmp_path,monkeypatch);(snap/'part.safetensors').unlink()
    assert not lp.audit_cache(tmp_path,'medgemma')['complete']


def test_index_must_cover_expected_shards(tmp_path,monkeypatch):
    snap,m=cache_fixture(tmp_path,monkeypatch)
    path=snap/'model.safetensors.index.json';path.write_text('{"weight_map":{"x":"other.safetensors"}}')
    entry=next(e for e in m['files'] if e['file']==path.name)
    entry.update(bytes=path.stat().st_size,digest=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='shards'):lp.audit_cache(tmp_path,'medgemma')


def test_git_blob_digest(tmp_path):
    p=tmp_path/'small';p.write_bytes(b'abc')
    assert lp.digest_file(p,'git_blob_sha1')==hashlib.sha1(b'blob 3\0abc').hexdigest()


def test_standalone_worker_refused():
    with pytest.raises(PermissionError,match='bounded parent'):
        lp.verify_parent(SimpleNamespace(execute_load=True,attestation_fd=None,parent_pid=os.getppid()))


def test_worker_pipe_nonce(tmp_path):
    r,w=os.pipe()
    with os.fdopen(w,'w') as s:json.dump(dict(nonce='test',model='medgemma',plan_sha256='sha'),s)
    args=SimpleNamespace(execute_load=True,attestation_fd=r,parent_pid=os.getppid(),
        nonce_sha256=hashlib.sha256(b'test').hexdigest(),model='medgemma',plan_sha256='sha')
    lp.verify_parent(args)


def test_no_inference_calls_in_load_script():
    tree=ast.parse(Path(lp.__file__).read_text())
    calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(f,ast.Attribute) and f.attr in {'generate','forward','train','backward','step'} for f in calls)


def test_explicit_load_flag_precedes_plan_or_backend():
    with pytest.raises(PermissionError,match='execute-load'):
        lp.run(SimpleNamespace(execute_load=False))


@pytest.mark.parametrize('child_ok',[True,False])
def test_parent_uses_supervisor_tuple_and_stops_on_failure(tmp_path,monkeypatch,child_ok):
    monkeypatch.setattr(lp,'check_plan',lambda *a:None)
    monkeypatch.setattr(lp,'code_hashes',lambda:{'synthetic':'hash'})
    calls=[]
    def fake_supervise(command,timeout,pass_fds):
        model=command[command.index('--model')+1]
        out=Path(command[command.index('--output')+1]);out.mkdir()
        assert 0<timeout<=lp.WORKER_SECONDS and len(pass_fds)==1
        calls.append(model)
        lp.write_json(out/'result.json',dict(status='completed',model=model,
            plan_sha256='sha',candidate_code_sha256={'synthetic':'hash'}))
        return child_ok,'completed' if child_ok else 'hard_timeout'
    monkeypatch.setattr(lp,'supervise',fake_supervise)
    args=SimpleNamespace(execute_load=True,prepared=tmp_path,plan=tmp_path/'plan',
        plan_sha256='sha',output=tmp_path/'results',cache=tmp_path/'cache')
    assert lp.run(args)==child_ok
    assert calls==(['medgemma','qwen'] if child_ok else ['medgemma'])
    r=json.loads((args.output/'session_result.json').read_text())
    assert r['generation_calls']==0 and r['provider_stopped'] is False
