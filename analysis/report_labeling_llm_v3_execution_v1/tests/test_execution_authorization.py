"""Private grant and self-stop checks using CPU-only fixtures; never a provider call."""
from datetime import datetime,timezone,timedelta
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import runtime_v3_execution as rt
import stop_watchdog as watch


@pytest.fixture
def authorized(tmp_path,monkeypatch):
    current=datetime(2026,9,15,10,tzinfo=timezone.utc);expected='f'*64
    grant=dict(approved=True,approval_reference='SYNTHETIC TEST; not actual user authorization',plan_sha256=expected,
        proposal_sha256=rt.sha(rt.ROOT/'configs/resource_proposal.json'),approved_at=(current-timedelta(minutes=6)).isoformat(),
        provider_start_requested_at=(current-timedelta(minutes=5)).isoformat(),provider_deadline=(current+timedelta(minutes=80)).isoformat(),
        maximum_usd=2.0,actual_compute_usd_per_hour=0.84,actual_storage_usd_per_hour=0.011,
        pod_id='synthetic-pod',gpu_name='NVIDIA RTX 6000 Ada Generation',gpu_count=1)
    path=tmp_path/'authorization.json';rt.write(path,grant)
    monkeypatch.setattr(rt,'checked_plan',lambda *args:({'synthetic':False},[]))
    monkeypatch.setattr(rt,'verify_watchdog',lambda *args:None)
    return dict(execute=True,prepared=tmp_path,expected_sha=expected,authorization=path,watchdog=tmp_path/'watchdog.json',current=current)


def test_valid_explicit_fixture_grant(authorized):
    assert rt.guard(**authorized)['approved'] is True


@pytest.mark.parametrize('field,value',[
    ('approved',False),('approved','true'),('approval_reference',''),('plan_sha256','wrong'),('proposal_sha256','wrong'),
    ('gpu_name','NVIDIA A100'),('gpu_count',2),('gpu_count',True),('pod_id',''),('maximum_usd',0),('maximum_usd',2.01),
    ('maximum_usd',0.25),('actual_compute_usd_per_hour',0.85),('actual_storage_usd_per_hour',0.02),
    ('actual_compute_usd_per_hour',-1),('provider_deadline','2026-09-15T09:59:00+00:00'),
    ('provider_deadline','2026-09-15T12:00:00+00:00'),('provider_deadline','2026-09-15T10:04:00+00:00'),
    ('provider_deadline','2026-09-15T11:00:00'),('approved_at','2026-09-15T10:01:00+00:00'),
    ('provider_start_requested_at','2026-09-15T10:01:00+00:00')])
def test_bad_grant_rejected(authorized,field,value):
    path=authorized['authorization'];grant=rt.read(path);grant[field]=value
    path.write_text(rt.json.dumps(grant))
    with pytest.raises((PermissionError,ValueError)):rt.guard(**authorized)


def test_world_readable_grant_rejected(authorized):
    authorized['authorization'].chmod(0o644)
    with pytest.raises(PermissionError):rt.guard(**authorized)


def test_missing_watchdog_rejected(authorized,monkeypatch):
    def fail(*args):raise PermissionError('Synthetic absent watchdog')
    monkeypatch.setattr(rt,'verify_watchdog',fail)
    with pytest.raises(PermissionError):rt.guard(**authorized)


def test_synthetic_plan_never_authorizes_real_model(authorized,monkeypatch):
    monkeypatch.setattr(rt,'checked_plan',lambda *args:({'synthetic':True},[]))
    with pytest.raises(PermissionError):rt.guard(**authorized)


@pytest.mark.parametrize('style,expected',[('modern',['pod','stop']),('legacy',['stop','pod'])])
def test_only_stop_self_command(style,expected,monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID','synthetic-pod')
    assert watch.command('synthetic-pod',style,'runpodctl')==['runpodctl']+expected+['synthetic-pod']
    with pytest.raises(PermissionError):watch.command('another-pod',style,'runpodctl')


def test_watchdog_waits_then_stops_without_real_sleep_or_provider():
    current=[datetime(2026,9,15,10,tzinfo=timezone.utc)];calls=[];slept=[]
    def sleep(seconds):slept.append(seconds);current[0]+=timedelta(seconds=seconds)
    def run(cmd,**kwargs):calls.append((cmd,current[0]));return SimpleNamespace(returncode=0)
    deadline=current[0]+timedelta(seconds=61)
    assert watch.wait_and_stop(deadline,['FAKE-STOP'],clock=lambda:current[0],sleep=sleep,run=run)
    assert slept==[30,30,1] and calls==[(['FAKE-STOP'],deadline)]


def test_watchdog_stop_failure_is_bounded():
    calls=[];current=datetime(2026,9,15,10,tzinfo=timezone.utc)
    def run(cmd,**kwargs):calls.append(cmd);raise OSError('synthetic')
    assert not watch.wait_and_stop(current,['FAKE-STOP'],clock=lambda:current,sleep=lambda n:None,run=run)
    assert len(calls)==3


def test_expired_proposal_rejected(authorized):
    authorized['current']=datetime(2026,9,17,10,tzinfo=timezone.utc)
    with pytest.raises(PermissionError,match='proposal expired'):rt.guard(**authorized)


@pytest.mark.parametrize('style',['modern','legacy'])
def test_cli_preflight_is_read_only(style,monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID','synthetic-pod');calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stdout='stop usage' if '--help' in cmd else 'synthetic-pod')
    assert watch.verify_cli('synthetic-pod',style,'runpodctl',run)
    assert len(calls)==2 and calls[0][-1]=='--help' and 'get' in calls[1]
    assert all(not ('stop' in c and c[-1]=='synthetic-pod') for c in calls)


def test_watchdog_refuses_unverified_cli(monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID','synthetic-pod')
    with pytest.raises(RuntimeError):
        watch.verify_cli('synthetic-pod','modern','runpodctl',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
