"""Pod-local self-stop backstop. Never starts compute or prints credentials."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def command(pod_id,style,executable):
    if not pod_id or pod_id!=os.environ.get('RUNPOD_POD_ID'):
        raise PermissionError('Only this pod may be stopped')
    if style not in ('modern','legacy'):raise ValueError('Unknown CLI syntax')
    return [executable]+(['pod','stop'] if style=='modern' else ['stop','pod'])+[pod_id]


def verify_cli(pod_id,style,executable,run=subprocess.run):
    stop=command(pod_id,style,executable)
    help_result=run(stop[:-1]+['--help'],capture_output=True,text=True,timeout=30,check=False)
    if help_result.returncode!=0 or 'stop' not in help_result.stdout.lower():
        raise RuntimeError('Installed CLI stop syntax is unverified')
    get=[executable]+(['pod','get'] if style=='modern' else ['get','pod'])+[pod_id]
    result=run(get,capture_output=True,text=True,timeout=30,check=False)
    if result.returncode!=0 or pod_id not in result.stdout:
        raise RuntimeError('CLI cannot inspect this pod; do not begin inference')
    return True


def wait_and_stop(deadline,cmd,clock=lambda:datetime.now(timezone.utc),sleep=time.sleep,run=subprocess.run):
    while (remaining:=(deadline-clock()).total_seconds())>0:
        sleep(min(30,remaining))
    # Retry stop requests only, never inference. Provider state is verified externally.
    for _ in range(3):
        try:
            result=run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30,check=False)
            if result.returncode==0:return True
        except (OSError,subprocess.TimeoutExpired):pass
        sleep(5)
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pod-id',required=True);p.add_argument('--deadline',required=True)
    p.add_argument('--style',choices=['modern','legacy'],required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--confirm-self-stop',action='store_true')
    a=p.parse_args()
    if not a.confirm_self_stop:raise PermissionError('Explicit self-stop authorization required')
    deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None or not 0<(deadline-datetime.now(timezone.utc)).total_seconds()<=5400:
        raise ValueError('Use a future deadline within 90 minutes')
    executable=shutil.which('runpodctl')
    if not executable:raise RuntimeError('Provider CLI missing; do not begin inference')
    cmd=command(a.pod_id,a.style,executable)
    verify_cli(a.pod_id,a.style,executable)
    a.receipt.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with a.receipt.open('x') as f:
        os.chmod(f.name,0o600)
        json.dump(dict(status='armed',pid=os.getpid(),pod_id=a.pod_id,deadline=a.deadline,
            command=cmd,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            armed_at=datetime.now(timezone.utc).isoformat(),cli_syntax_and_read_access_verified=True,
            provider_stop_verified=False),f)
        f.flush();os.fsync(f.fileno())
    success=wait_and_stop(deadline,cmd)
    raise SystemExit(0 if success else 1)


if __name__=='__main__':main()
