"""CPU-only lifecycle regression tests; every worker output is SYNTHETIC.

These tests execute real subprocesses, without Torch, models, CUDA or provider
calls. Fake CUDA memory is measured by the parent after worker teardown.
"""

import json
import os
from pathlib import Path
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isolated_runner  # noqa: E402
from inference_core import ContractError  # noqa: E402


SYNTHETIC_WORKER = r'''
import json
import os
from pathlib import Path
import sys
import time

placement, output, events, scenario = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
with open(events, 'a') as stream:
    stream.write(json.dumps({'kind': 'SYNTHETIC', 'placement': placement, 'pid': os.getpid()}) + '\n')
    stream.flush()

if scenario == 'timeout':
    time.sleep(60)
    raise SystemExit(99)
if scenario == 'missing':
    raise SystemExit(42)

result = {
    'kind': 'SYNTHETIC', 'placement': placement, 'pid': os.getpid(),
    'phase': 'prepare', 'status': 'cuda_oom',
    'error_type': 'OutOfMemoryError', 'error': 'SYNTHETIC CUDA out of memory',
}
exitcode = 42
if scenario == 'complete':
    result.update(phase='inference', status='complete', error_type=None, error=None)
    exitcode = 0
elif scenario == 'failed':
    result.update(status='failed', error_type='RuntimeError', error='SYNTHETIC non-OOM failure')
    exitcode = 1
elif scenario == 'inference_oom':
    result['phase'] = 'inference'
elif scenario == 'wrong_pid':
    result['pid'] += 100000
elif scenario == 'wrong_placement':
    result['placement'] = 'two_gpu' if placement == 'single_gpu' else 'single_gpu'
elif scenario == 'wrong_exit':
    exitcode = 1
elif scenario == 'false_complete':
    result.update(phase='inference', status='complete')
elif scenario == 'oom_exit_zero':
    exitcode = 0
elif scenario == 'malformed':
    (output / 'worker_result.json').write_text('not json')
    raise SystemExit(42)

(output / 'worker_result.json').write_text(json.dumps(result))
raise SystemExit(exitcode)
'''


def events_at(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


class FakeCuda:
    """Retained state disappears only after the worker process has exited."""

    def __init__(self, event_path, leak=False):
        self.event_path = event_path
        self.leak = leak
        self.synchronized = []
        self.measured_after_exit = []

    def device_count(self):
        return 2

    def synchronize(self, index):
        self.synchronized.append(index)

    def mem_get_info(self, index):
        events = events_at(self.event_path)
        if events:
            # A fresh memory decision may not happen while a failed CUDA-owning
            # worker is still alive or remains an unreaped child.
            assert_reaped(events[-1]['pid'])
            self.measured_after_exit.append((events[-1]['placement'], index))
        gib = 10.42 if self.leak and events and index == 0 else 14.46
        return int(gib * 2**30), int(14.56 * 2**30)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    worker = tmp_path / 'synthetic_worker.py'
    worker.write_text(SYNTHETIC_WORKER)
    events = tmp_path / 'events.jsonl'
    output = tmp_path / 'outputs'
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps({
        'config_path': str(ROOT / 'configs' / 'inference.json'),
        'data_dir': str(tmp_path / 'data'),
        'base_dir': str(tmp_path / 'base'),
        'adapter_dir': str(tmp_path / 'adapter'),
        'output_root': str(output),
        'asset_verification_seconds': 0.1,
        'dependency_install_seconds': 0.2,
    }))
    selected = []

    def install(scenarios, *, leak=False):
        def command(actual_request, placement, attempt_dir):
            assert Path(actual_request) == request_path
            assert Path(attempt_dir) == output / placement
            selected.append(placement)
            assert len(selected) <= 2, 'No third model attempt is permitted'
            if len(selected) == 2:
                assert_reaped(events_at(events)[0]['pid'])
            return [sys.executable, str(worker), placement, str(attempt_dir), str(events), scenarios[placement]]

        monkeypatch.setattr(isolated_runner, '_worker_command', command)
        return FakeCuda(events, leak=leak)

    return request_path, output, events, selected, install


def load_attempts(output):
    return json.loads((output / 'placement_attempts.json').read_text())['attempts']


def test_single_success_is_one_process_with_no_dual_attempt(harness):
    request, output, events, selected, install = harness
    cuda = install({'single_gpu': 'complete'})
    isolated_runner.run_isolated(request, time.monotonic() + 10, cuda=cuda)
    assert selected == ['single_gpu']
    assert_reaped(events_at(events)[0]['pid'])
    attempts = load_attempts(output)
    assert len(attempts) == 1
    assert attempts[0]['worker_result']['status'] == 'complete'
    assert attempts[0]['returncode'] == 0


def test_oom_allows_dual_only_after_failed_process_reaped_and_vram_recovers(harness):
    request, output, events, selected, install = harness
    cuda = install({'single_gpu': 'oom', 'two_gpu': 'complete'})
    isolated_runner.run_isolated(request, time.monotonic() + 10, cuda=cuda)
    assert selected == ['single_gpu', 'two_gpu']
    assert [row['placement'] for row in events_at(events)] == selected
    assert len({row['pid'] for row in events_at(events)}) == 2
    for row in events_at(events):
        assert_reaped(row['pid'])
    first, second = load_attempts(output)
    assert first['worker_result']['status'] == 'cuda_oom'
    assert first['returncode'] == 42
    assert first['free_vram_before_attempt_gib'] == pytest.approx([14.46, 14.46])
    assert first['free_vram_after_process_exit_gib'] == pytest.approx([14.46, 14.46])
    assert second['worker_result']['status'] == 'complete'
    assert {0, 1}.issubset(cuda.synchronized)
    assert ('single_gpu', 0) in cuda.measured_after_exit
    assert ('single_gpu', 1) in cuda.measured_after_exit


def test_unrecovered_vram_refuses_dual_process(harness):
    request, output, events, selected, install = harness
    cuda = install({'single_gpu': 'oom'}, leak=True)
    with pytest.raises(ContractError):
        isolated_runner.run_isolated(request, time.monotonic() + 10, cuda=cuda)
    assert selected == ['single_gpu']
    assert_reaped(events_at(events)[0]['pid'])
    assert load_attempts(output)[0]['free_vram_after_process_exit_gib'][0] == pytest.approx(10.42)


@pytest.mark.parametrize('scenario', [
    'failed', 'inference_oom', 'missing', 'wrong_pid', 'wrong_placement',
    'wrong_exit', 'false_complete', 'oom_exit_zero', 'malformed',
])
def test_non_oom_or_missing_or_contradictory_receipt_never_falls_back(harness, scenario):
    request, output, events, selected, install = harness
    cuda = install({'single_gpu': scenario})
    with pytest.raises((ContractError, ValueError)):
        isolated_runner.run_isolated(request, time.monotonic() + 10, cuda=cuda)
    assert selected == ['single_gpu']
    assert_reaped(events_at(events)[0]['pid'])
    assert len(load_attempts(output)) == 1


def test_dual_oom_aborts_without_a_third_process(harness):
    request, output, events, selected, install = harness
    cuda = install({'single_gpu': 'oom', 'two_gpu': 'oom'})
    with pytest.raises(ContractError):
        isolated_runner.run_isolated(request, time.monotonic() + 10, cuda=cuda)
    assert selected == ['single_gpu', 'two_gpu']
    assert len(load_attempts(output)) == 2
    for row in events_at(events):
        assert_reaped(row['pid'])


def test_timeout_kills_and_reaps_worker_without_fallback(harness):
    request, output, events, selected, install = harness
    cuda = install({'single_gpu': 'timeout'})
    started = time.monotonic()
    with pytest.raises((ContractError, TimeoutError)):
        isolated_runner.run_isolated(request, started + 0.75, cuda=cuda)
    assert time.monotonic() - started < 8
    assert selected == ['single_gpu']
    child_events = events_at(events)
    assert len(child_events) == 1
    assert_reaped(child_events[0]['pid'])
    attempts = load_attempts(output)
    assert len(attempts) == 1
    assert attempts[0]['returncode'] != 0


def test_existing_output_directory_is_not_overwritten(harness):
    request, output, events, selected, install = harness
    output.mkdir()
    historical = output / 'historical.txt'
    historical.write_text('SYNTHETIC historical output')
    cuda = install({'single_gpu': 'complete'})
    with pytest.raises((FileExistsError, ContractError)):
        isolated_runner.run_isolated(request, time.monotonic() + 10, cuda=cuda)
    assert historical.read_text() == 'SYNTHETIC historical output'
    assert selected == []
    assert events_at(events) == []
