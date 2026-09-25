"""Local inventory/checklist preparation only. No upload, download, SSH or execute."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

from observe_provider import ROOT, private_write, sha

STAGE = Path(__file__).resolve().parents[1]
PREPARED_SHA = 'c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc'
EXECUTION_SHA = 'ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5'
COMPONENTS = {
    'external_worker': 'analysis/qwen_execution_provisioning_runtime_v1/scripts/external_shutdown.py',
    'external_transport': 'analysis/qwen_execution_provisioning_runtime_v1/scripts/provider_transport.py',
    'external_network_supervisor': 'analysis/qwen_execution_provisioning_runtime_v1/scripts/network_supervisor.py',
    'outer_controller': 'analysis/qwen_execution_provisioning_runtime_v1/scripts/controller.py',
    'inner_handoff': 'analysis/qwen_execution_provisioning_runtime_v1/scripts/handoff.py',
    'inner_runner': 'analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py',
    'pod_watchdog': 'analysis/qwen_execution_v1/scripts/stop_watchdog.py',
    'pod_preflight_verifier': 'analysis/qwen_execution_v1/scripts/stop_watchdog.py',
    'cache_auditor': 'analysis/report_labeling_llm_v2/scripts/load_preflight.py',
    'cache_manifest': 'analysis/report_labeling_llm_v2/configs/weight_manifest.json',
}


def verify_protected(root=ROOT):
    manifest = json.loads((STAGE / 'configs/protected_sources.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise ValueError('Frozen source drift: ' + name)
    return len(manifest['files'])


def load_inner():
    path = ROOT / COMPONENTS['inner_runner']
    spec = importlib.util.spec_from_file_location('environment_frozen_inner', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(prepared, caches):
    count = verify_protected()
    policy = json.loads((STAGE / 'configs/preparation.json').read_text())
    if any(value is not False for key, value in policy.items() if key.endswith('_enabled')):
        raise ValueError('Preparation switches must all remain false')
    inner = load_inner()
    inner.validate_execution_plan(inner.ROOT / 'configs/execution_plan.json', EXECUTION_SHA)
    plan, rows = inner.check_plan(Path(prepared), PREPARED_SHA)
    # This is a byte/hash audit only; never create_encoder/create_backend/load_model.
    from load_preflight import audit_cache
    cache_results = [audit_cache(Path(cache), 'qwen') for cache in caches]
    manifest = json.loads((ROOT / COMPONENTS['cache_manifest']).read_text())['qwen']
    return {'stage': 'LOCAL_ENVIRONMENT_PREPARATION_ONLY',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'prepared_plan_sha256': PREPARED_SHA, 'execution_plan_sha256': EXECUTION_SHA,
        'protected_file_count': count, 'five_report_package_verified': len(rows) == 5,
        'prepared_files': plan['files'],
        'components': {k: {'path': path, 'sha256': sha((ROOT / path).read_bytes())}
                       for k, path in COMPONENTS.items()},
        'cache': {'model_id': manifest['model_id'], 'revision': manifest['revision'],
            'required_bytes': sum(f['bytes'] for f in manifest['files']),
            'required_file_count': len(manifest['files']), 'audits': cache_results,
            'complete_local_copy_found': any(a['complete'] for a in cache_results),
            'pod_copy_verified': False},
        'external_linux_environment_verified': False, 'pod_local_runpodctl_verified': False,
        'pod_local_watchdog_deployed': False, 'remote_transport_verified': False,
        'all_live_switches_false': True, 'deployment_performed': False,
        'ready_for_live_use': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cache', action='append', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.prepared, args.cache)
    private_write(args.output, result)
    print(json.dumps({k: result[k] for k in ['protected_file_count', 'five_report_package_verified', 'deployment_performed', 'ready_for_live_use']}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
