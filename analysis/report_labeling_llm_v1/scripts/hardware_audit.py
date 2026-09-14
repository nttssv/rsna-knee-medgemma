"""Read local hardware/package/cache metadata without model loading or network calls."""
import argparse
import importlib.metadata
import json
import platform
import shutil
from benchmark_core import state_dir, write_json, private_path


def audit(state):
    result = {'system': platform.system(), 'machine': platform.machine(),
              'python': platform.python_version(), 'disk_free_gib': shutil.disk_usage(state).free / 2**30,
              'packages': {}, 'model_inference_started': False, 'model_downloads_started': False}
    for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface_hub']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda_version=torch.version.cuda)
        result['gpus'] = [{'name': torch.cuda.get_device_name(i),
                           'vram_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
                          for i in range(torch.cuda.device_count())]
        result['mps_available'] = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    except ImportError:
        result.update(cuda_available=None, gpus=[])
    cache = state / 'cache/huggingface'
    result['configured_cache_exists'] = cache.exists()
    result['configured_cache_file_bytes'] = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file() and not p.is_symlink()) if cache.exists() else 0
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir'); p.add_argument('--output')
    a = p.parse_args(); state = state_dir(a.state_dir)
    result = audit(state)
    if a.output:
        write_json(private_path(state, a.output), result)
    print(json.dumps(result, indent=2))
