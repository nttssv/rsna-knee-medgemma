"""Freeze after complete development runs and passing repeatability, before validation."""
import argparse
from benchmark_core import state_dir, freeze

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir'); p.add_argument('--prepared', required=True)
    p.add_argument('--medgemma-run', required=True); p.add_argument('--qwen-run', required=True)
    p.add_argument('--repeatability', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    freeze(state_dir(a.state_dir), a.prepared, a.medgemma_run, a.qwen_run, a.repeatability, a.output)
    print('LLM execution freeze created. Original 40/18 split unchanged; validation remains exploratory.')
