"""Prepare only the original 58 report-only inputs; no inference or new split."""
import argparse
from benchmark_core import state_dir, prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = prepare(state_dir(a.state_dir), a.output)
    print('Prepared label-free inputs:', result['partition_sizes'], 'Unlabeled studies processed: 0')
