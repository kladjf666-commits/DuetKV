"""Require all native paths to finish and exercise compression before larger runs."""
from pathlib import Path
import argparse
import json


def check(folder):
    root = Path(folder)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    statuses = json.loads((root / 'status.json').read_text())
    methods = manifest['arguments']['methods']
    assert {s['method'] for s in statuses} == set(methods), 'Some methods did not run'
    assert all(s['exit_code'] == 0 for s in statuses), 'A native worker failed; inspect its log'
    for method in methods:
        rows = [json.loads(line) for line in (root / f'{method}.jsonl').read_text().splitlines()]
        assert rows and all(r['status'] == 'ok' for r in rows), f'{method} has incomplete/failed records'
        for row in rows:
            assert row['dtype'] == 'float16'
            trace = row['final_trace']
            length = row['input_tokens'] + row['cached_generated_tokens']
            assert row['final_storage']['physical_bytes'] > 0
            if method == 'kivi':
                assert trace['key_quantized'] and trace['value_quantized'], 'KIVI quantization not exercised'
            elif method == 'phasekv':
                assert trace['cold_segments'] > 0, 'PhaseKV cold packing not exercised'
                assert row['prefill_trace']['prompt_tokens'] == round(row['input_tokens'] * .7)
            elif method == 'snapkv':
                assert row['prefill_trace']['physical_tokens'] == 2048
                assert trace['physical_tokens'] == 2048 + row['cached_generated_tokens']
            elif method.startswith('scope'):
                assert row['scope_mode'].startswith('native_'), 'SCOPE decode schedule not exercised'
                assert trace['physical_tokens'] < 2048 + row['cached_generated_tokens'], 'SCOPE decode eviction not exercised'
            else:
                assert trace['physical_tokens'] == length
        print(f'{method}: FP16 native path + compression trigger checks passed')
    print('Deployment smoke passed. This is not dataset accuracy or publication-ready throughput.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory')
    check(parser.parse_args().run_directory)
