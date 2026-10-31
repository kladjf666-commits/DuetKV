"""Verify additive benchmark files; original runtime is checked by prepare.py."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'benchmark_bundle.json').read_text('utf-8'))
    for relative,expected in manifest['files'].items():
        path=ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Benchmark bundle mismatch: '+relative)
    print(f'Benchmark files verified: {len(manifest["files"])}')


if __name__=='__main__':
    main()
