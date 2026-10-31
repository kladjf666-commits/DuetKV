"""Check local checkpoint completeness without downloading or loading weights."""
from pathlib import Path
import argparse
import json
import struct


def check(directory):
    root = Path(directory)
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    index = json.loads((root / 'model.safetensors.index.json').read_text('utf-8'))
    for filename in sorted(set(index['weight_map'].values())):
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Invalid path in weight index')
        with path.open('rb') as handle:
            header_size = struct.unpack('<Q', handle.read(8))[0]
            if not 0 < header_size < 100_000_000:
                raise ValueError(f'Invalid safetensors header: {path}')
            header = json.loads(handle.read(header_size))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        expected = 8 + header_size + max(t['data_offsets'][1] for t in tensors.values())
        if path.stat().st_size != expected:
            raise ValueError(f'Incomplete shard: {filename}: expected {expected}, got {path.stat().st_size}')
        for tensor, owner in index['weight_map'].items():
            if owner == filename and tensor not in tensors:
                raise ValueError(f'Missing tensor {tensor} in {filename}')
        print(f'OK {filename}: {expected / 10**9:.3f} GB')
    print('Checkpoint structure/length checks passed (not a cryptographic authenticity check).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory')
    check(parser.parse_args().model_directory)
