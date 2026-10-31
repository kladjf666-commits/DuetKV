"""Verify both the additive ablation bundle and its frozen base dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT/'attention_ablation_bundle.json').read_text('utf-8'))
    for group in ('files', 'base_files'):
        for name, expected in manifest[group].items():
            path = ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Bundle/base mismatch: {name}. Use the original validated deployment plus this overlay.')
    from prepare import verify as verify_sources
    verify_sources()
    print('Attention ablation overlay and frozen base verified; no GPU checks claimed.')
    return manifest


if __name__ == '__main__':
    verify()
