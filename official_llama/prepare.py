"""Freeze pristine upstream trees and make auditable compatibility copies."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
REPOS = {
    'KIVI': ('https://github.com/jy-yuan/KIVI.git', '876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6'),
    'SnapKV': ('https://github.com/FasterDecoding/SnapKV.git', 'e216ddc84c5bd210378cbdbbba12ba02102aa640'),
    'SCOPE': ('https://github.com/Linking-ai/SCOPE.git', '8cbae48042e34da630c7a044f535de7a991c2999'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args])


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def verify():
    manifest = json.loads((ROOT / 'sources.json').read_text('utf-8'))
    for name, expected in manifest['files'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise RuntimeError(f'Source fingerprint mismatch: {name}; do not benchmark modified sources')
    return manifest


def prepare():
    manifest = {'schema': 1, 'repositories': {}, 'files': {}, 'changes': []}

    def save(relative, data):
        write(ROOT / relative, data)
        manifest['files'][relative] = sha(data)

    for name, (url, commit) in REPOS.items():
        repo = ROOT / 'vendor' / name
        actual = git(repo, 'rev-parse', 'HEAD').decode().strip()
        if actual != commit or git(repo, 'diff', 'HEAD', '--').strip():
            raise RuntimeError(f'{name} must be a clean checkout at {commit}')
        manifest['repositories'][name] = {'url': url, 'commit': commit}
        paths = git(repo, 'ls-files', '-z').decode().split('\0')
        for filename in filter(None, paths):
            # Use Git blobs so Windows CRLF conversion cannot change the bundle.
            data = git(repo, 'show', f'HEAD:{filename}')
            save(f'vendor/{name}/{filename}', data)
            save(f'runtime/{name}/{filename}', data)

    # Snapshot the user's current native PhaseKV; no imports from old baselines.
    for path in sorted((ROOT.parent / 'PhaseKV').rglob('*.py')):
        relative = path.relative_to(ROOT.parent).as_posix()
        save('runtime/' + relative, path.read_bytes())
    attention = ROOT.parent / 'models/Llama/llama_attention.py'
    save('runtime/phasekv_llama_attention.py', attention.read_bytes())

    def patch(relative, old, new, reason):
        path = ROOT / relative
        before = path.read_text('utf-8')
        if before.count(old) != 1:
            raise RuntimeError(f'Compatibility patch anchor changed: {relative}')
        after = before.replace(old, new)
        save(relative, after.encode())
        diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile=relative + '.upstream', tofile=relative))
        patch_name = 'patches/' + relative.replace('/', '__') + '.patch'
        save(patch_name, diff.encode())
        manifest['changes'].append({'file': relative, 'reason': reason, 'diff': patch_name})

    patch('runtime/SnapKV/snapkv/monkeypatch/llama_hijack_4_37.py',
          'cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)',
          'cos, sin = (kwargs.get("position_embeddings") or\n'
          '                self.rotary_emb(value_states, position_ids))',
          'Llama 3.1/Transformers 4.46 absolute-position RoPE API; selector and cache unchanged.')
    patch('runtime/SCOPE/model/llama_model.py',
          'from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position, StaticCache # Add to fix 4.44.2',
          'from transformers.models.llama.modeling_llama import LlamaModel, StaticCache\n'
          '_prepare_4d_causal_attention_mask_with_cache_position = LlamaModel._prepare_4d_causal_attention_mask_with_cache_position',
          'Transformers 4.46 moved this mask helper to LlamaModel; native H2O/schedule unchanged.')
    # Restrict to the flash/GQA class; do not modify the unused MHA implementation.
    relative = 'runtime/KIVI/models/llama_kivi.py'
    original = (ROOT / relative).read_text('utf-8')
    tail = original[original.index('class LlamaFlashAttention_KIVI'):]
    repaired = tail.replace('attn_output = torch.matmul(attn_weights, value_states_full)',
                            'attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))')
    if tail == repaired:
        raise RuntimeError('KIVI short-input GQA patch anchor missing')
    patch(relative, tail, repaired,
          'GQA shape fix for the unquantized value-only short-input branch; matches the adjacent residual branch. CUDA/Triton kernels untouched.')
    # The legacy registry is not part of the real packed implementation.
    relative = 'runtime/PhaseKV/__init__.py'
    before = (ROOT / relative).read_text('utf-8')
    patch(relative, before,
          '"""Standalone native packed PhaseKV snapshot; no legacy fake-quant registry."""\n',
          'Do not import the unrelated historical eval registry at package import time.')
    write(ROOT / 'sources.json', (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode())
    verify()
    print(f'Prepared and verified {len(manifest["files"])} source files; upstream kernels/selectors unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        result = verify()
        print(f'Verified {len(result["files"])} source files.')
    else:
        prepare()
