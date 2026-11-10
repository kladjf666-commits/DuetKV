"""Package only the frozen speed prompt and the validated keep80 runtime."""
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / 'official_llama'
DELIVERY = ROOT / 'PhaseKV与Full消融'
PREFIX = 'phasekv_4090d_speed/'
INPUT = 'performance/8192+256/0'


def build():
    files = {}
    for name in ('quick.py', 'run.sh', 'README.txt'):
        files[name] = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
    for name in ('native_runtime.py', 'ablation_runtime.py', 'ablation_attention.py', 'ablation_kernels.py',
                 'ablation_generate.py', 'config.json', 'check_model.py', 'runtime/phasekv_llama_attention.py',
                 'runtime/PhaseKV/prompt_tier.py', 'runtime/PhaseKV/hetero_cache.py', 'runtime/PhaseKV/quant.py'):
        files[name] = (OFFICIAL / name).read_bytes().replace(b'\r\n', b'\n')
    release = 'migration/keep80/'
    with zipfile.ZipFile(DELIVERY / 'PhaseKV_4090_keep80_full.zip') as source:
        manifest = json.loads(source.read(release + 'files.json'))
        for name, expected in manifest.items():
            if name.startswith('channel_impl/'):
                data = source.read(release + name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise RuntimeError(f'Corrupt keep80 release file {name}')
                files['runtime/PhaseKV/' + name.removeprefix('channel_impl/')] = data
    frozen = OFFICIAL / 'bench_assets/frozen'
    index = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    archive = frozen / 'samples.jsonl.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != index['samples_sha256']:
        raise RuntimeError('Frozen benchmark input archive changed')
    with gzip.open(archive, 'rt', encoding='utf-8') as inputs:
        sample, = [row for line in inputs if (row := json.loads(line))['id'] == INPUT]
    files['sample.json'] = json.dumps(sample, separators=(',', ':')).encode()
    files['model_files.json'] = json.dumps({file: index['tokenizer_files'][file] for file in
        ('config.json', 'model.safetensors.index.json')}, indent=2).encode()
    files['files.json'] = json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in files.items()}, indent=2).encode()
    target = DELIVERY / 'PhaseKV_4090D_speed_quick.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as dest:
        for name, data in files.items():
            dest.writestr(PREFIX + name, data)
    target.with_suffix('.tmp').replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    build()
