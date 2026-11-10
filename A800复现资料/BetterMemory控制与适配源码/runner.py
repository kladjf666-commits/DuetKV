#!/usr/bin/env python3
"""CPU controller. Reuses the existing environment; never changes old results."""
from pathlib import Path
import argparse, csv, gzip, io, os, subprocess, sys, tarfile
import common as c


def inputs():
    quality = []  # This campaign performs speed/memory only.
    with gzip.open(c.ROOT / 'data/speed.json.gz', 'rt', encoding='utf-8') as f:
        bank = __import__('json').load(f)
    import hashlib, json
    token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    for row in quality:
        if token_hash(row['input_ids']) != row['input_sha256'] or row.get('truncated_input'):
            raise ValueError('Quality input changed')
        if not row['stop_ids'] or row['max_new_tokens'] < 1: raise ValueError('Missing original stop/cap')
    if len(bank) != 8 or any(len(row) != 32768 for row in bank) or len({token_hash(r) for r in bank}) != 8:
        raise ValueError('Need eight distinct full speed prefixes')
    return quality, bank


def prepare(args):
    manifest = c.verify(c.ROOT); cfg = c.read(args.config)
    if cfg['bench_manifest'] != c.BASE: raise ValueError('Wrong frozen benchmark')
    c.verify(cfg['bench'], c.BASE)
    model = Path(cfg['model'])
    for name, checksum in c.read(c.ROOT / 'data/model_metadata.json').items():
        if name != 'configuration.json' and c.sha(model / name) != checksum:
            raise ValueError('Model metadata changed: ' + name)
    index = c.read(model / 'model.safetensors.index.json')
    weights = {name: [(model / name).stat().st_size, (model / name).stat().st_mtime_ns]
               for name in sorted(set(index['weight_map'].values()))}
    if weights != cfg['weights']: raise ValueError('Model weights identity changed')
    old = c.read(Path(cfg['old_out']) / 'results/phasekv/protocol.json')
    if old['weights'] != weights or old['model'] != cfg['model']:
        raise ValueError('Old protocol does not identify the same model')
    quality, bank = inputs()
    payload = dict(package_manifest=manifest, profile=c.PROFILE, cases=[list(case) for case in c.CASES],
        rounds=3, warmup=1, source_config=cfg, source_protocol=old,
        quality=[{**{k: r[k] for k in ('id', 'suite', 'task', 'input_sha256', 'max_new_tokens')},
                  'input_tokens': len(r['input_ids'])} for r in quality],
        speed_bank_digest=c.digest(bank))
    payload['mode'] = 'optimized_only_16_point_speed_memory'
    payload['history_sha256'] = c.sha(c.ROOT / 'data/historical_speed.json')
    payload['experiment'] = c.digest(payload)
    path = args.out / 'plan.json'
    if path.exists():
        if c.read(path) != payload: raise ValueError('Existing output belongs to different inputs/code')
    else: c.write(path, payload, new=True)
    return payload


def smi(query):
    result = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True)
    return [[x.strip() for x in row] for row in csv.reader(io.StringIO(result)) if row]


def select_gpu(index):
    cards = {i: dict(uuid=u, name=n, memory_mib=int(float(m)), used_mib=int(float(v)))
             for i, u, n, m, v in smi('--query-gpu=index,uuid,name,memory.total,memory.used')}
    card = cards.get(str(index))
    if not card or 'A800' not in card['name'] or card['memory_mib'] < 75000:
        raise ValueError('Requires A800 80GB')
    if card['used_mib'] > 512 or any(u == card['uuid'] for u, _ in smi('--query-compute-apps=gpu_uuid,pid')):
        raise RuntimeError('GPU occupied; no process was stopped')
    return card


def run(args):
    from report import report
    with c.lock(args.out / 'runner.lock'):
        plan = prepare(args)
        if report(args.out)['complete']:
            print('COMPLETE: no model loading or rerun'); return
        for path in c.expected_paths(plan):
            if (args.out / path).exists(): c.validate_saved(c.read(args.out / path), plan['experiment'])
        card = select_gpu(args.gpu)
        gpu_file = args.out / 'gpu.json'
        if gpu_file.exists() and c.read(gpu_file)['uuid'] != card['uuid']:
            raise ValueError('Resume must use the same physical GPU')
        if not gpu_file.exists(): c.write(gpu_file, card, new=True)
        cfg = plan['source_config']; env = dict(os.environ)
        for key in ('PYTHONPATH', 'PYTHONHOME'): env.pop(key, None)
        venv = Path(cfg['envs']) / 'main'; python = venv / 'bin/python'
        if not python.is_file(): raise ValueError('Preserved main environment missing')
        env.update(CUDA_VISIBLE_DEVICES=card['uuid'], PHASEKV_BENCH_ROOT=cfg['bench'],
            VIRTUAL_ENV=str(venv), PATH=str(venv / 'bin') + os.pathsep + env.get('PATH', ''),
            PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS=str(cfg['threads']), MKL_NUM_THREADS=str(cfg['threads']),
            OPENBLAS_NUM_THREADS='1', TORCH_CUDA_ARCH_LIST='8.0')
        with c.lock('/tmp/a800-bettermemory-' + card['uuid'] + '.lock'):
            with (args.out / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
                result = subprocess.run([str(python), '-u', str(c.ROOT / 'benchmark/gpu_worker.py'),
                                         '--out', str(args.out)], env=env, stdout=log, stderr=subprocess.STDOUT)
        c.verify(cfg['bench'], c.BASE)
        report(args.out)
        if result.returncode: raise RuntimeError('Worker stopped; inspect run.log/state.json before retry')
        print((args.out / 'summary.txt').read_text(encoding='utf-8'))


def export(args):
    c.verify(c.ROOT)
    target = args.out.parent / ('a800_bettermemory_final_results_' + __import__('time').strftime('%Y%m%d_%H%M%S') + '.tar.gz')
    with c.lock(args.out / 'runner.lock'):
        if target.exists(): raise FileExistsError(target)
        with tarfile.open(target, 'w:gz') as tar:
            for path in sorted(args.out.rglob('*')):
                if path.is_file() and not path.name.endswith(('.tmp', '.lock')):
                    tar.add(path, arcname='results/' + path.relative_to(args.out).as_posix())
            names = [line.split('  ', 1)[1] for line in (c.ROOT / 'SHA256SUMS').read_text().splitlines()]
            for name in names + ['SHA256SUMS']:
                tar.add(c.ROOT / name, arcname='test_package/' + name)
    target.with_name(target.name + '.sha256').write_bytes((c.sha(target) + '  ' + target.name + '\n').encode('ascii'))
    print(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'status', 'export'])
    parser.add_argument('--config', type=Path, default=Path('/data/phasekv-results/a800_quality_fulltext_v1/config.json'))
    parser.add_argument('--out', type=Path, default=Path('/data/phasekv-results/a800_bettermemory_final_v1'))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(); args.out = args.out.resolve()
    if args.action == 'prepare':
        with c.lock(args.out / 'runner.lock'): print('READY', prepare(args)['experiment'])
    elif args.action == 'run': run(args)
    elif args.action == 'export': export(args)
    else:
        from report import report
        report(args.out)
        print((args.out / 'summary.txt').read_text(encoding='utf-8'))


if __name__ == '__main__': main()
