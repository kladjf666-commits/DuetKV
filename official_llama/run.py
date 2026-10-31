"""Offline, isolated native Llama deployment smoke/performance runner.

This is NOT the 5,456-sample quality benchmark. Frozen input IDs can be supplied
as JSONL for later dataset integration; no invented accuracy metric is reported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare import verify, sha


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def generate(model, method, options, sample, *, progress=True):
    import torch
    from transformers.cache_utils import DynamicCache
    from native_runtime import reset_sample, forward, phase_prefill, cache_report, cache_trace
    limit = sample['max_new_tokens']
    effective, mode = reset_sample(model, method, options, limit)
    ids = torch.tensor([sample['input_ids']], dtype=torch.long, device='cuda')
    length = ids.shape[1]
    if length + limit > model.config.max_position_embeddings:
        raise ValueError('Input/output exceeds original model context limit')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        if method == 'phasekv':
            logits, cache = phase_prefill(model, ids, options)
        else:
            cache = None if method == 'kivi' else DynamicCache()
            logits, cache, _ = forward(model, ids, cache, 0, method)
        token = logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - started
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Non-finite prefill logits')
        prefill_logits = logits[0, -1].cpu().tolist() if sample.get('save_prefill_logits') else None
        prefill_storage = cache_report(cache)
        prefill_trace = cache_trace(cache, method)
        tokens = [int(token.item())]
        stops = set(sample.get('stop_ids', [])) if sample.get('natural_eos') else set()
        decode_s = 0.
        # One timing interval per chunk; no per-token synchronize in fixed-length runs.
        chunk_start = time.perf_counter()
        pending = []
        for step in range(1, limit):
            if stops and tokens[-1] in stops:
                break
            logits, cache, _ = forward(model, token, cache, length + step - 1, method)
            if method == 'phasekv':
                cache.finish_decode_step()
            token = logits[:, -1].argmax(-1, keepdim=True)
            if stops:
                tokens.append(int(token.item()))
            else:
                pending.append(token)
            if step % 128 == 0 or step == limit - 1:
                torch.cuda.synchronize()
                decode_s += time.perf_counter() - chunk_start
                if progress:
                    print(f'[{method}/{sample["id"]}] decoded {step + 1}/{limit}', flush=True)
                chunk_start = time.perf_counter()
        torch.cuda.synchronize()
        decode_s += time.perf_counter() - chunk_start
        if pending:
            tokens.extend(torch.cat(pending, dim=1)[0].cpu().tolist())
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Non-finite final logits')
        report = {
            'input_tokens': length, 'output_tokens': len(tokens),
            'cached_generated_tokens': len(tokens) - 1,
            'prefill_and_compress_s': prefill_s, 'decode_s': decode_s,
            'decode_tokens_per_s': (len(tokens) - 1) / decode_s if len(tokens) > 1 else None,
            'prefill_storage': prefill_storage, 'final_storage': cache_report(cache),
            'prefill_trace': prefill_trace, 'final_trace': cache_trace(cache, method),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'effective_options': effective, 'scope_mode': mode,
            'output_ids': tokens, 'prefill_logits': prefill_logits,
        }
    return report


def worker(job_file):
    import torch
    import flash_attn
    import transformers
    import triton
    from native_runtime import load_model
    verify()
    job = json.loads(Path(job_file).read_text('utf-8'))
    method = job['method']
    out = Path(job['out'])
    config = json.loads((ROOT / 'config.json').read_text('utf-8'))
    if flash_attn.__version__ != config['flash_attn']:
        raise RuntimeError('FlashAttention version differs from the deployment lock')
    torch.manual_seed(job['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, options, origin = load_model(job['model'], method, tiny=job['tiny'])
    env = {'method': method, 'dtype': 'float16', 'batch_size': 1,
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'transformers': transformers.__version__,
           'flash_attn': flash_attn.__version__, 'triton': triton.__version__,
           'native_sources': origin, 'attention_classes': sorted({type(x.self_attn).__name__ for x in model.model.layers}),
           'rope_scaling': model.config.rope_scaling, 'rope_theta': model.config.rope_theta,
           'synthetic_tiny_validation': job['tiny']}
    dump(out / f'{method}.environment.json', env)
    for sample in job['samples']:
        for warm in range(job['warmups']):
            print(f'[{method}/{sample["id"]}] warmup {warm + 1}/{job["warmups"]}', flush=True)
            generate(model, method, options, sample, progress=False)
        for repeat in range(job['repeats']):
            torch.cuda.empty_cache()
            print(f'[{method}/{sample["id"]}] measured repeat {repeat + 1}', flush=True)
            base = {'method': method, 'sample_id': sample['id'], 'repeat': repeat,
                    'input_sha256': sha(json.dumps(sample['input_ids']).encode()),
                    'kind': job['kind'], 'dtype': 'float16'}
            try:
                result = generate(model, method, options, sample)
                row = {**base, 'status': 'ok', **result}
            except Exception as error:
                row = {**base, 'status': 'oom' if isinstance(error, torch.OutOfMemoryError) else 'error',
                       'error': repr(error), 'traceback': traceback.format_exc()}
                with (out / f'{method}.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                raise
            with (out / f'{method}.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f'[{method}] ok: {row["decode_tokens_per_s"]} token/s, '
                  f'KV={row["final_storage"]["physical_bytes"] / 2**20:.2f} MiB', flush=True)


def prepare_samples(args):
    if args.inputs:
        samples = [json.loads(line) for line in Path(args.inputs).read_text('utf-8').splitlines() if line.strip()]
        for sample in samples:
            if not sample.get('input_ids') or not sample.get('id'):
                raise ValueError('Each frozen sample needs id, input_ids, max_new_tokens')
            if any(type(x) is not int or x < 0 for x in sample['input_ids']):
                raise ValueError('input_ids must contain nonnegative integers')
            if type(sample.get('max_new_tokens')) is not int or sample['max_new_tokens'] < 1:
                raise ValueError('max_new_tokens must be positive')
        if not samples or len({s['id'] for s in samples}) != len(samples):
            raise ValueError('Frozen samples must be nonempty and have unique IDs')
        return samples, 'frozen_inputs_unscored'
    if args.tiny:
        ids = [1] + [3 + i % 250 for i in range(args.input_tokens - 1)]
        stop_ids = [2]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if args.prompt_file:
            text = Path(args.prompt_file).read_text('utf-8')
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                tokenize=True, add_generation_prompt=True)
        else:
            # Synthetic throughput diagnostic only, not a dataset or accuracy test.
            paragraph = tokenizer.encode('The library contains notes about science, history and mathematics. ',
                                         add_special_tokens=False)
            ids = [tokenizer.bos_token_id] + (paragraph * (args.input_tokens // len(paragraph) + 1))[:args.input_tokens - 1]
        stop_ids = [tokenizer.eos_token_id]
        generation = Path(args.model) / 'generation_config.json'
        if generation.exists():
            eos = json.loads(generation.read_text('utf-8')).get('eos_token_id', [])
            stop_ids += eos if isinstance(eos, list) else [eos]
        stop_ids = sorted({x for x in stop_ids if x is not None})
    return [{'id': 'deployment_smoke', 'input_ids': ids, 'max_new_tokens': args.new_tokens,
             'stop_ids': stop_ids, 'natural_eos': args.natural_eos,
             'save_prefill_logits': args.tiny}], ('synthetic_cuda_validation' if args.tiny else
                                                'manual_prompt_unscored' if args.prompt_file else 'synthetic_throughput')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Local complete Llama-3.1-8B-Instruct directory; no network download')
    parser.add_argument('--methods', nargs='+', default=['full', 'kivi', 'snapkv', 'scope', 'phasekv'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--input-tokens', type=int, default=2304)
    parser.add_argument('--new-tokens', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--tiny', action='store_true', help='2-layer random Llama3-RoPE GPU validation, not 8B results')
    parser.add_argument('--natural-eos', action='store_true')
    parser.add_argument('--prompt-file')
    parser.add_argument('--inputs', help='Frozen JSONL with id/input_ids/max_new_tokens/stop_ids/natural_eos')
    args = parser.parse_args()
    manifest = verify()
    cfg = json.loads((ROOT / 'config.json').read_text('utf-8'))
    if any(method not in cfg['methods'] for method in args.methods):
        parser.error('Unknown method')
    if min(args.input_tokens, args.new_tokens, args.repeats) < 1 or args.warmups < 0:
        parser.error('Lengths/repeats must be positive; warmups >= 0')
    if not args.tiny and (not args.model or not Path(args.model).is_dir()):
        parser.error('--model must name an existing local model directory')
    if not args.tiny:
        from check_model import check
        check(args.model)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is nonempty; choose a new run name to avoid mixing batches')
    out.mkdir(parents=True, exist_ok=True)
    samples, kind = prepare_samples(args)
    dump(out / 'frozen_inputs.json', samples)
    provenance = {'repositories': manifest['repositories'], 'sources_sha256': sha((ROOT / 'sources.json').read_bytes()),
                  'driver_sha256': {p.name: sha(p.read_bytes()) for p in ROOT.glob('*.py')},
                  'config': cfg, 'arguments': vars(args), 'kind': kind}
    if args.model:
        provenance['model_files'] = {p.name: {'size': p.stat().st_size,
                    'sha256': sha(p.read_bytes()) if p.suffix == '.json' else None}
                    for p in Path(args.model).iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors')}
    dump(out / 'run_manifest.json', provenance)
    statuses = []
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'TOKENIZERS_PARALLELISM': 'false'}
    for method in args.methods:
        job = {'method': method, 'model': args.model, 'tiny': args.tiny, 'seed': args.seed,
               'out': str(out), 'samples': samples, 'warmups': args.warmups,
               'repeats': args.repeats, 'kind': kind}
        job_path = out / 'jobs' / f'{method}.json'
        dump(job_path, job)
        print(f'=== {method}: log={out / (method + ".log")} ===', flush=True)
        with (out / f'{method}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '__worker__', str(job_path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', env=env)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        statuses.append({'method': method, 'exit_code': code})
        dump(out / 'status.json', statuses)
    print(f'Completed deployment run: {out}; failures are recorded, not replaced by other implementations.')
    return int(any(s['exit_code'] for s in statuses))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '__worker__':
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
