"""One frozen 8K+256 prompt, Full FA2 and PhaseKV keep80 with GQA off/on."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
ARMS = ('Full_FA2', 'PhaseKV_GQA_off', 'PhaseKV_GQA_on')
EXPECTED_INPUT = '91d7a9999c6c84ae4ea01ffde7954083f05e1043e63a755d9cfda6183a5e0046'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def verify_bundle():
    manifest = json.loads((ROOT / 'files.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise RuntimeError(f'Upload package checksum mismatch: {name}')
    sample = json.loads((ROOT / 'sample.json').read_text(encoding='utf-8'))
    if (sample['id'] != 'performance/8192+256/0' or len(sample['input_ids']) != 8192
            or sample['max_new_tokens'] != 256 or sample['natural_eos']
            or digest(json.dumps(sample['input_ids']).encode()) != EXPECTED_INPUT
            or sample['input_sha256'] != EXPECTED_INPUT):
        raise ValueError('Frozen speed prompt mismatch')
    return sample, digest((ROOT / 'files.json').read_bytes())


def gpu_environment(torch, expected_gpu='NVIDIA GeForce RTX 4090 D'):
    import flash_attn
    import transformers
    import triton
    versions = dict(torch=str(torch.__version__), flash_attn=flash_attn.__version__,
                    transformers=transformers.__version__, triton=triton.__version__)
    if versions != dict(torch='2.5.1+cu121', flash_attn='2.7.4.post1', transformers='4.46.3', triton='3.1.0'):
        raise RuntimeError(f'Use the previously prepared native benchmark Python environment. Found {versions}')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'Expose exactly one {expected_gpu} using CUDA_VISIBLE_DEVICES.')
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name.replace(' ', '').lower() != expected_gpu.replace(' ', '').lower():
        raise RuntimeError(f'Expected {expected_gpu}, got {gpu.name}')
    def smi(fields, kind):
        return subprocess.check_output(
            ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader'], text=True).splitlines()
    devices = smi('uuid,name,driver_version', 'gpu')
    # A single visible card is sufficient even in a multi-GPU host.
    uuid = str(gpu.uuid) if getattr(gpu, 'uuid', None) else None
    selected = [line for line in devices if uuid and uuid.lower().removeprefix('gpu-') in line.lower()]
    if not selected and len(devices) == 1:
        selected = devices
    if len(selected) != 1:
        raise RuntimeError(f'Cannot identify the visible GPU: torch UUID {uuid}, nvidia-smi {devices}')
    selected_uuid = selected[0].split(',')[0].strip()
    for line in smi('gpu_uuid,pid', 'compute-apps'):
        active_uuid, _, pid = line.partition(',')
        if active_uuid.strip() == selected_uuid and pid.strip() != str(os.getpid()):
            raise RuntimeError(f'Selected GPU is busy (PID {pid.strip()}).')
    return versions, selected[0]


def measure(model, arm, options, sample):
    import ablation_runtime as rt
    from ablation_generate import generate

    packed = arm != ARMS[0]
    method = 'phasekv_triton' if packed else 'full_fa2'
    reuse = arm == ARMS[2]
    settings = dict(options, optimizations=dict(options['optimizations'], gqa_reuse=reuse)) if packed else {}
    for layer in model.model.layers:
        layer.self_attn.method = method
    trace = rt.cache_trace

    def audited(cache, path):
        result = trace(cache, path)
        if packed:
            kept = max(1, round(settings['lazy_g'] * settings['decode_keep_ratio']))
            for layer in cache.layers:
                opt = layer.optimizations
                if (opt.gqa_reuse != reuse or not opt.channel_contiguous_k or not opt.specialized_4bit
                        or (layer._decode_workspace is not None and not layer._decode_workspace.compatible_4bit)
                        or any(seg.length != kept for seg in layer.cold)):
                    raise RuntimeError('Unexpected PhaseKV execution path')
            result['cold_tokens'] = sum(segment.length for segment in cache.layers[0].cold)
        return result

    with patch.object(rt, 'cache_trace', audited):
        row = generate(model, method, settings, sample)
    steps = row['output_tokens'] - 1
    expected_steps = sample['max_new_tokens'] - 1
    if steps != expected_steps:
        raise RuntimeError(f'Expected {expected_steps} decode steps, got {steps}')
    if packed and row['final_trace']['cold_segments'] != max(0, (steps - settings['recent_window']) // settings['lazy_g']):
        raise RuntimeError('Unexpected decode length or cold-segment count')
    return dict(arm=arm, decode_steps=steps, ms_per_token=1000 * row['decode_s'] / steps,
                tokens_per_s=row['decode_tokens_per_s'], ttft_s=row['ttft_s'],
                decode_chunks=row['decode_chunks'], final_trace=row['final_trace'],
                final_kv_bytes=row['final_storage']['physical_bytes'],
                output_sha256=digest(json.dumps(row['output_ids']).encode()))


def summarize(rounds):
    median = statistics.median
    speeds = {arm: median(r[arm]['tokens_per_s'] for r in rounds) for arm in ARMS}
    ratios = [r[ARMS[2]]['tokens_per_s'] / r[ARMS[1]]['tokens_per_s'] for r in rounds]
    vs_full = [r[ARMS[2]]['tokens_per_s'] / r[ARMS[0]]['tokens_per_s'] for r in rounds]
    return dict(complete_rounds=len(rounds), median_tokens_per_s=speeds,
                median_ms_per_token={a: median(r[a]['ms_per_token'] for r in rounds) for a in ARMS},
                gqa_vs_off_paired_pct=100 * (median(ratios) - 1),
                gqa_faster_rounds=sum(x > 1 for x in ratios),
                gqa_vs_full_paired_pct=100 * (median(vs_full) - 1))


def campaign(model, options, sample, out, repeats, protocol):
    rows, rounds, signatures = [], [], {}
    state = dict(protocol=protocol, requested_rounds=repeats, status='running', runs=rows)
    path = out / 'summary.json'
    write_json(path, state)
    for index in range(repeats + 1):
        offset = max(0, index - 1) % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        group = {}
        for arm in order:
            result = measure(model, arm, options, sample)
            group[arm] = result
            rows.append(dict(round=index, warmup=index == 0, **result))
            write_json(path, state)
            if result['output_sha256'] != signatures.setdefault(arm, result['output_sha256']):
                raise RuntimeError(f'Greedy output varied across repeats for {arm}')
            print(f'{"warm-up" if not index else f"round {index}"} {arm}: '
                  f'{result["tokens_per_s"]:.2f} token/s, {result["ms_per_token"]:.3f} ms/token', flush=True)
        if group[ARMS[1]]['output_sha256'] != group[ARMS[2]]['output_sha256']:
            raise RuntimeError('PhaseKV GQA off/on greedy outputs differ for this prompt')
        if index:
            rounds.append(group)
            state['comparison'] = summarize(rounds)
            s = state['comparison']
            print(f'PAIRED {index}/{repeats}: GQA vs off {s["gqa_vs_off_paired_pct"]:+.2f}%, '
                  f'GQA vs Full {s["gqa_vs_full_paired_pct"]:+.2f}%, '
                  f'GQA wins {s["gqa_faster_rounds"]}/{index}', flush=True)
            state['status'] = 'complete' if index == repeats else 'running'
            write_json(path, state)
    print(f'FINISHED {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, choices=(1, 3, 6), default=3)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive claim prevents two launchers from mixing records in one run.
    with (out / 'run.claim').open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        benchmark(args, out)
    except (Exception, KeyboardInterrupt) as exc:
        path = out / 'summary.json'
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(path, state)
        raise


def benchmark(args, out, *, expected_gpu='NVIDIA GeForce RTX 4090 D', run_campaign=campaign):
    sample, checksum = verify_bundle()
    import torch
    import native_runtime  # puts the release's private runtime first
    import ablation_kernels
    import ablation_runtime as rt
    from check_model import check
    from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
    import PhaseKV.phasekv_cache as packed_cache

    if not Path(packed_cache.__file__).resolve().is_relative_to(native_runtime.RUNTIME):
        raise RuntimeError('PhaseKV imported from outside the upload package')
    versions, device = gpu_environment(torch, expected_gpu)
    check(args.model)
    for name, expected in json.loads((ROOT / 'model_files.json').read_text()).items():
        if digest((args.model / name).read_bytes()) != expected:
            raise ValueError(f'Checkpoint does not match the frozen benchmark: {name}')
    model_config = json.loads((ROOT / 'config.json').read_text())['methods']['phasekv']
    options = dict(model_config, decode_keep_ratio=.8,
                   optimizations=dict(model_config['optimizations'], channel_contiguous_k=True))
    protocol = dict(gpu=device, environment=versions, code_sha256=checksum, input_id=sample['id'],
                    input_sha256=EXPECTED_INPUT, model=str(args.model.resolve()), shape='8192+256',
                    batch=1, dtype='float16', options_gqa_off=options, warmup_per_arm=1,
                    timing='Entire 255 greedy decode steps; prefill and prompt packing excluded; no profiler')
    print(f'{device} | loading full Llama-3.1-8B once | results {out}', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode(), patch.object(ablation_kernels, 'segmented_phasekv_decode', segmented_phasekv_decode):
        model, _, _ = rt.load_model(args.model, 'full_fa2')
        if len(model.model.layers) != 32:
            raise RuntimeError('Expected all 32 Llama layers')
        run_campaign(model, options, sample, out, args.rounds, protocol)


if __name__ == '__main__':
    main()
