"""Lightweight host timing around first cooling, paired with uninstrumented runs."""
import argparse
import contextlib
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch


WINDOWS = {'reference129': (129, 129), 'before': (130, 143),
           'cooling144': (144, 144), 'first_cold145': (145, 145), 'after': (146, 159)}
STAGES = ('cache_append', 'cache_finish', 'workspace')
DOMAINS = ('host_elapsed_ms', 'thread_cpu_ms')
CONTRASTS = (('before', 'after'), ('before', 'cooling144'), ('reference129', 'first_cold145'))


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def clock_details():
    result = {}
    for domain, name in zip(DOMAINS, ('perf_counter', 'thread_time')):
        info = time.get_clock_info(name)
        clock = getattr(time, name + '_ns')
        samples = [clock() for _ in range(8192)]
        positive = [end - start for start, end in zip(samples, samples[1:]) if end > start]
        minimum = min(positive) if positive else None
        result[domain] = dict(implementation=info.implementation, resolution_s=info.resolution,
                             observed_min_increment_ns=minimum, positive_increments=len(positive),
                             suitable_for_short_calls=info.resolution <= 1e-6 and
                             minimum is not None and minimum <= 10000)
    return result


class HostTimer:
    """Three disjoint call scopes; no CUDA operations or profiler callbacks."""
    def __init__(self):
        self.slot = 0
        self.elapsed = [[0, 0, 0] for _ in range(31)]
        self.cpu = [[0, 0, 0] for _ in range(31)]
        self.counts = [[0, 0, 0] for _ in range(31)]

    def wrap(self, function, stage):
        wall_clock, cpu_clock = time.perf_counter_ns, time.thread_time_ns

        def timed(*args, **kwargs):
            started = wall_clock()
            cpu_started = cpu_clock()
            result = function(*args, **kwargs)
            cpu_ns = cpu_clock() - cpu_started
            elapsed_ns = wall_clock() - started
            # Bookkeeping is outside the recorded function duration.
            slot = self.slot
            self.elapsed[slot][stage] += elapsed_ns
            self.cpu[slot][stage] += cpu_ns
            self.counts[slot][stage] += 1
            return result
        return timed

    @contextlib.contextmanager
    def installed(self):
        from PhaseKV.phasekv_cache import PhaseKVCache
        import PhaseKV.kernels.segmented_attention as segmented
        with contextlib.ExitStack() as stack:
            for owner, name, stage in ((PhaseKVCache, 'append_hot', 0),
                                      (PhaseKVCache, 'finish_decode_step', 1),
                                      (segmented, 'ensure_workspace', 2)):
                stack.enter_context(patch.object(owner, name, self.wrap(getattr(owner, name), stage)))
            yield

    def summarize(self, layers):
        steps = {}
        for slot in range(31):
            if self.counts[slot] != [layers, 1, layers]:
                raise RuntimeError(f'Incomplete host timing at decode {slot + 129}: {self.counts[slot]}')
            row = {'counts': dict(zip(STAGES, self.counts[slot]))}
            for domain, values in zip(DOMAINS, (self.elapsed, self.cpu)):
                row[domain] = {name: values[slot][i] / 1e6 for i, name in enumerate(STAGES)}
                row[domain]['cache_total'] = sum(row[domain].values())
            steps[str(slot + 129)] = row
        windows = {name: {domain: {stage: statistics.mean(
            steps[str(step)][domain][stage] for step in range(start, end + 1))
            for stage in (*STAGES, 'cache_total')} for domain in DOMAINS}
            for name, (start, end) in WINDOWS.items()}
        return {'steps': steps, 'windows': windows}


def generation(model, sample, options, instrumented):
    import torch
    import ablation_runtime as rt

    method = 'phasekv_triton'
    for layer in model.model.layers:
        layer.self_attn.method = method
    rt.reset_sample(model, method, options, 256)
    ids = torch.tensor([sample['input_ids']], device='cuda')
    logits, cache = rt.phase_prefill(model, ids, options)
    token = logits[:, -1].argmax(-1, keepdim=True)
    generated = [token]

    def advance(step):
        nonlocal token
        logits, _, _ = rt.forward(model, token, cache, ids.shape[1] + step - 1, method)
        cache.finish_decode_step()
        token = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token)

    for step in range(1, 129):
        advance(step)
    torch.cuda.synchronize()
    timer = HostTimer() if instrumented else None
    wall, states = {}, []
    # Both arms synchronize only at identical window boundaries, never in a timed call.
    with timer.installed() if timer is not None else contextlib.nullcontext():
        for name, (start, end) in WINDOWS.items():
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            if timer is None:
                for step in range(start, end + 1):
                    advance(step)
            else:
                for step in range(start, end + 1):
                    timer.slot = step - 129
                    advance(step)
            torch.cuda.synchronize()
            wall[name] = (time.perf_counter_ns() - started) / 1e6 / (end - start + 1)
            expected_cold, expected_hot = max(0, (end - 128) // 16), 128 + (end - 128) % 16
            for layer in cache.layers:
                if len(layer.cold) != expected_cold or layer.hot_len != expected_hot:
                    raise RuntimeError(f'Unexpected cold/hot state after decode {end}')
            states.append(dict(end_step=end, cold_segments=expected_cold, hot_tokens=expected_hot))
    for step in range(160, 256):
        advance(step)
    torch.cuda.synchronize()
    for layer in cache.layers:
        if (len(layer.cold) != 7 or layer.hot_len != 143
                or any(segment.length != 13 for segment in layer.cold)
                or not layer.optimizations.gqa_reuse
                or not layer.optimizations.channel_contiguous_k
                or not layer._decode_workspace.compatible_4bit):
            raise RuntimeError('Final cache differs from the GQA-on keep80 reference')
    output = torch.cat(generated, dim=1)[0].tolist()
    if len(output) != 256:
        raise RuntimeError('Expected exactly 256 output tokens')
    row = dict(arm='host_timed' if instrumented else 'plain', wall_ms=wall,
               output_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest(), cache_states=states)
    if timer is not None:
        row.update(timer.summarize(len(model.model.layers)))
    return row


def aggregate_wall(wall):
    result = dict(wall)
    result['stable'] = (wall['before'] + wall['after']) / 2
    result['all31'] = sum(wall[name] * (end - start + 1)
                          for name, (start, end) in WINDOWS.items()) / 31
    return result


def summarize(runs):
    timed = [row for row in runs if row['arm'] == 'host_timed']
    paired = []
    for cpu in timed:
        plain, = [row for row in runs if row['pair'] == cpu['pair'] and row['arm'] == 'plain']
        a, b = aggregate_wall(plain['wall_ms']), aggregate_wall(cpu['wall_ms'])
        paired.append(dict(pair=cpu['pair'], order=cpu['order'], windows={name: {
            'plain_ms': a[name], 'timed_ms': b[name], 'delta_ms': b[name] - a[name],
            'delta_pct': 100 * (b[name] / a[name] - 1)} for name in a}))
    overhead = {name: {metric: statistics.median(row['windows'][name][metric] for row in paired)
                      for metric in ('plain_ms', 'timed_ms', 'delta_ms', 'delta_pct')}
                for name in paired[0]['windows']}
    # These empirical tolerances are not a statistical guarantee. Keep signed pair data.
    checks = {name: {'limit_abs_median_pct': 5.0 if name in
                    ('reference129', 'cooling144', 'first_cold145') else 3.0,
                    'observed_median_pct': row['delta_pct']} for name, row in overhead.items()}
    for row in checks.values():
        row['within_budget'] = abs(row['observed_median_pct']) <= row['limit_abs_median_pct']
    medians = {domain: {name: {stage: statistics.median(row['windows'][name][domain][stage] for row in timed)
                            for stage in (*STAGES, 'cache_total')} for name in WINDOWS} for domain in DOMAINS}
    contrasts = {after + '_minus_' + before: {domain: {stage: {
        'per_pair_ms': [row['windows'][after][domain][stage] - row['windows'][before][domain][stage]
                        for row in timed],
        'median_ms': statistics.median(row['windows'][after][domain][stage] -
                                       row['windows'][before][domain][stage] for row in timed)}
        for stage in (*STAGES, 'cache_total')} for domain in DOMAINS} for before, after in CONTRASTS}
    return dict(paired_overhead=paired, overhead_median=overhead, overhead_checks=checks,
                perturbation_check='within_budget' if all(row['within_budget'] for row in checks.values())
                else 'exceeds_budget_review_before_interpreting',
                medians=medians, contrasts=contrasts,
                wall_median_ms={arm: {name: statistics.median(
                    row['wall_ms'][name] for row in runs if row['arm'] == arm) for name in WINDOWS}
                    for arm in ('plain', 'host_timed')})


def campaign(model, sample, options, historical, out, pairs, clocks=None):
    reference, = {row['output_sha256'] for row in historical['runs']
                   if row['arm'] == 'PhaseKV_GQA_on' and not row['warmup']}
    report = dict(status='running', protocol=dict(
        base=historical['protocol'], gqa_reuse=True, layers=len(model.model.layers), pairs=pairs,
        windows=WINDOWS, stages=STAGES, profiler='none',
        clocks=clock_details() if clocks is None else clocks,
        definition='Decode 144 cools after attention; decode 145 first reads the cold block. '
                   '130-143 and 146-159 have matched hot lengths.',
        host_elapsed='Disjoint host function elapsed durations, including any CUDA API blocking or descheduling. '
                     'No synchronization is inserted inside calls. CPU and GPU time must not be added.',
        thread_cpu='CPU time consumed by the calling thread inside the same functions; excludes waiting '
                   'and work on other threads. Elapsed minus thread CPU is not necessarily GPU waiting.',
        instrumentation='Only append_hot, finish_decode_step, ensure_workspace are wrapped for decode 129-159. '
                        'Two clocks and preallocated counters; no activity tracing, events or profiler callbacks.',
        control='Fresh identical generation/cache for each arm, alternating plain/timed and timed/plain. '
                'One warmup per arm, excluded. Window-boundary synchronization is identical in both arms.',
        perturbation_policy='Absolute paired median wall difference <=3% for multi-step windows, <=5% '
                            'for single-step windows. Negative differences indicate timing variability. '
                            'This is an empirical check, not proof of zero instrumentation bias.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), warmups=[], runs=[])
    path = out / 'cpu_summary.json'
    write_json(path, report)
    for instrumented in (False, True):
        row = generation(model, sample, options, instrumented)
        if row['output_sha256'] != reference:
            raise RuntimeError('Warmup output differs from the original 4090D speed test')
        report['warmups'].append({key: row[key] for key in ('arm', 'wall_ms', 'output_sha256')})
        write_json(path, report)
        print(f'WARMUP {row["arm"]}: original-output parity passed.', flush=True)
    for pair in range(pairs):
        order = (False, True) if pair % 2 == 0 else (True, False)
        names = ['host_timed' if value else 'plain' for value in order]
        for instrumented in order:
            row = generation(model, sample, options, instrumented)
            if row['output_sha256'] != reference:
                raise RuntimeError('Measured output differs from the original 4090D speed test')
            report['runs'].append(dict(pair=pair + 1, order=names, **row))
            write_json(path, report)
        report['analysis'] = summarize(report['runs'])
        write_json(path, report)
        overhead = report['analysis']['paired_overhead'][-1]['windows']['stable']
        print(f'PAIR {pair + 1}/{pairs}: plain {overhead["plain_ms"]:.3f}, '
              f'host-timed {overhead["timed_ms"]:.3f} ms/token; '
              f'paired delta {overhead["delta_ms"]:+.3f} ms ({overhead["delta_pct"]:+.2f}%)', flush=True)
    report['status'] = 'complete'
    write_json(path, report)
    print('PERTURBATION_CHECK', report['analysis']['perturbation_check'], flush=True)
    for domain in DOMAINS:
        for label, stages in report['analysis']['medians'][domain].items():
            print(domain, label, ' '.join(f'{key}={value:.4f}' for key, value in stages.items()), flush=True)
    print('FINISHED', path, flush=True)


def benchmark(args):
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    import quick
    import torch
    import native_runtime
    import ablation_runtime as rt
    import ablation_kernels
    import PhaseKV.phasekv_cache as packed_cache
    from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
    from check_model import check

    if not Path(packed_cache.__file__).resolve().is_relative_to(native_runtime.RUNTIME):
        raise RuntimeError('PhaseKV imported from outside the frozen speed package')
    sample, checksum = quick.verify_bundle()
    historical = json.loads(args.history.read_text(encoding='utf-8'))
    if historical['status'] != 'complete':
        raise RuntimeError('Prior speed test is incomplete')
    base = historical['protocol']
    versions, device = quick.gpu_environment(torch)
    if (checksum != base['code_sha256'] or versions != base['environment']
            or device.split(',')[0] != base['gpu'].split(',')[0]
            or sample['input_sha256'] != base['input_sha256']):
        raise RuntimeError('GPU, environment, code or input differs from the original speed test')
    model_path = Path(base['model'])
    check(model_path)
    for name, expected in json.loads((project / 'model_files.json').read_text()).items():
        if hashlib.sha256((model_path / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Checkpoint metadata mismatch: {name}')
    options = dict(base['options_gqa_off'])
    options['optimizations'] = dict(options['optimizations'], gqa_reuse=True)
    clocks = clock_details()
    if not all(clock['suitable_for_short_calls'] for clock in clocks.values()):
        raise RuntimeError(f'Clock resolution is insufficient for short CPU scopes: {clocks}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f'{device} | frozen 8192+256 | loading all 32 Llama layers once', flush=True)
    print('Host timing only. No activity profiler. Pairs:', args.pairs, flush=True)
    with torch.inference_mode(), patch.object(ablation_kernels, 'segmented_phasekv_decode', segmented_phasekv_decode):
        model, _, _ = rt.load_model(model_path, 'full_fa2')
        if len(model.model.layers) != 32:
            raise RuntimeError('Expected all 32 Llama layers')
        print('MODEL_READY', flush=True)
        campaign(model, sample, options, historical, args.out, args.pairs, clocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/data/phasekv_4090d_speed'))
    parser.add_argument('--history', type=Path, default=Path(
        '/data/phasekv-results/4090d_speed_20261101_221816_361/summary.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs', type=int, choices=(2, 4, 6), default=4)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'run.claim').open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        benchmark(args)
    except (Exception, KeyboardInterrupt) as exc:
        path = args.out / 'cpu_summary.json'
        report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(path, report)
        raise


if __name__ == '__main__':
    main()
