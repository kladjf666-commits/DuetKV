"""Two matched 2048-token decode workloads on one loaded model and one GPU."""
import json

import full_speed
from full_speed import HERE, chunk_summary, quick


def samples(base, specs):
    if [s['input_tokens'] for s in specs] != [1024, 8192]:
        raise ValueError('Expected the short and long frozen prompt pair')
    for spec in specs:
        length = spec['input_tokens']
        ids = base['input_ids'][:length]
        if len(ids) != length or quick.digest(json.dumps(ids).encode()) != spec['input_sha256']:
            raise RuntimeError(f'Frozen {length}-token input differs from its source')
        shape = f'{length}+2048'
        yield dict(base, id=spec['id'], task=shape, cluster_id=spec['id'],
                   input_ids=ids, input_sha256=spec['input_sha256'], length_budget=length,
                   max_new_tokens=2048, natural_eos=False)


def campaign(model, options, base_sample, out, rounds, base_protocol):
    plan = json.loads((HERE / 'long_cases.json').read_text(encoding='utf-8'))
    workloads = list(samples(base_sample, plan['cases']))
    prior = plan['preceding_run']
    root = dict(status='running', gpu=base_protocol['gpu'], rounds=rounds,
                same_gpu_as_previous_run=base_protocol['gpu'].split(',')[0] == prior['gpu_uuid'],
                frozen_samples_sha256=plan['frozen_samples_sha256'], cases={})
    path = out / 'summary.json'
    for sample, spec in zip(workloads, plan['cases']):
        shape = sample['task']
        root['active_case'] = shape
        protocol = dict(base_protocol, shape=shape, input_id=sample['id'],
                        input_sha256=sample['input_sha256'], source_sample_id=spec['source_id'],
                        input_tokens=len(sample['input_ids']), output_tokens=2048,
                        timing='All 2047 greedy decode steps; prefill excluded; no profiler')

        def save(case):
            root['cases'][shape] = case
            quick.write_json(path, root)

        print(f'CASE {shape} | three arms | {rounds} formal rounds each', flush=True)
        case = full_speed.measure_case(model, options, sample, rounds, protocol, prior, save=save)
        case['window_median_ms_per_token'] = chunk_summary(case)
        save(case)
        print(f'CASE_DONE {shape}', flush=True)
    root.pop('active_case')
    root['status'] = 'complete'
    quick.write_json(path, root)
    print('FINISHED', path, flush=True)


if __name__ == '__main__':
    full_speed.main(campaign)
