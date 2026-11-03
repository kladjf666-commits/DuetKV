"""Compare Full FA2 with prompt100/cold80 for 9192+256 and 8192+2048."""
import json

import full_speed


def campaign(model, options, base_sample, out, rounds, protocol):
    prior = json.loads((full_speed.HERE / 'quality_reference.json').read_text(encoding='utf-8'))
    options = dict(options, prefill_keep_ratio=1., decode_keep_ratio=.8,
                   optimizations=dict(options['optimizations'], gqa_reuse=True))
    if (options != prior['options'] or protocol['code_sha256'] != prior['speed_runtime_sha256']
            or protocol['environment'] != prior['environment']):
        raise RuntimeError('Runtime or PhaseKV options differ from the completed quality test')
    plan = json.loads((full_speed.HERE / 'speed_inputs.json').read_text(encoding='utf-8'))
    cases = plan['cases']
    if [(len(s['input_ids']), s['max_new_tokens']) for s in cases] != [(9192, 256), (8192, 2048)]:
        raise ValueError('Expected exactly 9192+256 and 8192+2048')
    for s in cases:
        if (s['natural_eos'] or s['input_ids'][:len(base_sample['input_ids'])] != base_sample['input_ids']
                or full_speed.quick.digest(json.dumps(s['input_ids']).encode()) != s['input_sha256']):
            raise ValueError(f'Unexpected speed input: {s["id"]}')
    path = out / 'summary.json'
    report = dict(status='running', gpu=protocol['gpu'], rounds=rounds,
                  frozen_samples_sha256=plan['frozen_samples_sha256'], cases={})
    for s in cases:
        shape = s['task']
        output = s['max_new_tokens']
        report['active_case'] = shape
        case_protocol = dict(protocol, input_id=s['id'], shape=shape, input_sha256=s['input_sha256'],
                             source_sample_id=s['source_sample_id'], input_tokens=len(s['input_ids']),
                             output_tokens=output, timing=f'{output-1} greedy decode steps after first token; no profiler')
        def save(case):
            report['cases'][shape] = case
            full_speed.quick.write_json(path, report)

        print(f'CASE {shape} | Full FA2 vs PhaseKV prompt100/cold80', flush=True)
        case = full_speed.measure_case(model, options, s, rounds, case_protocol, prior,
                                      ratios={'Full_FA2': None, 'PhaseKV_prompt100_cold80': .8}, save=save)
        windows = full_speed.chunk_summary(case)
        case['window_median_ms_per_token'] = windows
        save(case)
        for arm, values in windows.items():
            print(f'{shape} {arm}: {case["arm_medians"][arm]["tokens_per_s"]:.2f} token/s; '
                  f'first 128: {values["first_128"]:.3f}, last 127: {values["last_127"]:.3f} ms/token',
                  flush=True)
    report.pop('active_case')
    report['status'] = 'complete'
    full_speed.quick.write_json(path, report)
    print('FINISHED', path, flush=True)


if __name__ == '__main__':
    full_speed.main(campaign)
