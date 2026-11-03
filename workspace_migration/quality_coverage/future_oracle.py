"""Paired fixed-budget hindsight diagnostic, using only the teacher's future accesses."""
import argparse
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from quality_compare import HERE, generate_one, quick
from future_attention import FutureTrace, choose

ARMS = ('prompt70', 'prompt85', 'prompt100', 'future70')
TASKS = ('gsm8k', 'csqa', 'mmlu')


def measure(model, options, tokenizer, sample, control):
    import torch
    import ablation_kernels
    import ablation_runtime as rt
    import PhaseKV.prompt_tier as prompt

    teacher = control['prompt100']
    ids = control['teacher_ids']
    if (len(ids) != teacher['output_tokens'] or not ids
            or quick.digest(json.dumps(ids).encode()) != teacher['output_sha256']):
        raise ValueError('Archived teacher trajectory hash mismatch')
    layers = len(model.model.layers)
    length = len(sample['input_ids'])
    trace = FutureTrace(layers, length, len(ids)-1)
    native_decode = ablation_kernels.segmented_phasekv_decode

    def observed(q, tiers, heads):
        context = native_decode(q, tiers, heads)
        trace.observe(q, tiers)
        return context

    # Replay known tokens without projecting decode logits, choosing an output or scoring answers.
    with torch.inference_mode(), patch.object(ablation_kernels, 'segmented_phasekv_decode', observed):
        teacher_options = dict(options, prefill_keep_ratio=1.)
        rt.reset_sample(model, 'phasekv_triton', teacher_options, sample['max_new_tokens'])
        prompt_ids = torch.tensor([sample['input_ids']], dtype=torch.long, device='cuda')
        logits, cache = rt.phase_prefill(model, prompt_ids, teacher_options)
        if int(logits[:, -1].argmax()) != ids[0] or cache.layers[0].prompt.length != length:
            raise ValueError('Teacher prefill differs from archived trajectory')
        tokens = torch.tensor([ids], dtype=torch.long, device='cuda')
        positions = torch.arange(length, length+len(ids)-1, device='cuda')
        for step in range(len(ids)-1):
            pos = positions[step:step+1]
            model.model(input_ids=tokens[:, step:step+1], past_key_values=cache,
                        position_ids=pos[None], cache_position=pos, use_cache=True, return_dict=True)
            cache.finish_decode_step()
        del cache, logits
    scores = trace.finish(len(ids)-1) if len(ids)>1 else []
    standard = prompt.select_prompt_indices
    audits = []

    def fixed(attentions, **settings):
        original = standard(attentions, **settings)
        if audits or settings['obs_window'] != 32 or settings['prefill_keep_ratio'] != .7:
            raise RuntimeError('Unexpected selection protocol')
        result = []
        for idx, old in enumerate(original):
            if not scores:  # First token already ended generation; no future observations.
                result.append(old)
                continue
            keep, mass = choose(scores[idx], old, min(32, old.shape[-1]))
            result.append(keep)
            audits.append(mass)
        return result

    # generate_one -> generate_standard still audits every packed layer/head and budget.
    with patch.object(prompt, 'select_prompt_indices', fixed):
        candidate = generate_one(model, options, sample, None, tokenizer, 'prompt70')
    candidate['arm'] = 'future70'
    if scores and len(audits) != layers:
        raise RuntimeError('Fixed selector did not run at every layer')
    mass = torch.stack(audits).mean(dim=(0, 2)).tolist() if audits else [None, None]
    candidate['teacher_steps'] = len(ids)-1
    candidate['teacher_output_sha256'] = teacher['output_sha256']
    candidate['teacher_prompt_mass_retained'] = dict(original70=mass[0], future70=mass[1])
    return candidate


def summarize(pairs):
    results = {}
    for task in TASKS:
        group = [p for p in pairs if p['task'] == task]
        if not group:
            continue
        stats = {arm: {field: statistics.mean(p['arms'][arm][field] for p in group)
                       for field in ('score', 'indexed_accuracy_diagnostic', 'completion_pct', 'severe_loop')}
                 for arm in ARMS}
        metric = 'indexed_accuracy_diagnostic'
        gain = stats['future70'][metric] - stats['prompt70'][metric]
        gap = stats['prompt85'][metric] - stats['prompt70'][metric]
        results[task] = dict(paired_prompts=len(group), arms=stats, indexed_gain_pp=gain,
                             indexed_gap_to_85_pp=stats['prompt85'][metric]-stats['future70'][metric],
                             fraction_of_70_to_85_gap_recovered=gain/gap if gap > 0 else None)
    return results


def campaign(model, options, unused, out, rounds, base):
    from transformers import AutoTokenizer
    with gzip.open(HERE/'oracle_reference.json.gz', 'rt', encoding='utf8') as f:
        ref = json.load(f)
    with gzip.open(HERE/'quality_samples.json.gz', 'rt', encoding='utf8') as f:
        samples = json.load(f)
    options = dict(options, optimizations=dict(options['optimizations'], gqa_reuse=True))
    if (options != ref['options'] or base['environment'] != ref['environment']
            or base['code_sha256'] != ref['code_sha256']):
        raise ValueError('Runtime differs from frozen 4090 protocol')
    if Counter(s['task'] for s in samples) != {t: 16 for t in TASKS} or len({s['id'] for s in samples}) != 48:
        raise ValueError('Incomplete input manifest')
    for s in samples:
        if (quick.digest(json.dumps(s['input_ids']).encode()) != s['input_sha256']
                or s['max_new_tokens'] != 4096 or not s['natural_eos'] or not s['stop_ids']):
            raise ValueError(f'Invalid frozen prompt {s["id"]}')
    path = Path(base['model'])
    for name, expected in ref['tokenizer_sha256'].items():
        if quick.digest((path/name).read_bytes()) != expected:
            raise ValueError(f'Tokenizer mismatch: {name}')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    for layer in model.model.layers:
        layer.self_attn.method = 'phasekv_triton'
    protocol = dict(gpu=base['gpu'], model=str(path), environment=base['environment'], options=options,
                    runtime=base['code_sha256'], package=quick.digest((HERE/'SHA256SUMS').read_bytes()),
                    generated_arms=['future70'], archived_arms=list(ARMS[:3]),
                    archived_fingerprints=ref['fingerprints'], scan_fingerprint=ref['scan_fingerprint'],
                    same_gpu_as_controls=base['gpu'].split(',')[0]==ref['gpu'].split(',')[0],
                    teacher='Forced replay of archived prompt100 free-generation tokens on packed 4-bit KV',
                    selector='Sum full-softmax prompt probabilities over all decode steps and GQA siblings; '
                             'top positions per layer/KV head, keep final 32 within round(.70*N).',
                    primary='Indexed accuracy; recovered fraction = (future70 - prompt70)/(prompt85 - prompt70) '
                            'using paired task means, undefined for nonpositive reference gain.')
    fp = quick.digest(json.dumps(protocol, sort_keys=True).encode())
    records = out/'records'
    manifest = out/'protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint'] != fp:
            raise ValueError('Different protocol; set PHASEKV_OUT to a fresh directory')
    elif records.exists() or (out/'summary.json').exists():
        raise ValueError('Saved results have no manifest')
    quick.write_json(manifest, dict(fingerprint=fp, protocol=protocol))
    records.mkdir(exist_ok=True)
    pairs = []

    def save(status, error=None):
        quick.write_json(out/'summary.json', dict(status=status, fingerprint=fp, planned_prompts=len(samples),
                         archived_control_records=3*len(samples), new_free_generations=len(pairs),
                         completed_prompts=len(pairs), by_task=summarize(pairs), error=error))

    # Interleave tasks so partially completed runs contain paired results on each task.
    groups = [sorted((s for s in samples if s['task']==t), key=lambda s: int(s['id'].split('/')[-1])) for t in TASKS]
    try:
        for group in zip(*groups):
            for s in group:
                control = ref['controls'][s['id']]
                record = records/(s['id'].replace('/', '_')+'.json')
                if record.exists():
                    item = json.loads(record.read_text(encoding='utf8'))
                else:
                    arms = {arm: control[arm] for arm in ARMS[:3]}
                    arms['future70'] = measure(model, options, tokenizer, s, control)
                    item = dict(id=s['id'], task=s['task'], fingerprint=fp, arms=arms)
                if (item['fingerprint'] != fp or item['id'] != s['id'] or item['task'] != s['task']
                        or set(item['arms']) != set(ARMS)):
                    raise ValueError(f'Invalid record for {s["id"]}')
                for arm, pct in zip(ARMS, (70, 85, 100, 70)):
                    r = item['arms'][arm]
                    if (r['input_sha256'] != s['input_sha256'] or r['id'] != s['id']
                            or r['retained_per_head'] != round(pct/100 * len(s['input_ids']))):
                        raise ValueError(f'Record verification failed: {s["id"]} {arm}')
                    if arm != 'future70':
                        if r != control[arm]:
                            raise ValueError('Historical control differs from manifest')
                    elif (r['arm'] != arm or quick.digest(json.dumps(r['output_ids']).encode()) != r['output_sha256']
                          or r['teacher_output_sha256'] != control['prompt100']['output_sha256']
                          or r['teacher_steps'] != control['prompt100']['output_tokens']-1):
                        raise ValueError('Invalid candidate or mismatched teacher')
                if not record.exists():
                    quick.write_json(record, item)
                pairs.append(item)
                save('running')
                print(f'Paired prompts: {len(pairs)}/48, {s["id"]}', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        save('failed', str(exc))
        raise
    save('complete')
    print('FINISHED', out/'summary.json', flush=True)


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.rounds = 1
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        quick.benchmark(args, args.out, expected_gpu='NVIDIA GeForce RTX 4090', run_campaign=campaign)
