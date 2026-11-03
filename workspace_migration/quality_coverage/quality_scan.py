"""Sweep standard PhaseKV prompt retention on the same 48 frozen LongGenBench prompts."""
import argparse
from collections import Counter
import gzip
import json
from math import isclose
from pathlib import Path

from quality_compare import HERE, generate_one, quick
from retention_curve import GRID, ORDER, TASKS, analyze, long_repeat


def brief(row):
    return {k:v for k,v in row.items() if k not in ('output_ids','prediction','decode_chunks')}


def validate_row(row, sample, fingerprint):
    pct = row['ratio_pct']
    output = row['output_ids']
    chunks = row['decode_chunks']
    steps = len(output)-1
    sec = row['decode_s']
    if (pct not in ORDER or row['arm'] != f'prompt{pct}' or row['fingerprint'] != fingerprint
            or row['task'] != sample['task'] or row['input_sha256'] != sample['input_sha256']
            or row['total_questions'] != len(sample['answers']) or row['prompt_tokens'] != len(sample['input_ids'])
            or row['retained_per_head'] != round((pct/100)*row['prompt_tokens'])
            or row['selection']['verified_layer_heads'] != 256 or row['layer_count'] != 32
            or not 0 < len(output) <= sample['max_new_tokens'] or row['output_tokens'] != len(output)
            or row['stopped_on_eos'] != (output[-1] in sample['stop_ids'])
            or row['hit_output_limit'] != (len(output)==sample['max_new_tokens'] and not row['stopped_on_eos'])
            or not (row['stopped_on_eos'] or row['hit_output_limit'])
            or quick.digest(json.dumps(output).encode()) != row['output_sha256']
            or row['decode_steps'] != steps or sum(c['tokens'] for c in chunks) != steps
            or not isclose(sum(c['seconds'] for c in chunks), sec)
            or steps and (sec <= 0 or not isclose(row['decode_tokens_per_s'], steps/sec))):
        raise ValueError(f'Invalid record {row["id"]} at {pct}%')


def run_scan(model, options, tokenizer, samples, reference, out, protocol, coarse):
    fingerprint = quick.digest(json.dumps(protocol,sort_keys=True).encode())
    manifest = out/'protocol.json'
    records = out/'records'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['fingerprint'] != fingerprint:
            raise ValueError('Existing results have a different protocol; choose a new PHASEKV_OUT')
    elif records.exists() or (out/'summary.json').exists():
        raise ValueError('Existing result directory has no protocol')
    quick.write_json(manifest,dict(fingerprint=fingerprint,protocol=protocol))
    records.mkdir(exist_ok=True)
    cached = {}
    by_id = {s['id']: s for s in samples}
    for path in sorted(records.glob('*.json')):
        row = json.loads(path.read_text(encoding='utf8'))
        key = row['id'], row['ratio_pct']
        if key in cached or key[0] not in by_id:
            raise ValueError(f'Unexpected saved record {path}')
        validate_row(row,by_id[key[0]],fingerprint)
        cached[key] = brief(row)
    total = len(samples)*len(ORDER)
    archived = reference['records']

    def save(status, error=None):
        results = list(cached.values())
        summary = dict(status=status,fingerprint=fingerprint,protocol=protocol,
                       archived_reference_records=len(archived),planned_new_records=total,
                       completed_new_records=len(results),by_task=analyze(archived+results,samples),
                       records=results)
        if error is not None:
            summary['error'] = error
        quick.write_json(out/'summary.json',summary)

    for layer in model.model.layers:
        layer.self_attn.method = 'phasekv_triton'
    save('running')
    try:
        for pct in (ORDER[:2] if coarse else ORDER):
            settings = dict(options,prefill_keep_ratio=pct/100)
            for s in samples:
                key = s['id'], pct
                if key in cached:
                    continue
                row = generate_one(model,settings,s,None,tokenizer,f'prompt{pct}')
                row.update(ratio_pct=pct,fingerprint=fingerprint,long_repeat=long_repeat(row['output_ids']))
                validate_row(row,s,fingerprint)
                quick.write_json(records/(s['id'].replace('/','_')+f'_prompt{pct}.json'),row)
                cached[key] = brief(row)
                save('running')
                print(f'{len(cached)}/{total} {pct}% {s["id"]}: score={row["score"]:.2f} '
                      f'answered={row["answered_questions"]}/{row["total_questions"]} severe={row["severe_loop"]}',flush=True)
            curve = analyze(archived+list(cached.values()),samples)
            print(f'RATIO {pct}% ' + ' | '.join(f'{t} {curve[t]["curve"][pct]["official_score"]:.2f}'
                                                for t in TASKS),flush=True)
        status = 'complete' if len(cached)==total else 'coarse_complete'
        if status != 'complete' and not coarse:
            raise RuntimeError('Incomplete full grid')
        save(status)
    except (Exception,KeyboardInterrupt) as exc:
        save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',str(exc))
        raise
    print('FINISHED',out/'summary.json',status,flush=True)


def campaign(model, options, unused_sample, out, base_protocol, coarse):
    from transformers import AutoTokenizer
    reference = json.loads((HERE/'scan_reference.json').read_text(encoding='utf8'))
    with gzip.open(HERE/'quality_samples.json.gz','rt',encoding='utf8') as f:
        samples = json.load(f)
    options = dict(options,decode_keep_ratio=.8,optimizations=dict(options['optimizations'],gqa_reuse=True))
    if (options != reference['options'] or reference['environment'] != base_protocol['environment']
            or reference['code_sha256'] != base_protocol['code_sha256']
            or reference['grid'] != list(GRID) or reference['order'] != list(ORDER)):
        raise ValueError('Runtime, settings or planned grid differ from the archived endpoints')
    if Counter(s['task'] for s in samples) != {t:16 for t in TASKS} or len({s['id'] for s in samples}) != 48:
        raise ValueError('Expected all 48 frozen compound prompts')
    for s in samples:
        if (not s['natural_eos'] or s['max_new_tokens']!=4096 or not s['stop_ids']
                or quick.digest(json.dumps(s['input_ids']).encode()) != s['input_sha256']):
            raise ValueError(f'Invalid quality input {s["id"]}')
    path = Path(base_protocol['model'])
    for name,sha in reference['tokenizer_sha256'].items():
        if quick.digest((path/name).read_bytes()) != sha:
            raise ValueError(f'Unexpected tokenizer: {name}')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
    protocol=dict(gpu=base_protocol['gpu'],environment=base_protocol['environment'],model=str(path),
                  code_sha256=base_protocol['code_sha256'],reference_fingerprints=reference['fingerprints'],
                  same_gpu_as_endpoints=base_protocol['gpu'].split(',')[0]==reference['gpu'].split(',')[0],
                  grid=list(GRID),run_order=list(ORDER),base_options=options,overridden_option='prefill_keep_ratio',
                  selector='standard',batch=1,dtype='float16',greedy=True,natural_eos=True,max_output=4096,
                  analysis='Per task: largest adjacent score gain; >=5pp and one-sided paired sign flip, Holm .05 '
                           'over six neighbors; sustained near100 = score >= score100-3 and zero severe loops '
                           'at this and all higher tested levels. Short-period repeat: >=128 tokens, period <=16.',
                  package_sha256=quick.digest((HERE/'SHA256SUMS').read_bytes()))
    if not protocol['same_gpu_as_endpoints']:
        print('GPU UUID differs from archived endpoints; recorded in protocol.',flush=True)
    run_scan(model,options,tokenizer,samples,reference,out,protocol,coarse)


if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--coarse',action='store_true',help='Run only 80 and 90 percent; resume without this flag for the full grid.')
    args=parser.parse_args()
    args.rounds=1
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        quick.benchmark(args,args.out,expected_gpu='NVIDIA GeForce RTX 4090',
                        run_campaign=lambda model,options,sample,out,rounds,protocol:
                            campaign(model,options,sample,out,protocol,args.coarse))
