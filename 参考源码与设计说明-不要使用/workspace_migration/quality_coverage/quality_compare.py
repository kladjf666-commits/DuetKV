"""Same-model paired free generation on 48 frozen compound prompts."""
import argparse
import collections
import gzip
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'base'))
import quick

ARMS = ('original70','coverage70')


def summarize(rows, candidate='coverage70'):
    arms = ('original70', candidate)
    delta_name = 'coverage_minus_original' if candidate=='coverage70' else 'prompt100_minus_original70'
    groups = collections.defaultdict(dict)
    for row in rows:
        groups[row['id']][row['arm']] = row
    paired = [group for group in groups.values() if set(group)==set(arms)]
    summary = {}
    for task in ('all','gsm8k','csqa','mmlu'):
        pairs = [g for g in paired if task=='all' or g['original70']['task']==task]
        if not pairs:
            continue
        stats = {}
        for arm in arms:
            records = [g[arm] for g in pairs]
            stats[arm] = dict(
                official_score=statistics.mean(r['score'] for r in records),
                indexed_accuracy_pct=statistics.mean(r['indexed_accuracy_diagnostic'] for r in records),
                answered=sum(r['answered_questions'] for r in records),
                total_questions=sum(r['total_questions'] for r in records),
                missing=sum(r['unanswered_questions'] for r in records),
                severe_loops=sum(r['severe_loop'] for r in records),
                frequent_headers=sum(r['max_same_header']>=100 for r in records))
            if all('decode_s' in r for r in records):
                steps = sum(r['decode_steps'] for r in records)
                seconds = sum(r['decode_s'] for r in records)
                stats[arm].update(decode_steps=steps, decode_s=seconds,
                                  decode_tokens_per_s=steps / seconds if seconds else None)
        fields = ('official_score','indexed_accuracy_pct','answered','missing','severe_loops')
        summary[task] = dict(complete_pairs=len(pairs),arms=stats,
            recovered_loops=[g['original70']['id'] for g in pairs
                             if g['original70']['severe_loop'] and not g[candidate]['severe_loop']],
            new_loops=[g[candidate]['id'] for g in pairs
                       if g[candidate]['severe_loop'] and not g['original70']['severe_loop']])
        summary[task][delta_name] = {k:stats[candidate][k]-stats['original70'][k] for k in fields}
    return summary


def generate_standard(model, options, sample):
    import ablation_runtime as rt
    from ablation_generate import generate
    import PhaseKV.prompt_tier as prompt
    from PhaseKV.phasekv_cache import PhaseKVCache

    length = len(sample['input_ids'])
    layers = len(model.model.layers)
    heads = model.config.num_key_value_heads
    ratio = options['prefill_keep_ratio']
    budget = max(1, round(ratio * length))
    if not 0 < ratio <= 1 or (options['prompt_bits'],options['cold_bits'],options['decode_keep_ratio'])!=(4,4,.8):
        raise ValueError('Expected 4-bit prompt/cold KV and 80% cold retention')
    standard, trace = prompt.select_prompt_indices, rt.cache_trace
    audited_layers = []

    def select(attentions, **settings):
        indices = standard(attentions, **settings)
        if settings['prompt_len']!=length or settings['prefill_keep_ratio']!=ratio or len(indices)!=layers:
            raise RuntimeError('Unexpected standard prompt selection')
        for keep in indices:
            if (keep.shape != (1,heads,budget) or
                    not ((keep[...,0] >= 0).all() & (keep[...,-1] < length).all()
                         & (keep[...,1:] > keep[...,:-1]).all()).item()):
                raise RuntimeError('Wrong budget, invalid or duplicated prompt positions')
        audited_layers.append(len(indices))
        return indices

    def packed_trace(cache, method):
        if method!='phasekv_triton' or type(cache) is not PhaseKVCache or len(cache.layers)!=layers:
            raise RuntimeError('Unexpected packed cache')
        for layer in cache.layers:
            if layer.prompt.length!=budget or (layer.prompt.key.bits,layer.prompt.value.bits)!=(4,4):
                raise RuntimeError('Unexpected length or precision in packed prompt')
            if any((seg.length,seg.key.bits,seg.value.bits)!=(13,4,4) for seg in layer.cold):
                raise RuntimeError('Unexpected generated cold block')
        return trace(cache, method)

    with patch.object(prompt,'select_prompt_indices',select), patch.object(rt,'cache_trace',packed_trace):
        row = generate(model,'phasekv_triton',options,sample)
    if audited_layers!=[layers]:
        raise RuntimeError('Expected one complete prompt audit')
    audit = dict(verified_layer_heads=layers*heads, all_original_positions=budget==length, prompt_bits=4)
    return row, layers, audit


def generate_coverage(model, options, sample, units, arm):
    from ablation_generate import generate
    import PhaseKV.prompt_tier as prompt
    from coverage_policy import CoveragePlan

    length = len(sample['input_ids'])
    plan = CoveragePlan(units,length,next(model.parameters()).device)
    standard = prompt.select_prompt_indices
    def select(attentions, **settings):
        if settings['prompt_len']!=length or settings['prefill_keep_ratio']!=.7 or settings['sink_tokens']!=0:
            raise ValueError('Selector received unexpected budget or sample')
        if (settings['obs_window'],settings['pool_kernel'],settings['pooling'])!=(32,7,'maxpool'):
            raise ValueError('Unexpected attention voting configuration')
        original = standard(attentions,**settings)
        if not len(original)==len(attentions)==len(model.model.layers):
            raise RuntimeError('Missing prompt selection layer')
        return [plan.apply(keep,scores,repair=arm=='coverage70') for keep,scores in zip(original,attentions)]
    with patch.object(prompt,'select_prompt_indices',select):
        generated = generate(model,'phasekv_triton',options,sample)
    if len(plan.layers)!=len(model.model.layers) or any(x['heads']!=model.config.num_key_value_heads for x in plan.layers):
        raise RuntimeError('Selection audit did not cover all layers and heads')
    trace = generated['prefill_trace']
    if trace['prompt_tokens']!=round(.7*length):
        raise RuntimeError('Packed cache has a different prompt budget')
    return generated, len(plan.layers), plan.totals()


def generate_one(model, options, sample, units, tokenizer, arm):
    from benchmark_metrics import score
    if arm.startswith('prompt'):
        pct = round(100 * options['prefill_keep_ratio'])
        if arm != f'prompt{pct}' or pct/100 != options['prefill_keep_ratio']:
            raise ValueError('Arm label does not match its prompt retention ratio')
        generated, layers, audit = generate_standard(model,options,sample)
    else:
        generated, layers, audit = generate_coverage(model,options,sample,units,arm)
    output = generated['output_ids']
    prediction = tokenizer.decode(output,skip_special_tokens=True)
    metrics = score(sample,prediction)
    header_counts = collections.Counter(re.findall(r'\bAnswer_(\d+):',prediction))
    max_header = max(header_counts.values(),default=0)
    if not generated['stopped_on_eos'] and not generated['hit_output_limit']:
        raise RuntimeError('Generation ended without EOS or the limit')
    return dict(id=sample['id'], task=sample['task'], arm=arm,input_sha256=sample['input_sha256'],
                prompt_tokens=len(sample['input_ids']),retained_per_head=generated['prefill_trace']['prompt_tokens'],
                layer_count=layers,
                output_tokens=len(output),output_sha256=quick.digest(json.dumps(output).encode()),
                output_ids=output,prediction=prediction,final_trace=generated['final_trace'],
                ttft_s=generated['ttft_s'],decode_steps=len(output)-1,decode_s=generated['decode_s'],
                decode_tokens_per_s=generated['decode_tokens_per_s'],decode_chunks=generated['decode_chunks'],
                peak_allocated_bytes=generated['peak_allocated_bytes'],final_kv_bytes=generated['final_storage']['physical_bytes'],
                stopped_on_eos=generated['stopped_on_eos'],hit_output_limit=generated['hit_output_limit'],
                selection=audit,max_same_header=max_header,
                severe_loop=max_header>=100 and metrics['answered_questions']==0 and generated['hit_output_limit'],
                **metrics)


def run_suite(model, options, samples, tokenizer, out, protocol, *, reference=None):
    arms = ('prompt100',) if reference is not None else ARMS
    candidate = arms[-1]
    by_id = {s['id']:s for s in samples}
    old = [] if reference is None else reference['records']
    if reference is not None:
        if len(old)!=len(samples) or {r['id'] for r in old}!=set(by_id):
            raise ValueError('Historical original70 baseline is incomplete')
        for r in old:
            if (r['arm']!='original70' or r['input_sha256']!=by_id[r['id']]['input_sha256']
                    or r['task']!=by_id[r['id']]['task'] or r['total_questions']!=len(by_id[r['id']]['answers'])):
                raise ValueError('Historical baseline differs from this prompt or task')
    for layer in model.model.layers:
        layer.self_attn.method='phasekv_triton'
    fingerprint = quick.digest(json.dumps(protocol,sort_keys=True).encode())
    manifest = out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['fingerprint']!=fingerprint:
        raise RuntimeError('Existing records have a different protocol; set PHASEKV_OUT to a fresh folder')
    if not manifest.exists() and (out/'records').exists():
        raise RuntimeError('Record directory has no protocol manifest')
    quick.write_json(manifest,dict(fingerprint=fingerprint,protocol=protocol))
    records = out/'records'
    records.mkdir(exist_ok=True)
    rows = []
    cache = {}
    for path in sorted(records.glob('*.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        key=(row['id'],row['arm'])
        if (row['fingerprint']!=fingerprint or key in cache or row['arm'] not in arms
                or row['id'] not in by_id or row['input_sha256']!=by_id[row['id']]['input_sha256']
                or quick.digest(json.dumps(row['output_ids']).encode())!=row['output_sha256']):
            raise RuntimeError(f'Invalid saved record {path}')
        cache[key]=row
    def brief(row):
        return {k:v for k,v in row.items() if k not in ('output_ids','prediction','decode_chunks')}
    rows.extend(brief(row) for row in cache.values())
    def save(status):
        quick.write_json(out/'summary.json',dict(status=status, protocol=protocol,fingerprint=fingerprint,
                     planned_prompts=len(samples),generated_arms=arms,completed_records=len(rows),
                     archived_reference_records=len(old),paired=summarize(old+rows,candidate),records=rows))
    save('running')
    try:
        for i,sample in enumerate(samples):
            units = None
            if reference is None:
                from coverage_policy import question_units
                units = question_units(tokenizer.backend_tokenizer,sample['input_ids'],sample['task'])
                stems = [u['question'] for u in units if u['kind']=='stem']
                start=sample['first_answer_number']
                if stems!=list(range(start,start+len(sample['answers']))):
                    raise RuntimeError(f'Question segmentation failed for {sample["id"]}')
            for arm in arms if i%2==0 else reversed(arms):
                if (sample['id'],arm) in cache:
                    continue
                row=generate_one(model,options,sample,units,tokenizer,arm)
                row['fingerprint']=fingerprint
                path=records/(sample['id'].replace('/','_')+'_'+arm+'.json')
                quick.write_json(path,row)
                rows.append(brief(row))
                save('running')
                print(f'{len(rows)}/{len(arms)*len(samples)} {sample["id"]} {arm}: '
                      f'score={row["score"]:.2f} answered={row["answered_questions"]}/{row["total_questions"]} '
                      f'severe_loop={row["severe_loop"]}',flush=True)
        if len(rows)!=len(arms)*len(samples) or summarize(old+rows,candidate)['all']['complete_pairs']!=len(samples):
            raise RuntimeError('Incomplete paired generation')
        save('complete')
    except (Exception,KeyboardInterrupt) as exc:
        save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed')
        raise
    print('PAIRED',json.dumps(summarize(old+rows,candidate)['all']),flush=True)
    print('FINISHED',out/'summary.json',flush=True)


def campaign(model, options, unused_speed_sample, out, unused_rounds, base_protocol):
    from transformers import AutoTokenizer
    with gzip.open(HERE/'quality_samples.json.gz','rt',encoding='utf-8') as f:
        samples=json.load(f)
    if collections.Counter(s['task'] for s in samples)!={'csqa':16,'gsm8k':16,'mmlu':16} or len({s['id'] for s in samples})!=48:
        raise ValueError('Expected all 48 frozen LongGenBench samples')
    for s in samples:
        if (not s['natural_eos'] or s['max_new_tokens']!=4096 or not s['stop_ids']
                or quick.digest(json.dumps(s['input_ids']).encode())!=s['input_sha256']):
            raise ValueError(f'Invalid frozen generation protocol: {s["id"]}')
    options=dict(options,decode_keep_ratio=.8,optimizations=dict(options['optimizations'],gqa_reuse=True))
    model_path=Path(base_protocol['model'])
    data=json.loads((HERE/'quality_manifest.json').read_text(encoding='utf-8'))
    reference = None
    if data.get('generation_arm')=='prompt100':
        reference=json.loads((HERE/'original70_reference.json').read_text(encoding='utf-8'))
        if (options!=reference['options'] or base_protocol['code_sha256']!=reference['runtime_sha256']
                or base_protocol['environment']!=reference['environment']
                or [s['id'] for s in samples]!=data['ordered_ids']):
            raise RuntimeError('Runtime, options or frozen prompts differ from the completed baseline')
        options=dict(options,prefill_keep_ratio=1.)
        print('Single arm: prompt100, 48 generations; original70 read from completed records.',flush=True)
    for name,sha in data['tokenizer_sha256'].items():
        if hashlib.sha256((model_path/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Tokenizer mismatch: {name}')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    protocol=dict(gpu=base_protocol['gpu'],environment=base_protocol['environment'],model=str(model_path),
                  code_sha256=base_protocol['code_sha256'],options=options,batch=1,dtype='float16',
                  greedy=True,natural_eos=True,max_output=4096,manifest=data,
                  package_sha256=hashlib.sha256((HERE/'SHA256SUMS').read_bytes()).hexdigest())
    if reference is not None:
        protocol['baseline_fingerprint']=reference['fingerprint']
        protocol['baseline_gpu']=reference['gpu']
        protocol['same_gpu_as_reference']=reference['gpu'].split(',')[0]==base_protocol['gpu'].split(',')[0]
    run_suite(model,options,samples,tokenizer,out,protocol,reference=reference)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.rounds=1
    args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        quick.benchmark(args,args.out,expected_gpu='NVIDIA GeForce RTX 4090',run_campaign=campaign)
