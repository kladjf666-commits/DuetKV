"""Rerun corrected SCOPE on exactly the prior 160 LongBench inputs.

Budget256 is the replacement run; budget32 is a separately labelled sensitivity
run. Natural EOS and original task generation limits are preserved. Old report
publication is a separate step after all records are checked.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from eval import metrics, runtime
from scripts import run_longbench as lb
from scripts.compare_longbench_kv import digest, forbid_fake_quantization, generate, save_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, default=ROOT / 'results/qwen2_1p5b_longbench_real_kv_20261023')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    args = p.parse_args()
    old = json.loads(args.previous.with_suffix('.json').read_text())
    old_records = [json.loads(l) for l in args.previous.with_suffix('.predictions.jsonl').read_text().splitlines()]
    refs = {r['sample_id']: r for r in old_records if r['method'] == 'full'}
    meta = old['meta']
    tasks, count = meta['tasks'], meta['n_samples_per_task']
    data_dir = Path(meta['persistent_dataset_dir'])
    hashes = {t: hashlib.sha256((data_dir / 'data' / f'{t}.jsonl').read_bytes()).hexdigest() for t in tasks}
    if hashes != meta['data_hashes']:
        raise ValueError('Dataset differs from the original run')
    outputs = {b: args.out_dir / f'qwen2_1p5b_scope_b{b}_20261024' for b in (256, 32)}
    if any(path.with_suffix('.predictions.jsonl').exists() for path in outputs.values()):
        raise FileExistsError('Rerun outputs already exist; do not overwrite measured records')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA access required')
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.manual_seed(meta['seed'])
    torch.set_num_threads(4)
    forbid_fake_quantization()
    model, tokenizer = runtime.load_model_and_tokenizer(meta['model'], dtype='bfloat16', attn_implementation='eager')
    lb.LONGBENCH_DIR = str(data_dir)
    samples = []
    for task in tasks:
        data = lb.load_task(task, count)
        if len(data) != count:
            raise ValueError('Insufficient samples')
        for i, sample in enumerate(data):
            sid = f'{task}:{i}'
            ids = lb.build_input_ids(tokenizer, task, sample, meta['max_length'])
            if digest(ids.tolist()) != refs[sid]['input_hash'] or sample['answers'] != refs[sid]['answers']:
                raise ValueError(f'Input/answer mismatch: {sid}')
            samples.append((sid, task, ids, sample['answers'], lb.DATASET2MAXLEN[task]))
    source_paths = ['SCOPE/compressor.py', 'eval/runtime.py', 'eval/compression_trace.py',
                    'scripts/rerun_scope_longbench.py']
    source_hashes = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in source_paths}
    dims = lb._kv_dims(model.config)
    for budget, prefix in outputs.items():
        spec = {'method': 'scope', 'recent_window': 16, 'max_cache_tokens': budget}
        protocol = {k: meta[k] for k in ('model', 'tasks', 'n_samples_per_task', 'max_length', 'dtype',
                                        'attention', 'greedy', 'seed', 'storage_protocol')}
        protocol.update(methods=[spec], data_hashes=hashes,
                        source_hashes=source_hashes, scope_implementation='scope_slide_current_query_v2',
                        trace_protocol='kv_transitions_v1', stop_on_eos=True,
                        generation_limits={t: lb.DATASET2MAXLEN[t] for t in tasks})
        run_meta = {**protocol, 'protocol_hash': digest(protocol), 'run_date': '2026-10-24',
                    'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                    'transformers': __import__('transformers').__version__,
                    'comparison_full_source': str(args.previous.with_suffix('.predictions.jsonl')),
                    'comparison_full_protocol_hash': meta['protocol_hash'],
                    'identical_inputs_and_answers_verified': True,
                    'input_hashes': {sid: refs[sid]['input_hash'] for sid, *_ in samples}}
        records = []
        gc.collect()
        torch.cuda.empty_cache()
        generate(model, tokenizer, samples[0][2], spec, 4)
        with prefix.with_suffix('.predictions.jsonl').open('x', buffering=1) as f:
            for index, (sid, task, ids, answers, max_new) in enumerate(samples):
                torch.cuda.synchronize()
                start = time.perf_counter()
                gen = generate(model, tokenizer, ids, spec, max_new)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - start) * 1000
                written = gen['prompt_len'] + len(gen['generated_ids']) - 1
                full_bytes = metrics.kv_cache_dtype_bytes(*dims, written)
                physical = gen['full_cache_bytes']
                trace = gen['compression_trace']
                assert trace['final_state']['physical_bytes'] == physical
                assert trace['summary']['decode']['checks'] == len(gen['generated_ids']) - 1
                r = {'method': 'scope', 'protocol_hash': run_meta['protocol_hash'],
                     'run_date': '2026-10-24', 'scope_decode_budget': budget,
                     'sample_id': sid, 'task': task, 'input_hash': refs[sid]['input_hash'],
                     'answers': answers, 'prediction': gen['text'].strip(), 'generated_ids': gen['generated_ids'],
                     'prompt_tokens': gen['prompt_len'], 'written_tokens': written,
                     'qa_f1': lb.best_qa_f1(gen['text'].strip(), answers),
                     'physical_kv_bytes': physical, 'logical_kv_bytes': gen['final_cache_bytes'],
                     'full_reference_bytes': full_bytes,
                     'physical_kv_saving_pct': 100 * (1-physical/full_bytes),
                     'logical_kv_saving_pct': 100 * (1-gen['final_cache_bytes']/full_bytes),
                     'peak_gpu_bytes': gen['peak_gpu_bytes'], 'wall_ms': wall,
                     'compression_trace': trace}
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                records.append(r)
                if (index+1) % 32 == 0:
                    print(f'SCOPE budget={budget}: {index+1}/{len(samples)}', flush=True)
                    save_report(prefix, run_meta, list(refs.values()) + records, [spec], tasks, False)
        save_report(prefix, run_meta, list(refs.values()) + records, [spec], tasks, True)
        triggered = sum(r['compression_trace']['status'] == 'triggered' for r in records)
        print(f'FINISHED budget={budget}: triggered={triggered}/{len(records)}', flush=True)


if __name__ == '__main__':
    main()
