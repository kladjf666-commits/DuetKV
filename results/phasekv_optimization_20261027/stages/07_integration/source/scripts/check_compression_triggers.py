"""Functional Qwen2 trigger checks, NOT a quality or speed benchmark.

Runs one shared LongBench prompt with fixed output length (ignores EOS) so cache
maintenance crosses its thresholds. Original LongBench prediction files are not
modified. The extra SCOPE run crosses its unchanged 256-token decode budget.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from scripts.compare_longbench_kv import forbid_fake_quantization
from scripts import run_longbench as lb
from eval import runtime
from eval.compressors import build_compressor
from eval.packed_runtime import generate_packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/compression_triggers_qwen2_20261024.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU access required')
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    forbid_fake_quantization()
    reference = json.loads((ROOT / 'results/m1_longbench_7b.json').read_text())
    specs = [{'method': r['method'], **r['params']} for r in reference['rows']]
    model, tokenizer = runtime.load_model_and_tokenizer(args.model, dtype='bfloat16', attn_implementation='eager')
    sample = json.loads((ROOT / '.cache/longbench/data/narrativeqa.jsonl').read_text().splitlines()[0])
    ids = lb.build_input_ids(tokenizer, 'narrativeqa', sample, 1536)
    runs = [(s, 80) for s in specs]
    runs += [(next(s for s in specs if s['method'] == 'scope'), 258)]
    payload = {'purpose': 'functional_trigger_validation_not_quality_or_speed',
               'gpu': torch.cuda.get_device_name(), 'model': args.model,
               'dtype': 'bfloat16', 'stop_on_eos': False, 'sample_id': 'narrativeqa:0',
               'prompt_tokens': ids.shape[1], 'input_hash': hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),
               'runs': [], 'complete': False}
    for spec, output_length in runs:
        method = spec['method']
        params = {k: v for k, v in spec.items() if k != 'method'}
        gc.collect()
        torch.cuda.empty_cache()
        if method in ('kivi', 'phasekv'):
            gen = generate_packed(model, tokenizer, ids, method=method, params=params,
                                  max_new_tokens=output_length, stop_on_eos=False, trace_compression=True)
        else:
            comp = build_compressor(method, **params)
            gen = runtime.generate_with_compression(model, tokenizer, ids, comp,
                    max_new_tokens=output_length, stop_on_eos=False, trace_compression=True)
        trace = gen['compression_trace']
        assert len(gen['generated_ids']) == output_length
        assert trace['summary']['decode']['checks'] == output_length - 1
        assert trace['final_state']['physical_bytes'] == gen['full_cache_bytes']
        if method == 'full':
            assert trace['status'] == 'baseline_no_compression'
        elif method == 'scope':
            dec = trace['summary']['decode']
            if output_length <= 257:
                assert trace['status'] == 'not_triggered'
            else:
                assert dec['first_trigger_step'] == 257 and dec['triggered_steps'] == 1
                assert dec['layer_token_totals']['evicted_tokens'] == model.config.num_hidden_layers
        else:
            assert trace['summary']['prefill']['triggered_steps'] == 1
        if method in ('snapkv', 'pyramidkv'):
            assert trace['summary']['decode']['triggered_steps'] == 0
        if method == 'phasekv':
            assert trace['summary']['decode']['first_trigger_step'] == 48
        if method == 'kivi':
            assert trace['summary']['decode']['events_by_action']['key_quantization'] >= 1
            assert trace['summary']['decode']['events_by_action']['value_quantization'] == output_length - 1
        payload['runs'].append({'method': method, 'params': params,
                                'generated_tokens': output_length, 'generated_ids': gen['generated_ids'],
                                'compression_trace': trace,
                                'storage_evidence': gen.get('storage_evidence')})
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        print(f"{method} output={output_length} status={trace['status']} "
              f"prefill={trace['summary']['prefill']['triggered_steps']} "
              f"decode={trace['summary']['decode']['triggered_steps']}", flush=True)
    payload['complete'] = True
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
