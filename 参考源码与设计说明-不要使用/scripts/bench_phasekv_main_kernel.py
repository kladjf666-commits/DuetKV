"""Compare K layouts or GQA reuse on local Qwen2/Llama weights."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import phasekv_optimization_support as support
from scripts.bench_phasekv_optimizations import assert_storage_equal, cache_factory, inputs_for, prepared_calls


def kernel_cases(configs, layouts, decode_keep_ratio=1.0):
    rows = []
    cold_length = round(support.TARGET['lazy_g'] * decode_keep_ratio)
    for heads, kvheads in layouts:
        for prompt, cold in ((1024, 0), (8192, 1), (8192, 32), (1024, 128)):
            inputs = inputs_for(prompt, cold, 136, heads, kvheads, 17, 'cuda', cold_length=cold_length)
            caches = [cache_factory(inputs, cfg, decode_keep_ratio=decode_keep_ratio) for cfg in configs]
            assert_storage_equal(*caches)
            work = [prepared_calls(inputs[5], c.layers[0], False)[0] for c in caches]
            old, new = [w['decode']() for w in work]
            torch.testing.assert_close(old, new, atol=.002, rtol=.01)
            graphs = []
            for w in work:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(64):
                        w['attention']()
                graphs.append(graph)
            measurements = [[], []]
            for i in range(8):
                for arm in (i % 2, 1 - i % 2):
                    graph = graphs[arm]
                    graph.replay()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    measurements[arm].append(start.elapsed_time(end) * 1000 / 64)
            row = dict(heads=heads, kv_heads=kvheads, input_tokens=prompt, cold_segments=cold,
                       hot_tokens=136, tokens_per_cold_segment=cold_length,
                       old_us=measurements[0], new_us=measurements[1],
                       max_abs_error=(old.float() - new.float()).abs().max().item())
            rows.append(row)
            print(f'kernel Q{heads}/KV{kvheads} {prompt=} {cold=}: '
                  f'{statistics.median(measurements[0]):.2f} -> {statistics.median(measurements[1]):.2f} us', flush=True)
    return rows


def model_case(path, configs, prompt, steps, repeats, decode_keep_ratio=1.0, llama_layers=None):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    family = config.model_type
    checkpoint_layers = config.num_hidden_layers
    if llama_layers is not None:
        if family != 'llama' or not 1 <= llama_layers <= checkpoint_layers:
            raise ValueError('--llama-layers requires a Llama checkpoint with at least that many layers')
        config.num_hidden_layers = llama_layers
    dtype = torch.bfloat16 if family == 'qwen2' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(path, config=config, torch_dtype=dtype,
        attn_implementation='eager', local_files_only=True).eval().cuda()
    if model.config.model_type == 'qwen2':
        from models.Qwen2.qwen2_attention import install_qwen2_phasekv_attention as install
        from models.Qwen2.qwen2 import build_phasekv_cache
    elif model.config.model_type == 'llama':
        from models.Llama.llama_attention import install_llama_phasekv_attention as install
        from models.packed_runtime import build_phasekv_cache
    else:
        raise ValueError('Expected Qwen2 or Llama')
    install(model)
    text = 'A blue train left the station. A red train arrived three minutes later. '
    context = tokenizer.encode(text, add_special_tokens=False)
    tail = tokenizer.encode('What color was the first train? Explain your answer.', add_special_tokens=False)
    tokens = (context * (prompt // len(context) + 1))[:prompt-len(tail)] + tail
    ids = torch.tensor([tokens], device='cuda')
    positions = torch.arange(prompt, device='cuda')
    dense = DynamicCache()
    # Long inputs only need attention scores for the final 32 query positions.
    tail_start = prompt - 32 if prompt > 2048 else 0
    for start in range(0, tail_start, 512):
        stop = min(start + 512, tail_start)
        pos = positions[start:stop]
        model.model(input_ids=ids[:, start:stop], past_key_values=dense, use_cache=True,
                    output_attentions=False, position_ids=pos[None], cache_position=pos)
    pos = positions[tail_start:]
    prefill = model.model(input_ids=ids[:, tail_start:], past_key_values=dense, use_cache=True,
                          output_attentions=True, position_ids=pos[None], cache_position=pos)
    first = model.lm_head(prefill.last_hidden_state[:, -1:]).argmax(-1)
    positions = torch.arange(prompt, prompt + steps, device='cuda')
    results = [[], []]
    pack_times = [[], []]
    packing = dict(support.TARGET, decode_keep_ratio=decode_keep_ratio)
    cold_length = max(1, round(packing['lazy_g'] * decode_keep_ratio))
    expected_segments = max(0, (steps - packing['recent_window']) // packing['lazy_g'])
    reference_ids = None
    max_difference = 0.
    for rep in range(repeats + 1):
        for arm in (rep % 2, 1 - rep % 2):
            torch.cuda.synchronize()
            pack_start = time.perf_counter()
            cache = build_phasekv_cache(prefill, model, prompt, optimizations=configs[arm],
                                        prefill_keep_ratio=.7, sink_tokens=0, obs_window=32, **packing)
            token = first
            sequence = []
            last_logits = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            if rep:
                pack_times[arm].append(1000 * (start - pack_start))
            for i in range(steps):
                pos = positions[i:i+1]
                hidden = model.model(input_ids=token, past_key_values=cache, use_cache=True,
                                     position_ids=pos[None], cache_position=pos).last_hidden_state
                cache.finish_decode_step()
                last_logits = model.lm_head(hidden)
                token = last_logits.argmax(-1)
                sequence.append(token)
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - start) / steps
            for layer in cache.layers:
                if len(layer.cold) != expected_segments or any(s.length != cold_length for s in layer.cold):
                    raise RuntimeError('Measured cold cache does not match the requested configuration')
            decoded = torch.cat(sequence, dim=1).tolist()[0]
            if not torch.isfinite(last_logits).all():
                raise RuntimeError('Non-finite logits')
            if reference_ids is None:
                reference_ids, reference_logits = decoded, last_logits.clone()
            if decoded != reference_ids:
                raise RuntimeError('Greedy output changed at decode step ' +
                                   str(next(i + 1 for i, (a,b) in enumerate(zip(decoded, reference_ids)) if a != b)))
            max_difference = max(max_difference, (last_logits.float()-reference_logits.float()).abs().max().item())
            if rep:
                results[arm].append(ms)
            print(f'model round {rep}, arm {arm}: {ms:.2f} ms/token', flush=True)
    return dict(model_type=model.config.model_type, model_path=str(path.resolve()), input_tokens=prompt, decode_steps=steps,
                loaded_layers=len(model.model.layers), checkpoint_layers=checkpoint_layers,
                generated_ids_sha256=hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest(),
                final_logit_max_abs_difference=max_difference, final_cold_segments=len(cache.layers[0].cold),
                old_ms=results[0], new_ms=results[1], old_prompt_pack_ms=pack_times[0], new_prompt_pack_ms=pack_times[1],
                dtype=str(dtype), packed_attention_dtype='float16')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prompt', type=int, nargs='+', default=[1024, 8192])
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--compare', choices=('k-layout', 'gqa-reuse'), default='k-layout')
    parser.add_argument('--decode-keep-ratio', type=float, choices=(.8, 1.0), default=1.0)
    parser.add_argument('--llama-layers', type=int,
                        help='Time only the first N Llama layers plus LM head on a small GPU.')
    args = parser.parse_args()
    if min(args.prompt) < 64 or args.steps < 145 or args.repeats < 2:
        parser.error('Require prompt >= 64, steps >= 145 (cold transition), repeats >= 2')
    support.bootstrap()
    from PhaseKV.optimizations import stage_configs
    base = stage_configs()['p5'].to_dict()
    channel = dict(base, channel_contiguous_k=True)
    configs = [channel, dict(channel, gqa_reuse=True)] if args.compare == 'gqa-reuse' else [base, channel]
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, configs=configs,
                  comparison=args.compare, decode_keep_ratio=args.decode_keep_ratio,
                  kernels=kernel_cases(configs, [(12, 2), (32, 8)], args.decode_keep_ratio))
    support.write_json(args.out, report)
    report['models'] = []
    for prompt in args.prompt:
        report['models'].append(model_case(args.model, configs, prompt, args.steps, args.repeats,
                                           args.decode_keep_ratio, args.llama_layers))
        support.write_json(args.out, report)


if __name__ == '__main__':
    main()
