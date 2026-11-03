"""Compare GQA off/on at fixed historical prefixes using one GPU-resident layer.

This local numerical replay uses all 32 Llama-3.1-8B layers, in FP16. Weight
transfers are outside the scope of the test. Use the resident benchmark to time
decoding. Run with the repository's CUDA Python environment.
"""
import gzip
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts import phasekv_optimization_support as support

support.bootstrap()
from models.Llama.llama_attention import PhaseKVLlamaAttention
from PhaseKV.optimizations import stage_configs
from PhaseKV.phasekv_cache import PhaseKVCache
from PhaseKV.prompt_tier import select_prompt_indices

MODEL = PROJECT / "weights/Llama-3.1-8B-Instruct/snapshots/master"
FROZEN = PROJECT / "official_llama/bench_assets/frozen/samples.jsonl.gz"
HISTORY = PROJECT / "不同算法的不同数据集测试/native_llama_20261029/records/phasekv"
STEPS = {"performance/8192+256/0": 255, "longgenbench/csqa/23": 272, "performance/1024+1024/0": 384}
BASE = replace(stage_configs()["p5"], channel_contiguous_k=True)
PACKING = dict(support.TARGET, decode_keep_ratio=0.8)


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def load_cases(embedding, rotary):
    cases = []
    with gzip.open(FROZEN, "rt", encoding="utf-8") as samples:
        for line in samples:
            sample = json.loads(line)
            sample_id = sample["id"]
            if sample_id not in STEPS:
                continue
            steps = STEPS[sample_id]
            record_name = hashlib.sha256(sample_id.encode()).hexdigest()[:24] + "_r0.json"
            record = json.loads((HISTORY / record_name).read_text(encoding="utf-8"))
            assert record["id"] == sample_id and record["input_sha256"] == sample["input_sha256"]
            assert record["status"] == "ok" and len(record["output_ids"]) > steps
            teacher_ids = record["output_ids"][:steps]
            prompt = embedding[torch.tensor([sample["input_ids"]], device="cuda")]
            decode = embedding[torch.tensor([teacher_ids], device="cuda")]
            length = prompt.shape[1]
            positions = torch.arange(length + steps, device="cuda")
            future = positions[:length][None] > positions[length - 32:length, None]
            mask = torch.zeros((1, 1, 32, length), device="cuda", dtype=torch.float16)
            mask.masked_fill_(future, torch.finfo(torch.float16).min)
            positions_by_phase = (positions[:length - 32], positions[length - 32:length], positions[length:])
            cases.append(dict(
                id=sample_id, input_sha256=sample["input_sha256"], teacher_sha256=ids_digest(teacher_ids),
                length=length, steps=steps, prefix=prompt[:, :-32].contiguous(), tail=prompt[:, -32:].contiguous(),
                decode=[decode, decode], mask=mask, rope=[rotary(prompt, pos[None]) for pos in positions_by_phase],
                layer_errors=[],
            ))
    assert len(cases) == len(STEPS)
    return cases


def replay_layer(layer, case, config):
    dense = DynamicCache()
    prefix_rope, tail_rope, decode_rope = case["rope"]
    case["prefix"] = layer(case["prefix"], past_key_value=dense, use_cache=True, position_embeddings=prefix_rope)[0]
    case["tail"], scores, _ = layer(
        case["tail"], past_key_value=dense, use_cache=True, output_attentions=True,
        attention_mask=case["mask"], position_embeddings=tail_rope,
    )
    keep = select_prompt_indices(
        [scores], prompt_len=case["length"], model_config=config, prefill_keep_ratio=0.7,
        obs_window=32, sink_tokens=0, pool_kernel=7, pooling="maxpool",
    )
    del scores
    original = layer.self_attn
    layer.self_attn = PhaseKVLlamaAttention(original)
    for arm, reuse in enumerate((False, True)):
        cache = PhaseKVCache.from_dense_prefill(
            dense, prompt_len=case["length"], keep_indices=keep, optimizations=replace(BASE, gqa_reuse=reuse), **PACKING,
        )
        states = []
        for step in range(case["steps"]):
            state = layer(
                case["decode"][arm][:, step:step + 1], past_key_value=cache, use_cache=True,
                position_embeddings=tuple(part[:, step:step + 1] for part in decode_rope),
            )[0]
            cache.finish_decode_step()
            states.append(state)
        tiers = cache.layers[0]
        assert tiers.optimizations.gqa_reuse == reuse and tiers._decode_workspace.compatible_4bit
        assert len(tiers.cold) == (case["steps"] - 128) // 16 and all(seg.length == 13 for seg in tiers.cold)
        case["decode"][arm] = torch.cat(states, dim=1)
    layer.self_attn = original
    before, after = case["decode"]
    case["layer_errors"].append((before.float() - after.float()).abs().max().item())


def compare_logits(case, norm, lm_head):
    token_ids = [[], []]
    maximum = 0.0
    for start in range(0, case["steps"], 8):
        logits = [torch.nn.functional.linear(norm(hidden[:, start:start + 8]), lm_head).float()
                  for hidden in case["decode"]]
        assert all(torch.isfinite(scores).all() for scores in logits)
        maximum = max(maximum, (logits[0] - logits[1]).abs().max().item())
        for ids, scores in zip(token_ids, logits):
            ids.extend(scores.argmax(-1)[0].tolist())
    return dict(
        id=case["id"], input_sha256=case["input_sha256"], teacher_ids_sha256=case["teacher_sha256"],
        input_tokens=case["length"], compared_positions=case["steps"], cold_segments=(case["steps"] - 128) // 16,
        cold_segment_length=13, top1_disagreements=[i for i, (a, b) in enumerate(zip(*token_ids), 1) if a != b],
        max_abs_vocab_logits=maximum, layer_hidden_max_abs=case["layer_errors"], top1_sha256=list(map(ids_digest, token_ids)),
    )


@torch.inference_mode()
def main():
    weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        with safe_open(str(MODEL / weight_map[name]), framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).to(device="cuda", dtype=torch.float16)

    config = LlamaConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    cases = load_cases(weight("model.embed_tokens.weight"), LlamaRotaryEmbedding(config=config, device="cuda"))
    for index in range(config.num_hidden_layers):
        # Each real layer gets a separate one-layer cache with index zero.
        with torch.device("meta"):
            layer = LlamaDecoderLayer(config, layer_idx=0)
        layer.load_state_dict({key: weight(f"model.layers.{index}.{key}") for key in layer.state_dict()}, assign=True)
        layer.eval()
        for case in cases:
            replay_layer(layer, case, config)
        del layer
        if (index + 1) % 8 == 0:
            print(f"Replayed {index + 1} / {config.num_hidden_layers} layers", flush=True)

    norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device="cuda", dtype=torch.float16)
    norm.weight.copy_(weight("model.norm.weight"))
    lm_head = weight("lm_head.weight")
    rows = [compare_logits(case, norm, lm_head) for case in cases]
    report = dict(
        gpu=torch.cuda.get_device_name(), model="Llama-3.1-8B-Instruct", model_path=str(MODEL),
        torch=torch.__version__, layers=config.num_hidden_layers,
        inputs="Frozen prompt and historical native PhaseKV output IDs, forced one token at a time.",
        prefill="Common SDPA prefix and eager final 32 query rows",
        mode="FP16 resident one layer at a time; numerical comparison only",
        frozen_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(), options=PACKING, prompt_keep_ratio=0.7,
        before=BASE.to_dict(), after=replace(BASE, gqa_reuse=True).to_dict(), samples=rows,
    )
    support.write_json(Path(__file__).with_name("llama_fixed_prefix.json"), report)
    for row in rows:
        print(row["id"], "max logit difference", row["max_abs_vocab_logits"], flush=True)
    if any(row["top1_disagreements"] or row["max_abs_vocab_logits"] for row in rows):
        raise RuntimeError("The two kernel configurations produced different logits")


if __name__ == "__main__":
    main()
