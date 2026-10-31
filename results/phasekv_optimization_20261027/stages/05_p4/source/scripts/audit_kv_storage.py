"""Audit resident KV storage with synthetic CPU tensors, not model quality scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers.cache_utils import DynamicCache

import baselines  # noqa: F401
import src.phasekv  # noqa: F401
from eval.compressors import build_compressor
from src.phasekv.phasekv_cache import PhaseKVCache
from src.phasekv.prompt_tier import select_prompt_indices


def dense_cache(length=1536):
    generator = torch.Generator().manual_seed(1234)
    cache = DynamicCache()
    for layer in range(2):
        cache.update(*(torch.randn(1, 2, length, 128, generator=generator,
                                   dtype=torch.float16) for _ in range(2)), layer)
    return cache


def snapshot(cache, compressor):
    tensors = cache.key_cache + cache.value_cache
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in tensors}
    return {
        "dtypes": sorted({str(t.dtype) for t in tensors}),
        "layer_tokens": [t.shape[-2] for t in cache.key_cache],
        "kv_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "kv_storage_bytes": sum(storages.values()),
        "reported_logical_bytes": compressor.cache_memory_bytes(cache),
    }


def audit(reference):
    torch.set_num_threads(2)
    config = SimpleNamespace(num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, hidden_size=512, head_dim=128)
    generator = torch.Generator().manual_seed(2345)
    attentions = tuple(torch.rand(1, 4, 32, 1536, generator=generator).softmax(-1)
                       for _ in range(2))
    rows = []
    for spec in reference["rows"]:
        compressor = build_compressor(spec["method"], **spec["params"])
        compressor.reset(config)
        cache = dense_cache()
        before = snapshot(cache, compressor)
        cache = compressor.on_prefill(cache, attentions=attentions,
                                      prompt_len=1536, model_config=config)
        prefill = snapshot(cache, compressor)
        # 128 output tokens require at most 127 cache appends after prefill.
        for step in range(127):
            decode_attentions = []
            for layer in range(2):
                token = torch.full((1, 2, 1, 128), (step + 1) / 128,
                                   dtype=torch.float16)
                cache.update(token, -token, layer)
                length = cache.key_cache[layer].shape[-2]
                decode_attentions.append(torch.full((1, 4, 1, length), 1 / length))
            cache = compressor.on_decode_step(cache, attentions=tuple(decode_attentions),
                                               step=step, model_config=config)
        rows.append({"method": spec["method"], "params": spec["params"],
                     "before": before, "prefill": prefill,
                     "after_127_appends": snapshot(cache, compressor)})

    dense = dense_cache()
    indices = select_prompt_indices(attentions, prompt_len=1536, model_config=config,
                                    prefill_keep_ratio=0.3, sink_tokens=0,
                                    obs_window=32, pool_kernel=7, pooling="maxpool")
    packed = PhaseKVCache.from_dense_prefill(dense, prompt_len=1536,
                                             keep_indices=indices)
    packed_info = {
        "implementation": "PhaseKVCache (separate from registered phasekv compressor)",
        "code_dtype": str(packed.layers[0].prompt.key.packed.dtype),
        "scale_dtype": str(packed.layers[0].prompt.key.scale.dtype),
        "physical_bytes_including_hot_capacity": packed.physical_nbytes(),
        "prompt_tokens": packed.get_seq_length(),
        "hot_capacity": packed.recent_window + packed.lazy_g,
    }
    by_name = {row["method"]: row for row in rows}
    assert by_name["kivi"]["prefill"]["kv_storage_bytes"] == by_name["full"]["prefill"]["kv_storage_bytes"]
    assert all(row["prefill"]["dtypes"] == ["torch.float16"] for row in rows)
    assert by_name["scope"]["after_127_appends"]["layer_tokens"] == [1663, 1663]
    assert packed_info["code_dtype"] == "torch.uint8"
    return {"kind": "synthetic CPU storage audit; no model inference or QA scores",
            "geometry": {"layers": 2, "kv_heads": 2, "head_dim": 128,
                         "prompt_tokens": 1536, "decode_appends": 127},
            "rows": rows, "separate_packed_phasekv": packed_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/m1_longbench_7b.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/kv_storage_audit_20261023.json")
    args = parser.parse_args()
    result = audit(json.loads(args.reference.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in result["rows"]:
        p = row["prefill"]
        print(row["method"], p["dtypes"], "resident", p["kv_storage_bytes"],
              "logical", p["reported_logical_bytes"])
    print(json.dumps(result["separate_packed_phasekv"], ensure_ascii=False))


if __name__ == "__main__":
    main()
