#!/usr/bin/env python3
"""Resumable, single-GPU real-KV benchmark. See BENCHMARK_4090D.md.

Only standard-library imports in the controller. CUDA lives in subprocesses.
This runner deliberately does not depend on the missing eval/ or baselines/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import string
import subprocess
import sys
import time
import traceback
import types
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full", "kivi", "snapkv", "scope", "pyramidkv", "phasekv")
TASK_LIMITS = {"narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
               "hotpotqa": 32, "2wikimqa": 32,
               "gov_report": 512, "qmsum": 512, "multi_news": 512}
SUMMARY_TASKS = {"gov_report", "qmsum", "multi_news"}
PROTOCOL = "phasekv_4090d_v1_chunked_sdpa_tail_attention"
VALIDATION_PROTOCOL = "v4_variable_window_retention_layerwise_oracle"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_records(path, *, repair=False):
    """Only a torn final append may be discarded; corruption in the middle is fatal."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, valid_bytes = [], 0
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt record in {path}, line {index + 1}")
            if repair:
                with path.open("r+b") as handle:
                    handle.truncate(valid_bytes)
            break
        valid_bytes += len(line)
    if repair and records and raw and not raw.endswith(b"\n") and valid_bytes == len(raw):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            print(f"Download: {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "PhaseKV-benchmark/1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def prepare_data(folder, tasks):
    folder = Path(folder)
    missing = [t for t in tasks if not (folder / "data" / f"{t}.jsonl").is_file()]
    if missing:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        archive = folder / "data.zip"
        download(f"{endpoint}/datasets/THUDM/LongBench/resolve/main/data.zip", archive)
        # Extract only exact expected task members, never arbitrary archive paths.
        with zipfile.ZipFile(archive) as source:
            for task in missing:
                name = f"data/{task}.jsonl"
                if name not in source.namelist():
                    raise ValueError(f"Official archive lacks {name}")
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(source.read(name))
                temporary.replace(target)
    prompts_path = folder / "dataset2prompt.json"
    download("https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json",
             prompts_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    result = {"prompts_sha256": file_hash(prompts_path), "tasks": {}}
    for task in tasks:
        rows = load_rows(folder / "data" / f"{task}.jsonl")
        if not rows or task not in prompts:
            raise ValueError(f"Missing data or prompt for {task}")
        for row in rows:
            if not isinstance(row.get("answers"), list) or not row["answers"] or not row.get("context"):
                raise ValueError(f"Invalid LongBench row: {task}")
        result["tasks"][task] = {"count": len(rows), "sha256": file_hash(folder / "data" / f"{task}.jsonl")}
    return result


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_indices(count, requested, seed, task, offset=0):
    indices = list(range(count))
    random.Random(f"{seed}:{task}").shuffle(indices)
    if offset < 0 or offset >= count:
        raise ValueError("Sample offset outside dataset")
    return sorted(indices[offset:] if requested == 0 else indices[offset:offset + requested])


def chunk_ranges(length, chunk_size, observation):
    """The final chunk must contain the complete selection observation window."""
    if min(length, chunk_size, observation) < 1 or chunk_size < observation:
        raise ValueError("Require positive sizes and chunk_size >= observation")
    end = max(0, length - min(length, observation))
    ranges = [(start, min(start + chunk_size, end)) for start in range(0, end, chunk_size)]
    # Reserve exactly W final queries, limiting eager-score workspace to W x N.
    ranges.append((end, length))
    return ranges


def normalize_answer(value):
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def qa_f1(prediction, answer):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(answer).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    return 2 * overlap / (len(pred) + len(gold)) if overlap else 0.0


def score_prediction(task, prediction, answers):
    if task not in SUMMARY_TASKS:
        return 100 * max(qa_f1(prediction, answer) for answer in answers)
    # Same rouge package / ROUGE-L F1 convention as LongBench v1 metrics.py.
    from rouge import Rouge
    if not prediction.strip():
        return 0.0
    scores = []
    for answer in answers:
        try:
            scores.append(Rouge().get_scores([prediction], [answer], avg=True)["rouge-l"]["f"])
        except ValueError:
            scores.append(0.0)
    return 100 * max(scores)


@dataclass
class CompressionConfig:
    name: str = "base"
    prefill_keep_ratio: float | None = None
    decode_keep_ratio: float | None = None
    sink_tokens: int = 0
    recent_window: int | None = None
    bits: int | None = None
    max_cache_tokens: int | None = None
    seed: int = 1234
    extra: dict = field(default_factory=dict)

    def validate(self):
        for name in ("prefill_keep_ratio", "decode_keep_ratio"):
            value = getattr(self, name)
            if value is not None and not 0 < value <= 1:
                raise ValueError(name)
        for name in ("recent_window", "max_cache_tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(name)
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens")


class KVCompressor:
    """Small import bridge for existing algorithms, private to this process."""
    def __init__(self, config):
        config.validate()
        self.config, self.name = config, config.name

    def reset(self, model_config):
        pass

    def describe(self):
        return {"name": self.name, "config": vars(self.config)}

    def cache_memory_bytes(self, cache):
        return sum(t.numel() * t.element_size() for t in cache.key_cache + cache.value_cache)


def bootstrap_algorithms():
    """No disk restoration, legacy fake quantization, or dependence on deleted eval files."""
    sys.path.insert(0, str(ROOT))
    if "eval" in sys.modules:
        raise RuntimeError("Run this benchmark in a fresh Python process")
    package = types.ModuleType("eval")
    package.__path__ = []
    interface = types.ModuleType("eval.interface")
    interface.CompressionConfig, interface.KVCompressor = CompressionConfig, KVCompressor
    registry = types.ModuleType("eval.compressors")
    registry.__path__ = []
    registered = {}
    registry.register = lambda name, cls: registered.__setitem__(name, cls)
    sys.modules.update({"eval": package, "eval.interface": interface, "eval.compressors": registry})
    package.interface, package.compressors = interface, registry
    from PhaseKV import hetero_cache
    ops = types.ModuleType("eval.compressors.head_ops")
    for name in ("cache_num_layers", "cache_seq_len", "num_kv_heads_of", "aggregate_to_kv_heads",
                 "observation_votes", "pool_scores", "evict_per_head"):
        setattr(ops, name, getattr(hetero_cache, name))

    def topk_keep_per_head(scores, select_k, prefix_len, recent_lo, recent_hi):
        import torch
        top = hetero_cache.topk_per_head(scores, min(select_k, prefix_len))
        tail = torch.arange(recent_lo, recent_hi, device=scores.device).view(1, 1, -1)
        tail = tail.expand(scores.shape[0], scores.shape[1], -1)
        return torch.cat((top, tail), dim=-1)

    ops.topk_keep_per_head = topk_keep_per_head
    registry.head_ops = ops
    sys.modules[ops.__name__] = ops


def make_compressor(spec):
    if spec["method"] in ("full", "kivi", "phasekv"):
        return None
    paths = {"snapkv": ("SnapKV.compressor", "SnapKVCompressor"),
             "scope": ("SCOPE.compressor", "SCOPECompressor"),
             "pyramidkv": ("PyramidKV.compressor", "PyramidKVCompressor")}
    module, name = paths[spec["method"]]
    fields = CompressionConfig.__dataclass_fields__
    values = {k: v for k, v in spec.items() if k != "method" and k in fields}
    extra = {k: v for k, v in spec.items() if k != "method" and k not in fields}
    return getattr(importlib.import_module(module), name)(CompressionConfig(name=spec["method"], extra=extra, **values))


def install_adapters(model):
    """Explicit SDPA-capable composition; originals handle all dense forwards.

    The older Llama installer rejects SDPA. Here we compose the same wrappers
    directly and require the GPU equivalence checks below before benchmarking.
    """
    family = model.config.model_type
    if family == "qwen2":
        from models.Qwen2.qwen2_attention import PhaseKVQwen2Attention as PhaseAttention
        from models.Qwen2.kivi_attention import KIVIQwen2Attention as KiviAttention
    elif family == "llama":
        from models.Llama.llama_attention import PhaseKVLlamaAttention as PhaseAttention
        from models.Llama.kivi_attention import KIVILlamaAttention as KiviAttention
    else:
        raise ValueError(f"Unsupported family: {family}")
    originals = []
    for layer in model.model.layers:
        original = layer.self_attn
        originals.append(original)
        layer.self_attn = KiviAttention(PhaseAttention(original))
    return originals


def prefill_model(model, ids, chunk_size, need_scores, observation=32):
    import torch
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    length = ids.shape[1]
    output = None
    for start, end in chunk_ranges(length, chunk_size, observation):
        if output is not None:
            del output
        positions = torch.arange(start, end, device=ids.device)
        output = model(input_ids=ids[:, start:end], past_key_values=cache,
                       position_ids=positions.unsqueeze(0), cache_position=positions,
                       use_cache=True, output_attentions=need_scores and end == length,
                       num_logits_to_keep=1)
        cache = output.past_key_values
    return output


def build_cache(model, output, length, spec):
    from KIVI.packed_cache import PackedKIVICache
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.prompt_tier import select_prompt_indices
    method = spec["method"]
    if method == "kivi":
        return PackedKIVICache.from_dense_prefill(output.past_key_values,
            bits=spec.get("bits", 2), group_size=spec.get("group_size", 32),
            residual_length=spec.get("recent_window", 64)), None
    if method == "phasekv":
        ratio = spec.get("prefill_keep_ratio", .3)
        keep = None if ratio == 1 else select_prompt_indices(output.attentions,
            prompt_len=length, model_config=model.config, prefill_keep_ratio=ratio,
            sink_tokens=spec.get("sink_tokens", 0), obs_window=spec.get("obs_window", 32),
            pool_kernel=7, pooling="maxpool")
        keys = ("prompt_bits", "cold_bits", "group_size", "recent_window", "lazy_g", "decode_keep_ratio")
        options = {k: spec[k] for k in keys if k in spec}
        return PhaseKVCache.from_dense_prefill(output.past_key_values, prompt_len=length,
                                               keep_indices=keep, **options), None
    compressor = make_compressor(spec)
    cache = output.past_key_values
    if compressor:
        compressor.reset(model.config)
        cache = compressor.on_prefill(cache, attentions=output.attentions,
                                     prompt_len=length, model_config=model.config)
    return cache, compressor


def physical_bytes(cache):
    if hasattr(cache, "physical_nbytes"):
        return cache.physical_nbytes()
    storages = {}
    for tensor in cache.key_cache + cache.value_cache:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def cache_signature(cache, method):
    if method == "phasekv":
        return sum(len(layer.cold) for layer in cache.layers)
    if method == "kivi":
        return sum((layer.key.orig_shape[2] if layer.key else 0) +
                   (layer.value.orig_shape[2] if layer.value else 0) for layer in cache.layers)
    return sum(t.shape[2] for t in cache.key_cache)


def stop_ids(model, tokenizer):
    for source in (model.generation_config, model.config, tokenizer):
        value = getattr(source, "eos_token_id", None)
        if value is not None:
            return set(value if isinstance(value, (list, tuple)) else [value])
    return set()


def generate(model, tokenizer, ids, spec, new_tokens, chunk_size, fixed=False):
    import torch
    from models.packed_runtime import storage_evidence
    method = spec["method"]
    length = ids.shape[1]
    if length + new_tokens > model.config.max_position_embeddings:
        raise ValueError("Input + output exceeds checkpoint context; no implicit RoPE extension")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()
    started = time.perf_counter()
    with torch.inference_mode():
        output = prefill_model(model, ids, chunk_size,
            method in ("snapkv", "pyramidkv", "phasekv"), spec.get("obs_window", 32))
        if not bool(torch.isfinite(output.logits).all()):
            raise RuntimeError("Non-finite prefill logits")
        token = output.logits[:, -1].argmax(-1)
        generated = [int(token.item())]
        cache, compressor = build_cache(model, output, length, spec)
        del output
        prefill_bytes = physical_bytes(cache)
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started
        decode_started = time.perf_counter()
        trigger_steps, first_trigger = 0, None
        eos = stop_ids(model, tokenizer)
        for step in range(1, new_tokens):
            if not fixed and generated[-1] in eos:
                break
            before = cache_signature(cache, method)
            output = model(input_ids=token.reshape(1, 1), past_key_values=cache,
                use_cache=True, output_attentions=method == "scope",
                position_ids=torch.tensor([[length + step - 1]], device=ids.device),
                cache_position=torch.tensor([cache.get_seq_length()], device=ids.device),
                num_logits_to_keep=1)
            cache = output.past_key_values
            if method in ("phasekv", "kivi"):
                cache.finish_decode_step()
            elif compressor:
                cache = compressor.on_decode_step(cache, attentions=output.attentions,
                                                 step=step - 1, model_config=model.config)
            after = cache_signature(cache, method)
            changed = (after > before if method in ("phasekv", "kivi")
                       else after < before + model.config.num_hidden_layers)
            if changed:
                trigger_steps += 1
                first_trigger = step if first_trigger is None else first_trigger
            if not bool(torch.isfinite(output.logits).all()):
                raise RuntimeError("Non-finite decode logits")
            token = output.logits[:, -1].argmax(-1)
            generated.append(int(token.item()))
            del output
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        peak = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        final_bytes = physical_bytes(cache)
        evidence = storage_evidence(cache) if method in ("phasekv", "kivi") else {"cache_class": type(cache).__name__}
        cfg = model.config
        full_bytes = 4 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * (length + len(generated) - 1)
        result = {"input_tokens": length, "output_tokens": len(generated), "generated_ids": generated,
            "prediction": tokenizer.decode(generated, skip_special_tokens=True),
            "stopped_on_eos": not fixed and generated[-1] in eos, "fixed_output": fixed,
            "prefill_and_compress_s": prefill_seconds, "decode_s": decode_seconds,
            "decode_tokens_per_s": (len(generated) - 1) / decode_seconds if len(generated) > 1 else None,
            "total_s": prefill_seconds + decode_seconds,
            "peak_allocated_bytes": peak, "peak_reserved_bytes": peak_reserved,
            "initial_allocated_bytes": initial_allocated,
            "prefill_kv_bytes": prefill_bytes, "final_kv_bytes": final_bytes,
            "equivalent_full_kv_bytes": full_bytes, "kv_saving_pct": 100 * (1 - final_bytes / full_bytes),
            "decode_compression_steps": trigger_steps, "first_decode_compression_step": first_trigger,
            "storage_evidence": evidence}
        del cache
    return result


def quality_input(tokenizer, prompt, sample, budget):
    import torch
    raw = prompt.format(**sample)
    body = tokenizer.encode(raw, add_special_tokens=False)
    original = len(body)
    # Include chat-template tokens in the actual budget and avoid duplicate BOS.
    allowance = budget
    for _ in range(8):
        if len(body) > allowance:
            left = allowance // 2
            selected = body[:left] + body[-(allowance - left):]
            text = tokenizer.decode(selected, skip_special_tokens=True)
        else:
            text = raw
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True)
        if len(ids) <= budget:
            return torch.tensor([ids], dtype=torch.long), {"original_body_tokens": original,
                "truncated": original > allowance, "input_hash": digest(ids)}
        allowance -= len(ids) - budget + 8
        if allowance < 16:
            break
    raise ValueError("Could not fit prompt and chat template into token budget")


def synthetic_input(tokenizer, length, seed, depth=None):
    import torch
    code = str(random.Random(seed).randint(10000000, 99999999))
    marker = "[[PHASEKV_INSERT_CONTEXT_HERE]]"
    if depth is None:
        query = "Write a detailed, well-structured analysis of this archive. Discuss each topic in depth."
    else:
        query = "What is the secret access code in the archive? Answer with only the eight digits."
    rendered = tokenizer.apply_chat_template([{"role": "user", "content":
                f"Read the following archive carefully.\n{marker}\n\n{query}"}],
                tokenize=False, add_generation_prompt=True)
    left, right = rendered.split(marker)
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    fact = tokenizer.encode(f"\nThe secret access code is {code}. Remember this exact code.\n", add_special_tokens=False) if depth is not None else []
    filler = tokenizer.encode("The archive describes a town library, its books, seasonal opening hours, and maintenance of the reading rooms.\n", add_special_tokens=False)
    size = length - len(prefix) - len(suffix) - len(fact)
    if size < 1:
        raise ValueError("Synthetic prompt budget is too small")
    content = (filler * ((size + len(filler) - 1) // len(filler)))[:size]
    location = int(size * depth) if depth is not None else 0
    ids = prefix + content[:location] + fact + content[location:] + suffix
    assert len(ids) == length
    return torch.tensor([ids], dtype=torch.long), code


def materialized_layer(layer):
    """Reconstruct the exact FP16 values consumed by a packed attention kernel."""
    import torch
    from PhaseKV.kernels.packing import unpack_dequantize
    if hasattr(layer, "key_residual"):
        def joined(packed, residual):
            return torch.cat([unpack_dequantize(packed), residual], dim=2) if packed else residual
        return joined(layer.key, layer.key_residual), joined(layer.value, layer.value_residual)
    segments = [layer.prompt, *layer.prompt_extensions, *layer.cold]
    key = torch.cat([unpack_dequantize(s.key) for s in segments] + [layer.hot_views()[0]], dim=2)
    value = torch.cat([unpack_dequantize(s.value) for s in segments] + [layer.hot_views()[1]], dim=2)
    return key, value


def materialized_cache(cache):
    """Independent oracle cache; preserve storage precision, including FP16."""
    from transformers.cache_utils import DynamicCache
    dense = DynamicCache()
    if hasattr(cache, "key_cache"):
        pairs = zip(cache.key_cache, cache.value_cache)
    else:
        pairs = map(materialized_layer, cache.layers)
    for index, (key, value) in enumerate(pairs):
        dense.update(key.clone(), value.clone(), index)
    return dense


def validated_packed_forward(model, cache, forward_kwargs, compare):
    """Validate each real kernel/adapter on identical inputs, without BF16 logit amplification.

    Only the tiny validation model enters this function. Kernel outputs are returned
    unchanged; temporary hooks and interception are removed even when a check fails.
    """
    import torch
    from contextlib import ExitStack
    from unittest.mock import patch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rope
    import KIVI.kernels as kivi_kernels
    import PhaseKV.kernels.attention_triton as phase_kernels

    rope = qwen_rope if model.config.model_type == "qwen2" else llama_rope
    state = {}
    layer_indices = {id(layer): index for index, layer in enumerate(cache.layers)}

    def prepare(index, base, args, kwargs):
        if index in state:
            raise AssertionError(f"Packed layer {index} executed twice in one decode step")
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        q = base.q_proj(hidden).view(1, 1, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden).view(1, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden).view(1, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        positions = kwargs.get("position_embeddings")
        if positions is None:
            positions = base.rotary_emb(v, kwargs["position_ids"])
        q, k = rope(q, k, *positions)
        old_k, old_v = materialized_layer(cache.layers[index])
        state[index] = {"query": q[0, :, 0].to(torch.float16),
                        "key": torch.cat((old_k, k.to(torch.float16)), dim=2),
                        "value": torch.cat((old_v, v.to(torch.float16)), dim=2),
                        "dtype": hidden.dtype, "base": base, "calls": 0}

    def intercept(original):
        def checked(query, layer, *args, **kwargs):
            index = layer_indices[id(layer)]
            entry = state[index]
            entry["calls"] += 1
            if entry["calls"] != 1:
                raise AssertionError(f"Packed kernel called twice for layer {index}")
            compare(query, entry["query"], "packed_query_identity", layer=index, atol=0, rtol=0)
            key, value = materialized_layer(layer)
            for kind, tensor in (("key", key), ("value", value)):
                compare(tensor, entry[kind], "packed_append_identity", layer=index,
                        cache=kind, atol=0, rtol=0)
            # FP32 oracle on the SAME dequantized FP16 values and query. No BF16 cast.
            heads = query.shape[0] // key.shape[1]
            keys = key[0].float().repeat_interleave(heads, dim=0)
            values = value[0].float().repeat_interleave(heads, dim=0)
            scores = (keys * entry["query"].float()[:, None, :]).sum(-1) * (query.shape[-1] ** -.5)
            expected = (scores.softmax(-1)[..., None] * values).sum(1).to(torch.float16)
            actual = original(query, layer, *args, **kwargs)
            # Same tolerances as tests/test_kivi_packed.py's packed-kernel oracle.
            compare(actual, expected, "packed_kernel_vs_fp32_oracle", layer=index, atol=.002, rtol=.01)
            context = actual.reshape(1, 1, -1).to(entry["dtype"])
            entry["projected"] = entry["base"].o_proj(context)
            return actual
        return checked

    def verify_output(index, output):
        entry = state[index]
        if entry["calls"] != 1 or output[1] is not None or output[2] is not cache:
            raise AssertionError(f"Packed adapter contract failed for layer {index}")
        compare(output[0], entry["projected"], "packed_output_projection_identity",
                layer=index, atol=0, rtol=0)

    with ExitStack() as stack:
        stack.enter_context(patch.object(kivi_kernels, "kivi_decode", intercept(kivi_kernels.kivi_decode)))
        stack.enter_context(patch.object(phase_kernels, "fused_phasekv_decode", intercept(phase_kernels.fused_phasekv_decode)))
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            base = attention.original.original
            before = attention.register_forward_pre_hook(
                lambda module, args, kwargs, i=index, b=base: prepare(i, b, args, kwargs), with_kwargs=True)
            stack.callback(before.remove)
            after = attention.register_forward_hook(
                lambda module, args, output, i=index: verify_output(i, output))
            stack.callback(after.remove)
        output = model(past_key_values=cache, **forward_kwargs)
        if len(state) != len(cache.layers) or not bool(torch.isfinite(output.logits).all()):
            raise AssertionError("Incomplete packed layer coverage or non-finite logits")
    return output


def validation_step(checks, context, operation, report_path=None):
    """Keep the first failure and completed checks, including its precise stage."""
    try:
        details = operation() or {}
    except Exception as error:
        failure = {**context, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
        if report_path:
            atomic_json(report_path, {"status": "failed", "validation_protocol": VALIDATION_PROTOCOL,
                                      "checks": checks, "failure": failure})
        label = json.dumps(context, ensure_ascii=False)
        raise RuntimeError(f"GPU validation failed at {label}: {error}") from error
    checks.append({**context, "status": "passed", **details})


def phasekv_validation_steps(spec):
    """Exercise at least two cooling events, including retained/dropped tokens."""
    return max(80, spec.get("recent_window", 32) + 2 * spec.get("lazy_g", 16))


def gpu_validation(profiles, report_path=None):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
    from transformers.cache_utils import DynamicCache
    bootstrap_algorithms()
    # FP32 semantic comparisons must not silently use TF32 products.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks = []

    def compare(actual, expected, profile, stage, *, atol, rtol, **context):
        def operation():
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
            return {"atol": atol, "rtol": rtol, "dtype": str(actual.dtype),
                    "max_abs_error": float((actual.float() - expected.float()).abs().max())}
        validation_step(checks, {"profile": profile, "check": stage, **context}, operation, report_path)

    with torch.inference_mode():
        for profile in profiles:
            cfg = profile["architecture"]
            family = cfg["model_type"]
            config_class, model_class = ((Qwen2Config, Qwen2ForCausalLM) if family == "qwen2"
                                        else (LlamaConfig, LlamaForCausalLM))
            torch.manual_seed(1234)
            options = dict(vocab_size=128, hidden_size=cfg["hidden_size"], intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=cfg["num_attention_heads"],
                num_key_value_heads=cfg["num_key_value_heads"], max_position_embeddings=1024,
                attention_dropout=0., use_cache=True, tie_word_embeddings=False)
            sdpa_config = config_class(**options)
            sdpa_config._attn_implementation = "sdpa"
            model = model_class(sdpa_config).to(device="cuda", dtype=torch.float32).eval()
            eager_config = config_class(**options)
            eager_config._attn_implementation = "eager"
            reference = model_class(eager_config).to(device="cuda", dtype=torch.float32).eval()
            reference.load_state_dict(model.state_dict())
            prompt_length = 160  # .3 retention now exercises historical top-k, not only the recent window.
            ids = (torch.arange(prompt_length, device="cuda") % 128).unsqueeze(0)
            print(f"[validate] {profile['name']}: FP32 chunked prefill versus full eager", flush=True)
            expected = reference(input_ids=ids, past_key_values=DynamicCache(), use_cache=True,
                                 output_attentions=True, num_logits_to_keep=1)
            actual = prefill_model(model, ids, 64, True, 32)
            compare(actual.logits, expected.logits, profile['name'], "fp32_chunked_logits", atol=1e-4, rtol=1e-4)
            for index in range(2):
                compare(actual.attentions[index], expected.attentions[index][:, :, -32:], profile['name'],
                        "fp32_observation_scores", layer=index, atol=1e-5, rtol=1e-4)
                for kind in ("key_cache", "value_cache"):
                    compare(getattr(actual.past_key_values, kind)[index], getattr(expected.past_key_values, kind)[index],
                            profile['name'], "fp32_chunked_cache", layer=index, cache=kind, atol=1e-4, rtol=1e-4)
            del actual, expected
            # Different BF16 GEMM shapes/backend choices are not an exact-equivalence oracle.
            # Instead, require exact transparency of our adapters under the SAME BF16 computation.
            model.to(dtype=torch.bfloat16)
            reference.to(dtype=torch.bfloat16)
            native = prefill_model(model, ids, 64, True, 32)
            install_adapters(model)
            actual = prefill_model(model, ids, 64, True, 32)
            print(f"[validate] {profile['name']}: BF16 native versus wrapped (identical chunks)", flush=True)
            compare(actual.logits, native.logits, profile['name'], "bf16_adapter_logits_identity", atol=0, rtol=0)
            for index in range(2):
                compare(actual.attentions[index], native.attentions[index], profile['name'],
                        "bf16_adapter_attention_identity", layer=index, atol=0, rtol=0)
                for kind in ("key_cache", "value_cache"):
                    compare(getattr(actual.past_key_values, kind)[index], getattr(native.past_key_values, kind)[index],
                            profile['name'], "bf16_adapter_cache_identity", layer=index, cache=kind, atol=0, rtol=0)
            eager_bf16 = reference(input_ids=ids, past_key_values=DynamicCache(), use_cache=True,
                                   output_attentions=True, num_logits_to_keep=1)
            def record_precision_difference():
                if not bool(torch.isfinite(actual.logits).all() & torch.isfinite(eager_bf16.logits).all()):
                    raise AssertionError("Non-finite BF16 prefill logits")
                delta = actual.logits.float() - eager_bf16.logits.float()
                return {"max_abs_error": float(delta.abs().max()),
                        "rms_error": float(delta.square().mean().sqrt()),
                        "argmax_equal": bool((actual.logits.argmax(-1) == eager_bf16.logits.argmax(-1)).all())}
            validation_step(checks, {"profile": profile['name'], "check": "bf16_cross_backend_finiteness"},
                            record_precision_difference, report_path)
            checks[-1]["comparison_role"] = "Finiteness gate; error/argmax are diagnostics, not equivalence assertions"
            print(f"[validate] BF16 chunk/full diagnostic: max_abs={checks[-1]['max_abs_error']:.6g}, "
                  f"rms={checks[-1]['rms_error']:.6g}", flush=True)
            del actual, native, eager_bf16
            # Dense decode must use the same backend and score request as production.
            # The eager model above is only the FP32 prefill semantic oracle.
            weights = reference.state_dict()
            del reference
            reference = model_class(sdpa_config).to(device="cuda", dtype=torch.bfloat16).eval()
            reference.load_state_dict(weights)
            del weights
            for spec in profile["methods"]:
                name = spec["method"]
                decode_steps = phasekv_validation_steps(spec) if name == "phasekv" else 80
                print(f"[validate] {profile['name']}: {name}, {decode_steps} decode steps", flush=True)
                output = prefill_model(model, ids, max(64, spec.get("obs_window", 32)),
                                       name in ("phasekv", "snapkv", "pyramidkv"), spec.get("obs_window", 32))
                cache, compressor = build_cache(model, output, prompt_length, spec)
                del output
                # Cross PhaseKV cooling, KIVI residual flush and SCOPE eviction boundaries.
                for step in range(decode_steps):
                    kwargs = dict(input_ids=torch.tensor([[(step + 97) % 128]], device="cuda"),
                        use_cache=True, position_ids=torch.tensor([[prompt_length + step]], device="cuda"),
                        cache_position=torch.tensor([cache.get_seq_length()], device="cuda"),
                        output_attentions=name == "scope", num_logits_to_keep=1)
                    if name in ("kivi", "phasekv"):
                        def layer_compare(actual, expected, stage, **details):
                            compare(actual, expected, profile['name'], stage, method=name, step=step + 1, **details)
                        # Capture non-comparison failures too (kernel launch, append contract, etc.).
                        holder = {}
                        def packed_step():
                            holder['output'] = validated_packed_forward(model, cache, kwargs, layer_compare)
                        validation_step(checks, {"profile": profile['name'], "check": "packed_decode_step",
                                                "method": name, "step": step + 1}, packed_step, report_path)
                        actual = holder.pop('output')
                    else:
                        dense = materialized_cache(cache)
                        expected = reference(past_key_values=dense, **kwargs)
                        actual = model(past_key_values=cache, **kwargs)
                        compare(actual.logits, expected.logits, profile['name'], "dense_decode_same_backend",
                                method=name, step=step + 1, atol=0, rtol=0)
                        for index in range(2):
                            for kind in ("key_cache", "value_cache"):
                                compare(getattr(actual.past_key_values, kind)[index], getattr(dense, kind)[index],
                                        profile['name'], "dense_decode_cache_identity", method=name,
                                        step=step + 1, layer=index, cache=kind, atol=0, rtol=0)
                            if name == "scope":
                                compare(actual.attentions[index], expected.attentions[index], profile['name'],
                                        "scope_decode_scores_identity", method=name, step=step + 1,
                                        layer=index, atol=0, rtol=0)
                        del expected, dense
                    cache = actual.past_key_values
                    if name in ("kivi", "phasekv"):
                        cache.finish_decode_step()
                    elif compressor:
                        cache = compressor.on_decode_step(cache, attentions=actual.attentions,
                                                           step=step, model_config=model.config)
                    del actual
                if name == "phasekv":
                    cooled = max(0, (decode_steps - spec.get("recent_window", 32)) // spec.get("lazy_g", 16))
                    retained = max(1, round(spec.get("decode_keep_ratio", 1.0) * spec.get("lazy_g", 16)))
                    if cooled < 2 or not all(len(layer.cold) == cooled for layer in cache.layers):
                        raise AssertionError("PhaseKV cooling boundary count mismatch")
                    if not all(segment.length == retained for layer in cache.layers for segment in layer.cold):
                        raise AssertionError("PhaseKV decode retention count mismatch")
                if name == "scope" and cache.get_seq_length() != prompt_length + spec["max_cache_tokens"]:
                    raise AssertionError("SCOPE eviction was not exercised")
                if name in ("phasekv", "kivi"):
                    from models.packed_runtime import storage_evidence
                    storage_evidence(cache)
                checks.append({"profile": profile["name"], "check": f"{name}_{decode_steps}_decode_steps_validated", "status": "passed"})
                del cache
            del model, reference
            gc.collect()
            torch.cuda.empty_cache()
    return checks


def clean_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Recorder:
    def __init__(self, path, protocol_hash, model, spec):
        self.path, self.protocol_hash, self.model, self.spec = Path(path), protocol_hash, model, spec
        self.records = read_records(path, repair=True)
        if any(r["protocol_hash"] != protocol_hash for r in self.records):
            raise ValueError("Resume protocol mismatch")
        self.done = {r["case_id"]: r for r in self.records}

    def add(self, key, record):
        record.update(case_id=key, protocol_hash=self.protocol_hash, model=self.model,
                      method=self.spec["method"], parameters=self.spec)
        append_record(self.path, record)
        self.done[key] = record
        self.records.append(record)
        print(f"[{self.model}/{self.spec['method']}] {key}: {record['status']}"
              + (f" | peak={record['peak_allocated_bytes']/2**30:.2f} GiB | {record.get('output_tokens')} output tokens"
                 if "peak_allocated_bytes" in record else ""), flush=True)
        return record


def run_case(recorder, key, model, tokenizer, ids, spec, output_limit, args, *, fixed=False,
             metadata=None, task=None, answers=None, needle=None):
    import torch
    if key in recorder.done:
        return recorder.done[key]
    clean_cuda()
    metadata = {**(metadata or {}), "requested_output_tokens": output_limit,
                "input_hash": digest(ids.tolist()), "input_tokens": ids.shape[1]}
    if ids.shape[1] + output_limit > model.config.max_position_embeddings:
        return recorder.add(key, {**metadata, "status": "context_limit", "score": None})
    # Exceptions leave generate() before empty_cache(), releasing all stack-held tensors.
    oom_message = None
    print(f"[start] {recorder.model}/{spec['method']} {key}", flush=True)
    try:
        result = generate(model, tokenizer, ids.to("cuda"), spec, output_limit, args["chunk_size"], fixed)
    except torch.cuda.OutOfMemoryError as error:
        oom_message = str(error).split("\n")[0]
    if oom_message is not None:
        clean_cuda()
        return recorder.add(key, {**metadata, "status": "oom", "error": oom_message, "score": None})
    score = None
    if task:
        score = score_prediction(task, result["prediction"], answers)
    if needle:
        score = 100.0 if re.search(rf"(?<!\d){re.escape(needle)}(?!\d)", result["prediction"]) else 0.0
    return recorder.add(key, {**metadata, **result, "status": "ok", "score": score,
                              "answers": answers if answers is not None else ([needle] if needle else None)})


def capacity_search(limit, target_bytes, attempt):
    """Grow then bisect at 256-token precision; target is a ceiling, not dummy allocation."""
    if limit < 256:
        return {"max_completed_input": 0, "target_safe_input": 0, "reason": "context_limit"}
    low, best, high, candidate = 0, 0, None, min(1024, limit)
    while True:
        record = attempt(candidate)
        success = record["status"] == "ok"
        if success:
            best = max(best, candidate)
        if success and record["peak_allocated_bytes"] <= target_bytes:
            low = candidate
            if candidate == limit:
                break
        else:
            high = candidate
        if high is None:
            candidate = min(limit, candidate * 2)
        else:
            if high - low <= 256:
                break
            candidate = ((low + high) // 2 // 256) * 256
            candidate = max(low + 256, candidate)
            if candidate >= high:
                break
    return {"max_completed_input": best, "target_safe_input": low,
            "upper_boundary_input": high,
            "reason": "checkpoint_or_requested_limit" if low == limit else "memory_target_or_oom"}


def load_model(job):
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(job["model_path"], trust_remote_code=False)
    expected = job["architecture"]
    keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "max_position_embeddings", "rope_theta", "tie_word_embeddings")
    different = {k: [expected[k], getattr(cfg, k, None)] for k in keys if expected[k] != getattr(cfg, k, None)}
    if different:
        raise ValueError(f"Checkpoint does not match {job['name']}: {different}")
    if getattr(cfg, "quantization_config", None):
        raise ValueError("Use original BF16 weights; weight quantization would change the comparison")
    if getattr(cfg, "use_sliding_window", False) or getattr(cfg, "pretraining_tp", 1) != 1:
        raise ValueError("Sliding-window or tensor-parallel checkpoints are not validated")
    tokenizer = AutoTokenizer.from_pretrained(job["model_path"], trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("The original chat/instruct tokenizer template is required")
    model = AutoModelForCausalLM.from_pretrained(job["model_path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
        trust_remote_code=False).eval()
    install_adapters(model)
    return model, tokenizer


def worker(job):
    import torch
    args, spec = job["args"], job["spec"]
    bootstrap_algorithms()
    torch.set_num_threads(args["cpu_threads"])
    torch.manual_seed(args["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer = load_model(job)
    recorder = Recorder(job["records_path"], job["protocol_hash"], job["name"], spec)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    external_bytes = max(0, total_vram - free - torch.cuda.memory_reserved())
    target = min(total_vram * args["target_vram"], total_vram - external_bytes - 512 * 2**20)
    if torch.cuda.memory_allocated() >= target:
        raise RuntimeError("Weights already exceed usable memory target. Stop other GPU processes or increase target.")
    atomic_json(Path(job["records_path"]).with_suffix(f".environment.{time.time_ns()}.json"), {
        "model_path": job["model_path"], "checkpoint_commit": getattr(model.config, "_commit_hash", None),
        "config": model.config.to_dict(), "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": digest(tokenizer.chat_template), "gpu": torch.cuda.get_device_name(),
        "total_vram_bytes": total_vram, "external_bytes_at_start": external_bytes,
        "target_allocated_bytes": target, "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.import_module("transformers").__version__,
        "triton": importlib.import_module("triton").__version__})
    # Warm all maintenance boundaries before the first workload; timing is not reported.
    warm_ids, _ = synthetic_input(tokenizer, 128, args["seed"])
    warm_tokens = phasekv_validation_steps(spec) + 1 if spec["method"] == "phasekv" else 80
    warm = generate(model, tokenizer, warm_ids.to("cuda"), spec, warm_tokens, args["chunk_size"], True)
    if spec["method"] in ("scope", "phasekv", "kivi") and warm["decode_compression_steps"] == 0:
        raise RuntimeError(f"{spec['method']} did not compress in the warmup")
    del warm
    clean_cuda()
    context = model.config.max_position_embeddings
    limit = min(args["max_context"], context - 128)
    if "capacity" in args["phases"]:
        def attempt(length):
            ids, _ = synthetic_input(tokenizer, length, args["seed"])
            return run_case(recorder, f"capacity/{length}", model, tokenizer, ids, spec, 128,
                args, fixed=True, metadata={"phase": "capacity", "length_budget": length})
        capacity = capacity_search(limit, target, attempt)
        atomic_json(Path(job["records_path"]).with_suffix(".capacity.json"), capacity)
    else:
        capacity = {"target_safe_input": limit}
    safe_length = capacity["target_safe_input"]
    if not safe_length:
        raise RuntimeError("No tested context fits the memory target")

    if "needle" in args["phases"]:
        lengths = sorted(set(args["lengths"] + [safe_length]))
        for length in lengths:
            for depth in (.1, .5, .9):
                for sample in range(args["needle_samples"]):
                    key = f"needle/{length}/{depth}/{sample}"
                    meta = {"phase": "needle", "length_budget": length, "depth": depth, "sample": sample}
                    if key in recorder.done:
                        continue
                    if length > safe_length or length + 32 > context:
                        recorder.add(key, {**meta, "status": "over_capacity_or_context", "score": None})
                        continue
                    ids, code = synthetic_input(tokenizer, length, args["seed"] + sample, depth)
                    run_case(recorder, key, model, tokenizer, ids, spec, 32, args, metadata=meta, needle=code)

    if "performance" in args["phases"]:
        # Same shapes for every method. A capacity-specific shape is separately labelled.
        shapes = {(length, 256, "context_sweep") for length in args["lengths"]}
        shapes |= {(1024, output, "output_sweep") for output in args["outputs"]}
        shapes |= {(8192, 1024, "joint_sweep"), (16384, 2048, "joint_sweep"), (24576, 4096, "joint_sweep")}
        shapes.add((safe_length, 128, "capacity_specific"))
        if args.get("performance_shapes") is not None:
            shapes = {tuple(shape) for shape in args["performance_shapes"]}
        for length, output, sweep in sorted(shapes):
            for repeat in range(args["repeats"] + 1):
                key = f"performance/{sweep}/{length}/{output}/{repeat}"
                meta = {"phase": "performance", "sweep": sweep, "length_budget": length,
                        "repeat": repeat, "is_warmup": repeat == 0}
                if key in recorder.done:
                    continue
                if length > safe_length or length + output > context:
                    recorder.add(key, {**meta, "requested_output_tokens": output,
                        "status": "over_capacity_or_context", "score": None})
                    continue
                ids, _ = synthetic_input(tokenizer, length, args["seed"])
                run_case(recorder, key, model, tokenizer, ids, spec, output, args, fixed=True, metadata=meta)

    if "quality" in args["phases"]:
        folder = Path(args["data_dir"])
        prompts = json.loads((folder / "dataset2prompt.json").read_text(encoding="utf-8"))
        for task in args["tasks"]:
            data = load_rows(folder / "data" / f"{task}.jsonl")
            indices = sample_indices(len(data), args["samples"], args["seed"], task, args.get("sample_offset", 0))
            for length in args["quality_lengths"]:
                for index in indices:
                    key = f"quality/{task}/{length}/{index}"
                    meta = {"phase": "quality", "task": task, "length_budget": length,
                            "sample_index": index, "metric": "rouge_l" if task in SUMMARY_TASKS else "qa_f1"}
                    if key in recorder.done:
                        continue
                    if length > safe_length or length + TASK_LIMITS[task] > context:
                        recorder.add(key, {**meta, "status": "over_capacity_or_context", "score": None})
                        continue
                    ids, details = quality_input(tokenizer, prompts[task], data[index], length)
                    run_case(recorder, key, model, tokenizer, ids, spec, TASK_LIMITS[task], args,
                        metadata={**meta, **details, "dataset_id": data[index].get("_id", str(index))},
                        task=task, answers=data[index]["answers"])
    del model
    clean_cuda()


def confidence_interval(values, seed=1234):
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return [estimates[24], estimates[974]]


def summarize(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for path in sorted((out / "records").glob("*.jsonl")):
        records.extend(read_records(path))
    expected_methods = set(manifest["settings"]["methods"])
    paired = defaultdict(dict)
    for record in records:
        if record.get("phase") in ("quality", "needle"):
            paired[(record["model"], record["case_id"])][record["method"]] = record
    complete = set()
    for key, methods in paired.items():
        if set(methods) == expected_methods and all(r["status"] == "ok" for r in methods.values()):
            if len({r["input_hash"] for r in methods.values()}) != 1:
                raise ValueError(f"Unfair comparison: input hash differs for {key}")
            complete.add(key)
    grouped = defaultdict(list)
    for record in records:
        if record.get("phase") in ("quality", "needle") and (record["model"], record["case_id"]) in complete:
            key = (record["model"], record["phase"], record.get("task", "custom_passkey"),
                   record["length_budget"], record["method"])
            grouped[key].append(record)
    rows = []
    for (model, phase, task, length, method), selected in sorted(grouped.items()):
        scores = [r["score"] for r in selected]
        differences = []
        for record in selected:
            baseline = paired[(model, record["case_id"])].get("full")
            if baseline:
                differences.append(record["score"] - baseline["score"])
        rows.append({"model": model, "phase": phase, "task": task, "length_budget": length,
            "method": method, "n_paired": len(scores), "score": statistics.mean(scores),
            "delta_vs_full": statistics.mean(differences) if differences else None,
            "paired_delta_95pct_bootstrap": confidence_interval(differences),
            "mean_final_kv_mib": statistics.mean(r["final_kv_bytes"] for r in selected) / 2**20,
            "decode_triggered_samples": sum(r["decode_compression_steps"] > 0 for r in selected)})
    perf_groups = defaultdict(list)
    for r in records:
        if r.get("phase") == "performance" and not r.get("is_warmup") and r["status"] == "ok":
            perf_groups[(r["model"], r["method"], r["sweep"], r["input_tokens"], r["output_tokens"])].append(r)
    performance = []
    for (model, method, sweep, inputs, outputs), selected in sorted(perf_groups.items()):
        performance.append({"model": model, "method": method, "sweep": sweep,
            "input_tokens": inputs, "output_tokens": outputs, "n": len(selected),
            "median_decode_tokens_per_s": statistics.median(r["decode_tokens_per_s"] for r in selected),
            "median_prefill_and_compress_s": statistics.median(r["prefill_and_compress_s"] for r in selected),
            "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in selected) / 2**30})
    statuses = dict(Counter(r["status"] for r in records))
    atomic_json(out / "summary.json", {"protocol": manifest["protocol"], "record_statuses": statuses,
        "paired_quality": rows, "performance": performance,
        "excluded_unpaired_cases": len(paired) - len(complete)})
    lines = ["# PhaseKV 单卡测试结果", "", f"记录状态：`{json.dumps(statuses, ensure_ascii=False)}`。",
        f"排除未完成全部方法的质量/检索案例：{len(paired)-len(complete)}。", "",
        "分数仅在所有选定方法成功且输入哈希一致的样本交集上比较；OOM 不计作零分。",
        "参数是各方法显式配置，不是等 KV 字节预算；SCOPE 为本仓库 ALLKV+Slide 版本。",
        "截断 LongBench、custom_passkey 与官方完整排行榜/RULER 有不同口径。",
        "长输出测速忽略 EOS，不参与质量评分。速度含生成循环、缓存维护和有限值检查；不是纯 kernel 速度。",
        "容量测试追踪真实 allocated 峰值；reserved 显存不代表有效 KV；不做虚假显存填充。", "",
        "| 模型 | 任务 | 输入预算 | 方法 | 配对样本 | 分数 | 相对 full | KV MiB | decode触发样本 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        delta = "—" if r["delta_vs_full"] is None else f"{r['delta_vs_full']:+.2f}"
        lines.append(f"| {r['model']} | {r['task']} | {r['length_budget']} | {r['method']} | {r['n_paired']} | {r['score']:.2f} | {delta} | {r['mean_final_kv_mib']:.1f} | {r['decode_triggered_samples']} |")
    lines.extend(["", "## 固定长度测速（已排除各形状第0次预热）", "",
        "capacity_specific 各方法长度可能不同，不能直接比较其速度。", "",
        "| 模型 | 方法 | 类型 | 输入 | 输出 | 次数 | decode token/s | prefill+压缩 s | 峰值 GiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in performance:
        lines.append(f"| {r['model']} | {r['method']} | {r['sweep']} | {r['input_tokens']} | {r['output_tokens']} | {r['n']} | {r['median_decode_tokens_per_s']:.2f} | {r['median_prefill_and_compress_s']:.3f} | {r['peak_allocated_gib']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return statuses


def parse_list(value, kind=str):
    return [kind(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--qwen", default="Qwen/Qwen2.5-7B-Instruct", help="Local original weights or HF model ID")
    parser.add_argument("--qwen-profile", choices=["Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct"], default="Qwen2.5-7B-Instruct")
    parser.add_argument("--llama", default="meta-llama/Llama-2-7b-chat-hf", help="Local weights or authorized HF model ID")
    parser.add_argument("--models", default="qwen,llama")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--phases", default="capacity,needle,performance,quality")
    parser.add_argument("--tasks", default=",".join(TASK_LIMITS))
    parser.add_argument("--samples", type=int, default=32, help="Per task; 0 = entire task")
    parser.add_argument("--quality-lengths", default="1536,3072,8192,16384")
    parser.add_argument("--lengths", default="1024,2048,3072,4096,8192,16384,24576,32000")
    parser.add_argument("--outputs", default="256,1024,2048,3072,4096")
    parser.add_argument("--compression-config", help="Optional JSON with a methods list; same parameters for both models")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--target-vram", type=float, default=.92)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--needle-samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats after a shape warmup")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--data-dir", default=str(ROOT / ".cache" / "longbench"))
    parser.add_argument("--out", default=str(ROOT / "results" / "4090d_suite"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect plan; no GPU/download required")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--install-deps", action="store_true", help="Install pinned dependencies, leaving torch untouched")
    args = parser.parse_args(argv)
    for key in ("models", "methods", "phases", "tasks"):
        setattr(args, key, parse_list(getattr(args, key)))
    for key in ("quality_lengths", "lengths", "outputs"):
        setattr(args, key, sorted(set(parse_list(getattr(args, key), int))))
    allowed = {"models": {"qwen", "llama"}, "methods": set(METHODS),
               "phases": {"capacity", "needle", "performance", "quality"}, "tasks": set(TASK_LIMITS)}
    for key, choices in allowed.items():
        value = getattr(args, key)
        if not value or len(value) != len(set(value)) or not set(value) <= choices:
            parser.error(f"--{key}: choose unique items from {sorted(choices)}")
    for key in ("quality_lengths", "lengths", "outputs"):
        if not getattr(args, key) or min(getattr(args, key)) < 1:
            parser.error(f"--{key.replace('_', '-')}: require positive integers")
    if not .5 <= args.target_vram <= .97:
        parser.error("--target-vram must be within 0.50..0.97")
    if args.samples < 0 or args.chunk_size < 32 or min(args.repeats, args.needle_samples, args.cpu_threads) < 1 or args.max_context < 256:
        parser.error("Invalid sample/chunk/repeat/context setting")
    args.out = str(Path(args.out).resolve())
    args.data_dir = str(Path(args.data_dir).resolve())
    return args


def profiles_for(args):
    profiles = []
    for alias in args.models:
        name = args.qwen_profile if alias == "qwen" else "Llama-2-7b-chat-hf"
        folder = ROOT / "models" / name
        profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
        profile["architecture"] = json.loads((folder / "architecture.json").read_text(encoding="utf-8"))
        config_path = Path(args.compression_config) if args.compression_config else folder / "compression.json"
        all_methods = json.loads(config_path.read_text(encoding="utf-8"))["methods"]
        by_name = {spec["method"]: spec for spec in all_methods}
        if len(by_name) != len(all_methods):
            raise ValueError("Duplicate method names in compression configuration")
        profile["methods"] = [by_name[name] for name in args.methods]
        for spec in profile["methods"]:
            validate_spec(spec, args.chunk_size)
        source = args.qwen if alias == "qwen" else args.llama
        profile["model_path"] = str(Path(source).resolve()) if Path(source).exists() else source
        profiles.append(profile)
    return profiles


def validate_spec(spec, chunk_size):
    allowed = {"full": set(), "kivi": {"bits", "group_size", "recent_window"},
        "snapkv": {"max_cache_tokens", "prefill_keep_ratio", "obs_window", "pool_kernel", "pooling"},
        "pyramidkv": {"max_cache_tokens", "prefill_keep_ratio", "obs_window", "pool_kernel", "pooling", "beta"},
        "scope": {"max_cache_tokens", "recent_window", "decoding_metric"},
        "phasekv": {"prompt_bits", "cold_bits", "prefill_keep_ratio", "decode_keep_ratio", "sink_tokens",
                    "obs_window", "recent_window", "lazy_g", "group_size"}}
    method = spec["method"]
    unknown = set(spec) - allowed[method] - {"method"}
    if unknown:
        raise ValueError(f"Unsupported {method} parameters: {unknown}")
    for key in ("prefill_keep_ratio", "decode_keep_ratio"):
        if key in spec and not 0 < spec[key] <= 1:
            raise ValueError(f"Invalid {method}/{key}")
    if not 1 <= spec.get("obs_window", 32) <= min(96, chunk_size):
        raise ValueError("This runner validates observation windows up to 96 and <= chunk-size")
    if any(spec.get(key, 2) not in (2, 4, 8) for key in ("bits", "prompt_bits", "cold_bits")):
        raise ValueError("Packed bit width must be 2, 4, or 8")
    # Keep mandatory boundary checks practical; other parameters need a new validation protocol.
    if method == "phasekv":
        if spec.get("recent_window", 32) not in (32, 64, 128, 256) or spec.get("lazy_g", 16) not in (16, 32, 64):
            raise ValueError("PhaseKV validates hot windows 32/64/128/256 and cooling batches 16/32/64")
        if spec.get("group_size", 32) != 32:
            raise ValueError("This suite validates PhaseKV group_size=32")
    if method == "kivi" and spec.get("recent_window", 64) != 64:
        raise ValueError("This suite validates KIVI recent_window=64")
    if method == "scope" and (spec.get("recent_window", 16), spec.get("max_cache_tokens", 32), spec.get("decoding_metric", "slide")) != (16, 32, "slide"):
        raise ValueError("This suite validates SCOPE Slide, recent_window=16, decode budget=32")


def source_hashes():
    files = {Path(__file__), ROOT / "run_4090d.sh"}
    for folder in ("PhaseKV", "KIVI", "SnapKV", "PyramidKV", "SCOPE", "models"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix in (".py", ".json") and not set(path.parts) & {"official", "reference_snapshots", "__pycache__"}:
                files.add(path)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in sorted(files)}


def check_sources():
    required = ["PhaseKV/phasekv_cache.py", "PhaseKV/hetero_cache.py", "PhaseKV/kernels/attention_triton.py",
                "KIVI/packed_cache.py", "KIVI/kernels.py", "SnapKV/compressor.py", "PyramidKV/compressor.py",
                "SCOPE/compressor.py", "models/Qwen2/qwen2_attention.py", "models/Qwen2/kivi_attention.py",
                "models/Llama/llama_attention.py", "models/Llama/kivi_attention.py", "models/packed_runtime.py"]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("Upload algorithm/model directories too: " + ", ".join(missing))


def check_environment():
    import torch
    import transformers
    import triton
    import rouge
    if sys.platform != "linux":
        raise RuntimeError("CUDA benchmark requires Linux; --dry-run works on Windows")
    if transformers.__version__ != "4.46.3":
        raise RuntimeError("Requires transformers==4.46.3; use --install-deps")
    if torch.__version__.split("+")[0] != "2.5.1" or triton.__version__ != "3.1.0":
        raise RuntimeError("Requires torch==2.5.1 and triton==3.1.0; see BENCHMARK_4090D.md")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and native BF16 support are required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU with CUDA_VISIBLE_DEVICES=0")
    print(f"GPU: {torch.cuda.get_device_name()}, {torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB", flush=True)


def checkpoint_identities(profiles):
    from huggingface_hub import snapshot_download
    result = {}
    for profile in profiles:
        source = profile["model_path"]
        remote = not Path(source).is_dir()
        path = Path(snapshot_download(source, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"])) if remote else Path(source)
        weights = sorted(path.glob("*.safetensors"))
        if not weights:
            raise ValueError(f"Original safetensors checkpoint required: {source}")
        files = weights + sorted(p for p in path.iterdir() if p.suffix in (".json", ".model", ".txt", ".tiktoken"))
        print(f"Hashing checkpoint: {profile['name']} ({len(weights)} weight files)", flush=True)
        result[profile["name"]] = {"source": source, "resolved_path": str(path.resolve()),
            "snapshot_revision": path.name if remote else None,
            "files_sha256": {p.name: file_hash(p) for p in files}}
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.report_only:
        print(summarize(args.out))
        return 0
    check_sources()
    profiles = profiles_for(args)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "settings": vars(args),
            "models": [{"name": p["name"], "source": p["model_path"],
                        "context_limit": p["architecture"]["max_position_embeddings"], "methods": p["methods"]} for p in profiles],
            "notes": ["No dependency on missing eval/ or baselines/", "Mandatory GPU numerical checks",
                      "Llama-2 input + output <= 4096", "--samples 0 evaluates complete tasks",
                      "Default suite can take many hours; each sample is checkpointed"]}, ensure_ascii=False, indent=2))
        return 0
    if args.download_only:
        print(json.dumps(prepare_data(args.data_dir, args.tasks), indent=2))
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError(f"Output exists: {out}. Use --resume or a new --out")
    lock_path = out / "RUNNING.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Check PID in {lock_path}; remove stale lock only after confirming it stopped.")
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)
    try:
        if args.install_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.46.3", "tokenizers==0.20.3",
                "accelerate==0.34.2", "huggingface_hub==0.36.2", "safetensors==0.8.0", "numpy==1.26.3",
                "sentencepiece==0.2.0", "rouge==1.0.1"], check=True)
        subprocess.run([sys.executable, str(Path(__file__)), "__check_environment__"], check=True)
        data = prepare_data(args.data_dir, args.tasks) if "quality" in args.phases else {}
        settings = {k: v for k, v in vars(args).items() if k not in
                    ("resume", "install_deps", "dry_run", "report_only", "download_only")}
        manifest = {"protocol": PROTOCOL, "settings": settings, "profiles": profiles,
                    "source_hashes": source_hashes(), "data": data}
        identity_job, identity_output = out / "identity_job.json", out / "identity_current.json"
        atomic_json(identity_job, profiles)
        subprocess.run([sys.executable, str(Path(__file__)), "__identity__", str(identity_job), str(identity_output)], check=True)
        identities = json.loads(identity_output.read_text(encoding="utf-8"))
        manifest["checkpoints"] = identities
        manifest["protocol_hash"] = digest(manifest)
        if (out / "manifest.json").exists():
            old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            if old["protocol_hash"] != manifest["protocol_hash"]:
                raise ValueError("Resume rejected: code/data/checkpoint/settings changed. Choose a new --out.")
        else:
            atomic_json(out / "manifest.json", manifest)
        validation_job = out / "validation_job.json"
        atomic_json(validation_job, profiles)
        subprocess.run([sys.executable, str(Path(__file__)), "__validate__", str(validation_job),
                        str(out / f"gpu_validation_{time.time_ns()}.json")], check=True)
        for folder in ("records", "logs", "jobs"):
            (out / folder).mkdir(exist_ok=True)
        failures = []
        state = {"status": "running", "finished": [], "failed": failures}
        atomic_json(out / "run_state.json", state)
        for profile in profiles:
            for spec in profile["methods"]:
                name = profile["name"] + "__" + spec["method"]
                job = {"name": profile["name"], "model_path": identities[profile["name"]]["resolved_path"],
                       "architecture": profile["architecture"], "spec": spec,
                       "args": settings, "protocol_hash": manifest["protocol_hash"],
                       "records_path": str(out / "records" / f"{name}.jsonl")}
                job_path = out / "jobs" / f"{name}.json"
                atomic_json(job_path, job)
                print(f"\nStarting {name}. Ctrl-C is resumable.", flush=True)
                log_path = out / "logs" / f"{name}.log"
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "__worker__", str(job_path)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                if code:
                    failures.append({"job": name, "exit_code": code, "log": str(log_path)})
                else:
                    state["finished"].append(name)
                atomic_json(out / "run_state.json", state)
                summarize(out)
        state["status"] = "finished_with_errors" if failures else "complete"
        atomic_json(out / "run_state.json", state)
        print(f"\n{state['status']}: {out / 'summary.md'}", flush=True)
        return 1 if failures else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "__check_environment__":
            check_environment()
        elif len(sys.argv) > 1 and sys.argv[1] == "__identity__":
            atomic_json(sys.argv[3], checkpoint_identities(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))))
        elif len(sys.argv) > 1 and sys.argv[1] == "__validate__":
            checks = gpu_validation(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), report_path=sys.argv[3])
            atomic_json(sys.argv[3], {"status": "passed", "validation_protocol": VALIDATION_PROTOCOL, "checks": checks})
            print("GPU numerical validation PASSED", flush=True)
        elif len(sys.argv) > 1 and sys.argv[1] == "__worker__":
            worker(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
        else:
            raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same arguments plus --resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
