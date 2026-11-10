"""Small single-prompt runner shared by model-specific PhaseKV entry points."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .phasekv_cache import PhaseKVCache
from .prompt_tier import select_prompt_indices


def add_arguments(parser: argparse.ArgumentParser, default_model: str) -> None:
    parser.add_argument("--model", default=default_model, help="Hugging Face 模型 ID 或本地权重目录")
    parser.add_argument("--prompt", default="Please briefly explain what a KV cache is.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-keep-ratio", type=float, default=1.0)
    parser.add_argument("--prompt-bits", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--cold-bits", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--lazy-g", type=int, default=16)
    parser.add_argument("--decode-keep-ratio", type=float, default=1.0)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-cuda", action="store_true", help="只验证 CUDA，不下载模型")


def validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.max_new_tokens <= 0 or args.recent_window <= 0 or args.lazy_g <= 0 or args.group_size <= 0:
        parser.error("max-new-tokens、recent-window、lazy-g、group-size 必须为正")
    if not 0 < args.prefill_keep_ratio <= 1 or not 0 < args.decode_keep_ratio <= 1:
        parser.error("保留比例必须位于 (0, 1]")


def check_cuda() -> str:
    if torch.version.cuda is None:
        raise RuntimeError("当前 PyTorch 不含 CUDA")
    try:
        torch.cuda.init()
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(f"当前进程无法使用 CUDA：{exc}") from exc
    return torch.cuda.get_device_name()


def load_model(args, model_type: str):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    if model.config.model_type != model_type:
        raise TypeError(f"{args.model} 的 model_type={model.config.model_type!r}，需要 {model_type!r}")
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str, *, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        )
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    if ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("prompt 必须产生一个非空、无 padding 的序列")
    return ids.to("cuda")


def pack_prefill(prefill, dense_cache, model, prompt_len: int, args) -> PhaseKVCache:
    keep = None
    if args.prefill_keep_ratio < 1:
        keep = select_prompt_indices(
            prefill.attentions, prompt_len=prompt_len, model_config=model.config,
            prefill_keep_ratio=args.prefill_keep_ratio, sink_tokens=4,
            obs_window=32, pool_kernel=7, pooling="maxpool",
        )
    return PhaseKVCache.from_dense_prefill(
        dense_cache, prompt_len=prompt_len, keep_indices=keep,
        prompt_bits=args.prompt_bits, cold_bits=args.cold_bits,
        group_size=args.group_size, recent_window=args.recent_window,
        lazy_g=args.lazy_g, decode_keep_ratio=args.decode_keep_ratio,
    )


@torch.inference_mode()
def generate(model, tokenizer, input_ids, args, *, dense_prefill, packed_step) -> tuple[str, PhaseKVCache]:
    """Return generated text and cache; packed_step advances one non-EOS token."""
    prompt_len = input_ids.shape[1]
    max_positions = model.config.max_position_embeddings
    if prompt_len + args.max_new_tokens > max_positions:
        raise ValueError(
            f"prompt {prompt_len} + 生成 {args.max_new_tokens} 超过模型位置上限 {max_positions}"
        )
    prefill = dense_prefill(model, input_ids, args.prefill_keep_ratio < 1)
    dense_cache = prefill.past_key_values
    if not hasattr(dense_cache, "key_cache"):
        from transformers.cache_utils import DynamicCache
        dense_cache = DynamicCache.from_legacy_cache(dense_cache)
    cache = pack_prefill(prefill, dense_cache, model, prompt_len, args)
    logits = prefill.logits[:, -1, :].clone()
    del prefill, dense_cache

    eos = model.generation_config.eos_token_id
    eos_ids = set(int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos]) if x is not None)
    generated = []
    for _ in range(args.max_new_tokens):
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("生成 logits 含 NaN/Inf")
        token = logits.argmax(dim=-1)
        token_id = int(token.item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        logits = packed_step(model, cache, token)
    return tokenizer.decode(generated, skip_special_tokens=True), cache
