"""PhaseKV compatibility boundary for DeepSeek-V4-Flash.

DeepSeek V4's CSA/HCA pools, sparse indexer, shared K=V, partial RoPE,
attention sinks and grouped output projection require a model-specific cache
and kernel. The existing PhaseKVCache contract represents dense per-token K/V
and cannot be installed on this model without changing its computation.
"""


def install_deepseek_v4_flash_phasekv_attention(model) -> None:
    """Fail explicitly until a validated CSA/HCA PhaseKV implementation exists."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "deepseek_v4":
        raise TypeError(f"需要 model_type='deepseek_v4'，实际为 {model_type!r}")
    raise NotImplementedError(
        "DeepSeek V4 Flash 的 CSA/HCA 混合注意力和索引缓存尚未接入 PhaseKV。"
        "请使用本目录的 vLLM 原生部署入口；不要套用 Qwen/Llama 适配器。"
    )
