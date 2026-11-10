"""PhaseKV algorithm package.

phasekv_cache.py and kernels/ implement genuine packed KV storage and attention.
compressor.py / quant.py retain the historical fake-quant evaluation path for
compatibility; the real benchmark explicitly forbids that path. Model adapters
are grouped under qwen2/, llama/, and opt/.
"""
from __future__ import annotations

from eval.compressors import register

from .compressor import PhaseKVCompressor
from .phasekv_cache import PhaseKVCache

register("phasekv", PhaseKVCompressor)

__all__ = ["PhaseKVCompressor", "PhaseKVCache", "install_qwen2_phasekv_attention"]


def __getattr__(name):
    # Compatibility for existing clients; model-specific code lives in qwen2/.
    # Load it only on request so the shared cache package does not import models.
    if name == "install_qwen2_phasekv_attention":
        from PhaseKV.qwen2.qwen2_attention import install_qwen2_phasekv_attention
        return install_qwen2_phasekv_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
