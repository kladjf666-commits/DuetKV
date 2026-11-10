"""Qwen2 model integration; run the chat entry point with ``python -m qwen2``.

New callers should import the entry module from ``qwen2.qwen2`` and the
adapter from ``qwen2.qwen2_attention``. Lazy attribute access preserves reads
such as ``import qwen2; qwen2.MODEL_ID`` used by existing benchmark clients.
"""
from importlib import import_module


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".qwen2", __name__), name)
