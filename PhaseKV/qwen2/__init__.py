"""Compatibility package for models.Qwen2."""
from importlib import import_module as _import_module
_impl = _import_module("models.Qwen2")
__all__ = getattr(_impl, "__all__", [])
def __getattr__(name):
    return getattr(_impl, name)
