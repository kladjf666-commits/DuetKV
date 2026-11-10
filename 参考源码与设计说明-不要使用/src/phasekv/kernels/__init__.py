"""Compatibility package; implementation lives in PhaseKV.kernels."""
from importlib import import_module as _import_module
_impl = _import_module("PhaseKV.kernels")
__all__ = getattr(_impl, "__all__", [])


def __getattr__(name):
    return getattr(_impl, name)
