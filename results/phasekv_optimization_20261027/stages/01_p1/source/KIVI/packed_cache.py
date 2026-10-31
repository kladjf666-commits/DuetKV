"""Packed KIVI with independent key/value residual schedules.

Schedule follows https://github.com/jy-yuan/KIVI/blob/main/models/llama_kivi.py:
K flushes a complete residual_length block; V retains a sliding residual window.
Quantized tensors are uint8 bit packs with FP16 group scales/minima. No dense shadow.
"""
from dataclasses import dataclass, replace

import torch
from transformers.cache_utils import Cache

from PhaseKV.kernels.packing import PackedTensor, quantize_pack


def append_packed(old, new):
    if old is None:
        return new
    if (old.axis, old.bits, old.group_size) != (new.axis, new.bits, new.group_size):
        raise ValueError("Incompatible packed blocks")
    dim = -1 if old.axis == 2 else 2
    shape = list(old.orig_shape)
    shape[2] += new.orig_shape[2]
    return replace(old, packed=torch.cat((old.packed, new.packed), dim=dim),
                   scale=torch.cat((old.scale, new.scale), dim=dim),
                   mins=torch.cat((old.mins, new.mins), dim=dim),
                   orig_shape=tuple(shape),
                   axis_len=shape[old.axis])


@dataclass
class KIVILayer:
    key: PackedTensor | None
    value: PackedTensor | None
    key_residual: torch.Tensor
    value_residual: torch.Tensor

    @property
    def length(self):
        return (self.key.orig_shape[2] if self.key else 0) + self.key_residual.shape[2]


class PackedKIVICache(Cache):
    def __init__(self, layers, *, seen_tokens, bits=2, group_size=32, residual_length=64):
        super().__init__()
        self.layers = layers
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length
        self._seen_tokens = seen_tokens
        self._pending = set()

    @classmethod
    @torch.no_grad()
    def from_dense_prefill(cls, dense, *, bits=2, group_size=32, residual_length=64):
        if bits not in (2, 4, 8) or group_size <= 0 or residual_length <= 0:
            raise ValueError("Invalid packed KIVI configuration")
        if residual_length % group_size or group_size % (8 // bits):
            raise ValueError("K residual blocks must align to quantization groups and packed bytes")
        layers = []
        length = dense.key_cache[0].shape[2]
        for key, value in zip(dense.key_cache, dense.value_cache):
            if key.dtype not in (torch.float16, torch.bfloat16) or key.shape != value.shape or key.shape[0] != 1:
                raise ValueError("Packed KIVI requires batch=1 FP16/BF16 K/V with matching shapes")
            key, value = key.to(torch.float16), value.to(torch.float16)
            if not bool(torch.isfinite(key).all() & torch.isfinite(value).all()):
                raise ValueError("KIVI K/V overflow when converting residual storage to FP16")
            if key.shape[3] % group_size or key.shape[2] != length:
                raise ValueError("Head dimension must align to group_size and layers must match")
            nk = length // residual_length * residual_length
            nv = max(0, length - residual_length)
            layers.append(KIVILayer(
                quantize_pack(key[:, :, :nk], bits, axis=2, group_size=group_size) if nk else None,
                quantize_pack(value[:, :, :nv], bits, axis=3, group_size=group_size) if nv else None,
                key[:, :, nk:].clone().contiguous(), value[:, :, nv:].clone().contiguous()))
        return cls(layers, seen_tokens=length, bits=bits, group_size=group_size,
                   residual_length=residual_length)

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx or 0].length

    def get_max_cache_shape(self):
        return None

    @property
    def true_seq_length(self):
        return self._seen_tokens

    def update(self, *args, **kwargs):
        raise RuntimeError("Packed KIVI must use packed attention; dense Cache.update is forbidden")

    def append(self, layer_idx, key, value):
        if layer_idx in self._pending:
            raise RuntimeError("Layer appended twice in one decode step")
        layer = self.layers[layer_idx]
        layer.key_residual = torch.cat((layer.key_residual, key), dim=2)
        layer.value_residual = torch.cat((layer.value_residual, value), dim=2)
        self._pending.add(layer_idx)

    @torch.no_grad()
    def finish_layer(self, layer_idx):
        # Quantize only after attention consumed this step's full-precision residuals.
        layer = self.layers[layer_idx]
        if layer.key_residual.shape[2] == self.residual_length:
            new = quantize_pack(layer.key_residual, self.bits, axis=2, group_size=self.group_size)
            layer.key = append_packed(layer.key, new)
            layer.key_residual = layer.key_residual[:, :, :0].clone()
        if layer.value_residual.shape[2] > self.residual_length:
            excess = layer.value_residual.shape[2] - self.residual_length
            new = quantize_pack(layer.value_residual[:, :, :excess], self.bits,
                                axis=3, group_size=self.group_size)
            layer.value = append_packed(layer.value, new)
            layer.value_residual = layer.value_residual[:, :, excess:].clone().contiguous()

    def finish_decode_step(self):
        if self._pending != set(range(len(self.layers))):
            raise RuntimeError("Not every KIVI layer was updated")
        if any(layer.length != self._seen_tokens + 1 for layer in self.layers):
            raise RuntimeError("Inconsistent KIVI layer lengths")
        self._pending.clear()
        self._seen_tokens += 1

    def physical_nbytes(self):
        return sum((layer.key.nbytes if layer.key else 0)
                   + (layer.value.nbytes if layer.value else 0)
                   + layer.key_residual.untyped_storage().nbytes()
                   + layer.value_residual.untyped_storage().nbytes()
                   for layer in self.layers)
