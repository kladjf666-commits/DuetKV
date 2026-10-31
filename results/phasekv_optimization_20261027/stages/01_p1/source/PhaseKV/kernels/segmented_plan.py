"""Cached pointer table for immutable PhaseKV segments (no Triton import).

Each row describes one nonempty split of one segment. Addresses are GPU
addresses, not copied/dequantized KV. The owner keeps referenced tensors alive.
Cold segments are usually tiny, so one block per head/segment is sufficient;
long prompt segments still use the existing split planner.
"""
from dataclasses import dataclass

import torch

from .launch_config import plan_kivi_decode


DESCRIPTOR_WIDTH = 24


def segment_rows(segment, heads, head_dim, device, sm_count,
                 num_splits=None, block_n=None, num_warps=None):
    k, v = segment.key, segment.value
    if (len(k.orig_shape) != 4 or k.orig_shape[0] != 1
            or v.orig_shape != k.orig_shape or k.orig_shape[2] <= 0
            or k.orig_shape[3] != head_dim or heads % k.orig_shape[1]
            or k.axis != 2 or v.axis != 3):
        raise ValueError("Expected batch=1 K/V with token-axis K and channel-axis V")
    for packed in (k, v):
        if packed.bits not in (2, 4, 8) or packed.group_size <= 0:
            raise ValueError("Unsupported quantization parameters")
        if (packed.packed.dtype != torch.uint8 or packed.scale.dtype != torch.float16
                or packed.mins.dtype != torch.float16
                or packed.scale.stride() != packed.mins.stride()
                or packed.scale.shape != packed.mins.shape):
            raise ValueError("Expected uint8 codes and matching FP16 scale/min layouts")
        if any(t.device != device for t in (packed.packed, packed.scale, packed.mins)):
            raise ValueError("All packed tensors must be on the query device")
    config = plan_kivi_decode(k.orig_shape[2], heads, head_dim, sm_count,
                              num_splits, block_n, num_warps)
    addresses = [t.data_ptr() for t in (k.packed, k.scale, k.mins,
                                       v.packed, v.scale, v.mins)]
    layout = [k.bits, v.bits, k.group_size, v.group_size,
              *k.packed.stride()[1:], *k.scale.stride()[1:],
              *v.packed.stride()[1:], *v.scale.stride()[1:]]
    rows = []
    for split in range(config.num_splits):
        start = split * config.split_len
        rows.append(addresses + [start, min(start + config.split_len, k.orig_shape[2])]
                    + layout)
    return rows


@dataclass
class DecodeWorkspace:
    key: tuple
    owners: tuple
    descriptors: torch.Tensor
    maxima: torch.Tensor
    denominators: torch.Tensor
    accumulators: torch.Tensor

    @property
    def partial_count(self):
        # One extra partial always represents hot KV, even when it is empty.
        return self.descriptors.shape[0] + 1


def build_workspace(tiers, heads, head_dim, device, sm_count, key,
                    num_splits=None, block_n=None, num_warps=None):
    owners = (tiers.prompt, *tiers.prompt_extensions, *tiers.cold)
    rows = []
    hkv = tiers.hot_k.shape[1]
    for segment in owners:
        if segment.key.orig_shape[1] != hkv:
            raise ValueError("All tiers must have the same KV head count")
        rows.extend(segment_rows(segment, heads, head_dim, device, sm_count,
                                 num_splits, block_n, num_warps))
    descriptors = torch.tensor(rows, dtype=torch.int64, device=device)
    count = len(rows) + 1
    return DecodeWorkspace(
        key, owners, descriptors,
        torch.empty((heads, count), dtype=torch.float32, device=device),
        torch.empty((heads, count), dtype=torch.float32, device=device),
        torch.empty((heads, count, head_dim), dtype=torch.float32, device=device),
    )
