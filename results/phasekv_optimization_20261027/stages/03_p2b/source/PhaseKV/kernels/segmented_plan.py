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
    valid_count: int
    cold_count: int
    extension_count: int

    @property
    def partial_count(self):
        return self.valid_count + 1

    @property
    def capacity(self):
        return self.maxima.shape[1]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.descriptors, self.maxima, self.denominators, self.accumulators))


def _capacity(count, reuse):
    return max(32, 1 << (count - 1).bit_length()) if reuse else count


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
    capacity = _capacity(len(rows) + 1, tiers.optimizations.reuse_workspace)
    descriptors = torch.empty((capacity - 1, DESCRIPTOR_WIDTH), dtype=torch.int64, device=device)
    if rows:
        descriptors[:len(rows)].copy_(torch.tensor(rows, dtype=torch.int64, device=device))
    return DecodeWorkspace(
        key, owners, descriptors,
        torch.empty((heads, capacity), dtype=torch.float32, device=device),
        torch.empty((heads, capacity), dtype=torch.float32, device=device),
        torch.empty((heads, capacity, head_dim), dtype=torch.float32, device=device),
        len(rows), len(tiers.cold), len(tiers.prompt_extensions),
    )


def ensure_workspace(tiers, heads, head_dim, device, sm_count, key,
                     num_splits=None, block_n=None, num_warps=None):
    """Append only new cold descriptors; old pointer owners remain alive.

    Prompt insertion/reorder/stream or launch changes rebuild the table. Frozen
    segment contents are immutable by contract. No synchronization is added.
    """
    workspace = tiers._decode_workspace
    if (workspace is None or workspace.key != key
            or workspace.owners[0] is not tiers.prompt
            or workspace.extension_count != len(tiers.prompt_extensions)
            or workspace.cold_count > len(tiers.cold)):
        workspace = build_workspace(tiers, heads, head_dim, device, sm_count, key,
                                    num_splits, block_n, num_warps)
    elif workspace.cold_count != len(tiers.cold):
        if not tiers.optimizations.reuse_workspace:
            workspace = build_workspace(tiers, heads, head_dim, device, sm_count, key,
                                        num_splits, block_n, num_warps)
        else:
            added = tuple(tiers.cold[workspace.cold_count:])
            rows = []
            for segment in added:
                if segment.key.orig_shape[1] != tiers.hot_k.shape[1]:
                    raise ValueError("All tiers must have the same KV head count")
                rows.extend(segment_rows(segment, heads, head_dim, device, sm_count,
                                         num_splits, block_n, num_warps))
            count = workspace.valid_count + len(rows)
            if count + 1 > workspace.capacity:
                capacity = _capacity(count + 1, True)
                descriptors = torch.empty((capacity - 1, DESCRIPTOR_WIDTH), dtype=torch.int64, device=device)
                descriptors[:workspace.valid_count].copy_(workspace.descriptors[:workspace.valid_count])
                workspace.descriptors = descriptors
                workspace.maxima = torch.empty((heads, capacity), device=device, dtype=torch.float32)
                workspace.denominators = torch.empty_like(workspace.maxima)
                workspace.accumulators = torch.empty((heads, capacity, head_dim), device=device, dtype=torch.float32)
            if rows:
                workspace.descriptors[workspace.valid_count:count].copy_(
                    torch.tensor(rows, dtype=torch.int64, device=device))
            workspace.valid_count = count
            workspace.cold_count = len(tiers.cold)
            workspace.owners += added
    tiers._decode_workspace = workspace
    return workspace
