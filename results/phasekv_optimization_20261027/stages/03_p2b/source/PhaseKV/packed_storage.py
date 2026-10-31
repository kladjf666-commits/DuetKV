"""Append-only typed pages. A slot never changes its quantization boundary.

Each layout has its own pool, including short tails and mixed bit/group sizes.
Segments own views; the pool owns full allocations including unused slots.
Reordering detaches existing views and resets the pool for future appends.
"""
from dataclasses import dataclass
import torch
from .kernels.packing import PackedTensor


def storage_identity(tensor):
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage._cdata)


def unique_storage_bytes(tensors):
    seen = set()
    total = 0
    for tensor in tensors:
        identity = storage_identity(tensor)
        if identity not in seen:
            total += tensor.untyped_storage().nbytes()
            seen.add(identity)
    return total


@dataclass
class PackedPage:
    tensors: tuple
    used: int = 0


class PackedPagePool:
    def __init__(self, slots=32):
        if type(slots) is not int or slots <= 0:
            raise ValueError("page slots must be positive")
        self.slots = slots
        self.pages = {}

    def reserve(self, shape, bits, group_size, device):
        """Return K/V PackedTensor views into a fresh slot, without quantizing."""
        shape = tuple(shape)
        device = torch.device(device)
        signature = (shape, bits, group_size, device)
        pages = self.pages.setdefault(signature, [])
        layouts = []
        for axis in (2, 3):
            length = shape[axis]
            gs = length if group_size is None or group_size <= 0 else min(group_size, length)
            prefix = tuple(n for i, n in enumerate(shape) if i != axis)
            layouts.append((axis, length, gs, (*prefix, (length + 8 // bits - 1) // (8 // bits)),
                            (*prefix, (length + gs - 1) // gs)))
        if not pages or pages[-1].used == self.slots:
            tensors = []
            for _, _, _, codes, meta in layouts:
                for size, dtype in ((codes, torch.uint8), (meta, torch.float16), (meta, torch.float16)):
                    tensors.append(torch.empty((self.slots, *size), dtype=dtype, device=device))
            pages.append(PackedPage(tuple(tensors)))
        page = pages[-1]
        slot = page.used
        page.used += 1
        packed = []
        for i, (axis, length, gs, _, _) in enumerate(layouts):
            packed.append(PackedTensor(*(t[slot] for t in page.tensors[i * 3:i * 3 + 3]),
                                       bits, axis, gs, shape, length))
        return tuple(packed)

    def storage_tensors(self):
        for pages in self.pages.values():
            for page in pages:
                for i, tensor in enumerate(page.tensors):
                    yield ("codes" if i % 3 == 0 else "metadata"), tensor
