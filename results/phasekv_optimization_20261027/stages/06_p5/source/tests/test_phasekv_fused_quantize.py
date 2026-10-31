"""Byte equality, including FP16 metadata underflow; CUDA tests never interpret."""
import importlib.util
import unittest
import sys
import torch
from scripts.benchmark_4090d import bootstrap_algorithms
if "eval" not in sys.modules:
    bootstrap_algorithms()
from PhaseKV.kernels.packing import (allocate_packed, quantize_pack,
    quantize_pack_fast, quantize_pack_into)


class QuantizeContract:
    def check_case(self, x, axis, gs=32):
        expected = quantize_pack(x, 4, axis, gs)
        actual = quantize_pack_fast(x, 4, axis, gs)
        out = allocate_packed(x.shape, 4, axis, gs, x.device)
        quantize_pack_into(x, out, fast=True)
        for result in (actual, out):
            self.assertEqual((result.orig_shape, result.axis, result.group_size),
                             (expected.orig_shape, expected.axis, expected.group_size))
            for name in ("packed", "scale", "mins"):
                a, b = getattr(result, name), getattr(expected, name)
                self.assertTrue(torch.equal(a.contiguous().view(torch.uint8),
                                            b.contiguous().view(torch.uint8)), name)

    def run_matrix(self, device):
        torch.manual_seed(31)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for length in (1, 7, 15, 16, 17, 31, 32, 33, 717, 4096):
                x = torch.randn(1, 2, length, 64, device=device, dtype=dtype)[..., ::2]
                for axis in (2, 3):
                    with self.subTest(dtype=dtype, length=length, axis=axis):
                        self.check_case(x, axis)
        # Constant groups, tiny ranges, and exact/near round-to-even boundaries.
        for values in ([0.] * 32, [1.] * 32, [0., 1.e-9] * 16,
                       [0., 15.] + [i + .5 for i in range(15)] + [0.] * 15):
            x = torch.tensor(values, device=device, dtype=torch.float32).reshape(1, 1, 1, 32)
            for axis in (2, 3):
                self.check_case(x, axis)


class TestQuantizeCPU(unittest.TestCase, QuantizeContract):
    def test_reference_fallback(self):
        self.run_matrix("cpu")

    def test_other_bits_and_odd_groups(self):
        x = torch.randn(1, 2, 33, 32, dtype=torch.float16)
        for bits in (2, 8):
            self.assertTrue(torch.equal(quantize_pack_fast(x, bits, 2, 32).packed,
                                        quantize_pack(x, bits, 2, 32).packed))
        self.check_case(x, 2, 7)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"),
                     "requires actual CUDA + Triton")
class TestQuantizeCUDA(unittest.TestCase, QuantizeContract):
    def test_raw_byte_equivalence(self):
        self.run_matrix("cuda")

    def test_ring_batch_and_strided_outputs(self):
        for start in (0, 1, 127, 137):
            x = torch.randn(1, 4, 144, 128, device="cuda", dtype=torch.float16)
            logical = x.index_select(2, (torch.arange(16, device="cuda") + start) % 144)
            for axis in (2, 3):
                expected = quantize_pack(logical, 4, axis, 32)
                out = allocate_packed(logical.shape, 4, axis, 32, x.device)
                # Noncontiguous destination views must use their real strides.
                for name in ("packed", "scale", "mins"):
                    t = getattr(out, name)
                    backing = torch.empty((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype, device=t.device)
                    setattr(out, name, backing[..., ::2])
                quantize_pack_into(x, out, fast=True, token_start=start, token_count=16)
                for name in ("packed", "scale", "mins"):
                    self.assertTrue(torch.equal(getattr(out, name).contiguous().view(torch.uint8),
                                                getattr(expected, name).contiguous().view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
