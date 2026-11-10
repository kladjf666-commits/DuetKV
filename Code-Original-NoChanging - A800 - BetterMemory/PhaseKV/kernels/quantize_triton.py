"""Opt-in four-bit quantization directly from strided/ring FP storage.

One program owns an entire quantization group and its output bytes. Odd group
boundaries that share a byte are rejected by the dispatcher, never raced.
This module is imported only for supported CUDA inputs.
"""
import triton
import triton.language as tl
try:
    from triton.language.extra.cuda import libdevice as device_math
except ImportError:  # Triton 2.1 ships libdevice functions in language.math.
    from triton.language import math as device_math


@triton.jit
def _quantize_4bit(X, P, S, M,
                   X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr,
                   H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                   CAP: tl.constexpr, START, AXIS: tl.constexpr,
                   LENGTH: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                   P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
                   M0: tl.constexpr, M1: tl.constexpr, M2: tl.constexpr, M3: tl.constexpr,
                   BLOCK: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    width: tl.constexpr = D if AXIS == 2 else T
    b, h, outer = row // (H * width), (row // width) % H, row % width
    i = tl.arange(0, BLOCK)
    logical = tl.minimum(group * GS + i, LENGTH - 1)
    if AXIS == 2:
        address = b * X0 + h * X1 + ((START + logical) % CAP) * X2 + outer * X3
    else:
        address = b * X0 + h * X1 + ((START + outer) % CAP) * X2 + logical * X3
    x = tl.load(X + address, i < GS, other=0).to(tl.float32)
    lo = tl.min(tl.where(i < GS, x, float("inf")), 0)
    hi = tl.max(tl.where(i < GS, x, -float("inf")), 0)
    # PyTorch CUDA true_divide by a CPU scalar uses an FP32 reciprocal multiply.
    # Tensor/tensor division below is correctly rounded, with no approximate rcp.
    scale = tl.maximum(hi - lo, 1.e-8) * (1.0 / 15.0)
    ratio = device_math.div_rn(x - lo, scale)
    rounded = device_math.nearbyint(ratio)
    codes = tl.minimum(tl.maximum(rounded, 0.), 15.).to(tl.int32)
    codes = tl.where((i < GS) & (group * GS + i < LENGTH), codes, 0)
    pairs = tl.reshape(codes, (BLOCK // 2, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), axis=1)
    j = tl.arange(0, BLOCK // 2)
    byte = group * GS // 2 + j
    tl.store(P + b * P0 + h * P1 + outer * P2 + byte * P3,
             packed, (2 * j < GS) & (group * GS + 2 * j < LENGTH))
    tl.store(S + b * S0 + h * S1 + outer * S2 + group * S3, scale)
    tl.store(M + b * M0 + h * M1 + outer * M2 + group * M3, lo)


def quantize_into(x, out, token_start=0, token_count=None):
    b, h, capacity, d = x.shape
    tokens = capacity if token_count is None else token_count
    rows = b * h * (d if out.axis == 2 else tokens)
    _quantize_4bit[(rows, out.scale.shape[-1])](
        x, out.packed, out.scale, out.mins, *x.stride(), h, tokens, d,
        capacity, token_start, out.axis, out.axis_len, out.group_size,
        out.scale.shape[-1], *out.packed.stride(), *out.scale.stride(), *out.mins.stride(),
        BLOCK=max(2, triton.next_power_of_2(out.group_size)),
        num_warps=4, enable_fp_fusion=False)
    return out
