"""Two-launch PhaseKV decode, independent of the number of frozen segments.

The first launch maps blocks to query heads and descriptor ranges, including a
hot block. Optional GQA lanes share each KV tile across several queries. The
second launch performs stable softmax reduction over all valid partials.
There is no concatenation, requantization, or dense KV intermediate. Scratch
and pointer descriptors have independent capacities and valid counts. Immutable
cold additions update only the new rows when workspace reuse is enabled.
"""
import torch
import triton
import triton.language as tl

from .launch_config import plan_kivi_decode
from .segmented_plan import DESCRIPTOR_WIDTH, ensure_workspace


@triton.jit
def _segmented_attention(Q, DESC, HK, HV, PM, PL, PA,
                         NPACKED, NPARTIAL, HOT,
                         QH, QD, KHH, KHS, KHD, VHH, VHS, VHD,
                         HD: tl.constexpr, GROUP: tl.constexpr,
                         BN: tl.constexpr, DW: tl.constexpr, CAPACITY: tl.constexpr = 0,
                         PREFIX=0, COLD_GROUP: tl.constexpr = 1,
                         NROWS=0, FOUR_BIT: tl.constexpr = False,
                         QPER: tl.constexpr = 1, HOT_CAP: tl.constexpr = 0, HOT_START=0):
    stride = CAPACITY if CAPACITY else NPARTIAL
    lane = tl.arange(0, QPER)
    if QPER == 1:
        heads = tl.program_id(0) + lane
        kvhead = tl.program_id(0) // GROUP
        live = lane == 0
    else:
        kvhead = tl.program_id(0)
        heads = kvhead * GROUP + lane
        live = lane < GROUP
    part = tl.program_id(1)
    d = tl.arange(0, HD)
    query = tl.load(Q + heads[:, None] * QH + d[None, :] * QD,
                    mask=live[:, None], other=0).to(tl.float32)
    maximum = tl.full((QPER,), -float("inf"), tl.float32)
    denominator = tl.full((QPER,), 0., tl.float32)
    acc = tl.full((QPER, HD), 0., tl.float32)
    if part < NPACKED:
        first_row = tl.where(part < PREFIX, part, PREFIX + (part - PREFIX) * COLD_GROUP)
        last_row = tl.minimum(first_row + tl.where(part < PREFIX, 1, COLD_GROUP),
                              tl.where(NROWS > 0, NROWS, NPACKED))
        for row_index in range(first_row, last_row):
            row = DESC + row_index * DW
            kp = tl.load(row + 0).to(tl.pointer_type(tl.uint8))
            ks = tl.load(row + 1).to(tl.pointer_type(tl.float16))
            km = tl.load(row + 2).to(tl.pointer_type(tl.float16))
            vp = tl.load(row + 3).to(tl.pointer_type(tl.uint8))
            vs = tl.load(row + 4).to(tl.pointer_type(tl.float16))
            vm = tl.load(row + 5).to(tl.pointer_type(tl.float16))
            start = tl.load(row + 6).to(tl.int32)
            end = tl.load(row + 7).to(tl.int32)
            if FOUR_BIT:
                kb, vb, vgs = 4, 4, 32
                kgs = tl.where(row_index < PREFIX, 32, 16)
            else:
                kb = tl.load(row + 8).to(tl.int32)
                vb = tl.load(row + 9).to(tl.int32)
                kgs = tl.load(row + 10).to(tl.int32)
                vgs = tl.load(row + 11).to(tl.int32)
            kph, kpd, kpb = tl.load(row + 12), tl.load(row + 13), tl.load(row + 14)
            ksh, ksd, ksg = tl.load(row + 15), tl.load(row + 16), tl.load(row + 17)
            vph, vps, vpb = tl.load(row + 18), tl.load(row + 19), tl.load(row + 20)
            vsh, vss, vsg = tl.load(row + 21), tl.load(row + 22), tl.load(row + 23)
            kvpb = 8 // kb
            vvpb = 8 // vb
            for offset in range(start, end, BN):
                n = offset + tl.arange(0, BN)
                valid = n < end
                codes = tl.load(kp + kvhead * kph + d[None, :] * kpd
                                + (n[:, None] // kvpb) * kpb,
                                mask=valid[:, None], other=0).to(tl.int32)
                codes = (codes >> ((n[:, None] % kvpb) * kb)) & ((1 << kb) - 1)
                ki = kvhead * ksh + d[None, :] * ksd + (n[:, None] // kgs) * ksg
                key = (codes.to(tl.float32) * tl.load(ks + ki, mask=valid[:, None], other=0).to(tl.float32)
                       + tl.load(km + ki, mask=valid[:, None], other=0).to(tl.float32))
                scores = tl.sum(key[None, :, :] * query[:, None, :], axis=2) * (HD ** -0.5)
                scores = tl.where(valid[None, :], scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, axis=1))
                correction = tl.exp(maximum - new_max)
                probs = tl.exp(scores - new_max[:, None])
                codes_v = tl.load(vp + kvhead * vph + n[:, None] * vps
                                  + (d[None, :] // vvpb) * vpb,
                                  mask=valid[:, None], other=0).to(tl.int32)
                codes_v = (codes_v >> ((d[None, :] % vvpb) * vb)) & ((1 << vb) - 1)
                vi = kvhead * vsh + n[:, None] * vss + (d[None, :] // vgs) * vsg
                value = (codes_v.to(tl.float32) * tl.load(vs + vi, mask=valid[:, None], other=0).to(tl.float32)
                         + tl.load(vm + vi, mask=valid[:, None], other=0).to(tl.float32))
                acc = acc * correction[:, None] + tl.sum(probs[:, :, None] * value[None, :, :], axis=1)
                denominator = denominator * correction + tl.sum(probs, axis=1)
                maximum = new_max
    else:
        for offset in range(0, HOT, BN):
            n = offset + tl.arange(0, BN)
            valid = n < HOT
            physical = (HOT_START + n) % HOT_CAP if HOT_CAP else n
            key = tl.load(HK + kvhead * KHH + physical[:, None] * KHS + d[None, :] * KHD,
                          mask=valid[:, None], other=0).to(tl.float32)
            value = tl.load(HV + kvhead * VHH + physical[:, None] * VHS + d[None, :] * VHD,
                            mask=valid[:, None], other=0).to(tl.float32)
            scores = tl.sum(key[None, :, :] * query[:, None, :], axis=2) * (HD ** -0.5)
            scores = tl.where(valid[None, :], scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.exp(maximum - new_max)
            probs = tl.exp(scores - new_max[:, None])
            acc = acc * correction[:, None] + tl.sum(probs[:, :, None] * value[None, :, :], axis=1)
            denominator = denominator * correction + tl.sum(probs, axis=1)
            maximum = new_max
    tl.store(PM + heads * stride + part, maximum, mask=live)
    tl.store(PL + heads * stride + part, denominator, mask=live)
    tl.store(PA + (heads[:, None] * stride + part) * HD + d[None, :], acc, mask=live[:, None])


@triton.jit
def _merge_segments(PM, PL, PA, OUT, NPARTIAL,
                    HD: tl.constexpr, BP: tl.constexpr, CAPACITY: tl.constexpr = 0):
    """Bound register use even with thousands of partials; no giant stack."""
    stride = CAPACITY if CAPACITY else NPARTIAL
    head = tl.program_id(0)
    d = tl.arange(0, HD)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    acc = tl.full((HD,), 0., tl.float32)
    for start in range(0, NPARTIAL, BP):
        p = start + tl.arange(0, BP)
        valid = p < NPARTIAL
        ms = tl.load(PM + head * stride + p, mask=valid, other=-float("inf"))
        ls = tl.load(PL + head * stride + p, mask=valid, other=0.)
        values = tl.load(PA + (head * stride + p[:, None]) * HD + d[None, :],
                         mask=valid[:, None], other=0.)
        new_max = tl.maximum(maximum, tl.max(ms, axis=0))
        # An empty hot tier has (-inf, 0, 0). It must contribute zero, not NaN.
        safe_max = tl.where(new_max == -float("inf"), 0., new_max)
        correction = tl.exp(maximum - safe_max)
        weights = tl.exp(ms - safe_max)
        denominator = denominator * correction + tl.sum(weights * ls, axis=0)
        acc = acc * correction + tl.sum(weights[:, None] * values, axis=0)
        maximum = new_max
    tl.store(OUT + head * HD + d, acc / denominator)


def segmented_phasekv_decode(q, tiers, num_attention_heads,
                             num_splits=None, block_n=None, num_warps=None):
    from ..optimizations import uses_optimized_attention
    if not uses_optimized_attention(tiers.optimizations):
        # Preserve the original two-launch CUDA implementation by default, and
        # isolate P1 quantization from changes to attention instruction layout.
        from .segmented_baseline import segmented_phasekv_decode as baseline_decode
        return baseline_decode(q, tiers, num_attention_heads, num_splits, block_n, num_warps)
    if q.device.type != "cuda" or q.dtype != torch.float16 or q.ndim != 2:
        raise ValueError("PhaseKV decode requires CUDA FP16 [query_heads, head_dim]")
    heads, hd = q.shape
    hk, hv = tiers.hot_k, tiers.hot_v
    if (heads != num_attention_heads or hk is None or hv is None
            or hk.ndim != 4 or hk.shape != hv.shape or hk.shape[0] != 1
            or hk.shape[-1] != hd or heads % hk.shape[1]
            or hk.dtype != q.dtype or hv.dtype != q.dtype
            or hk.device != q.device or hv.device != q.device
            or not 0 <= tiers.hot_len <= hk.shape[2]):
        raise ValueError("Incompatible query and hot KV layout")
    # Import lazily to avoid an import cycle with the legacy reference path.
    from .attention_triton import _device_sm_count
    sms = _device_sm_count(q.device)
    config = plan_kivi_decode(max(1, hk.shape[2]), heads, hd, sms,
                              num_splits, block_n, num_warps)
    # Stream identity prevents concurrent callers from sharing scratch. Caller
    # still owns normal PyTorch synchronization for KV produced on other streams.
    stream = torch.cuda.current_stream(q.device).cuda_stream
    key = (heads, hd, q.device, stream, num_splits, block_n, num_warps, tiers.optimizations)
    workspace = ensure_workspace(tiers, heads, hd, q.device, sms, key,
                                 num_splits, config.block_n, config.num_warps)
    count = workspace.partial_count
    out = torch.empty((heads, hd), device=q.device, dtype=q.dtype)
    specialized = tiers.optimizations.specialized_4bit and workspace.compatible_4bit
    gqa = tiers.optimizations.gqa_reuse and heads // hk.shape[1] > 1
    qper = triton.next_power_of_2(heads // hk.shape[1]) if gqa else 1
    _segmented_attention[(hk.shape[1] if gqa else heads, count)](
        q, workspace.descriptors, hk, hv, workspace.maxima, workspace.denominators,
        workspace.accumulators, count - 1, count, tiers.hot_len,
        *q.stride(), *hk.stride()[1:], *hv.stride()[1:],
        HD=hd, GROUP=heads // hk.shape[1], BN=config.block_n, DW=DESCRIPTOR_WIDTH, CAPACITY=workspace.capacity,
        PREFIX=workspace.prefix_count, COLD_GROUP=workspace.cold_group, NROWS=workspace.valid_count,
        FOUR_BIT=specialized, QPER=qper,
        HOT_CAP=hk.shape[2] if tiers.optimizations.hot_ring else 0, HOT_START=tiers.hot_start,
        num_warps=config.num_warps, enable_fp_fusion=False)
    _merge_segments[(heads,)](
        workspace.maxima, workspace.denominators, workspace.accumulators, out, count,
        HD=hd, BP=16, CAPACITY=workspace.capacity, num_warps=4, enable_fp_fusion=False)
    return out
