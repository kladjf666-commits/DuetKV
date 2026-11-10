"""Single-query GQA attention reading KIVI's independently packed K and V.

Dequantization happens inside Triton tiles; no full FP16 KV is materialized.
"""
import torch
import triton
import triton.language as tl

from PhaseKV.kernels.attention_triton import _reduce_splits


@triton.jit
def _attention(Q, KP, KS, KM, VP, VS, VM, KR, VR, PM, PL, PA,
               NK, NV, TOTAL, KBYTES, KGROUPS, KRES, VRES, SPLIT_LEN,
               HD: tl.constexpr, HGRP: tl.constexpr, BITS: tl.constexpr,
               GS: tl.constexpr, SPLITS: tl.constexpr, BN: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    kvhead = head // HGRP
    dims = tl.arange(0, HD)
    q = tl.load(Q + head * HD + dims).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    z = tl.full((), 0., tl.float32)
    acc = tl.full((HD,), 0., tl.float32)
    VPB: tl.constexpr = 8 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    for start in range(split * SPLIT_LEN, tl.minimum((split + 1) * SPLIT_LEN, TOTAL), BN):
        pos = start + tl.arange(0, BN)
        valid = (pos < TOTAL) & (pos < (split + 1) * SPLIT_LEN)
        kmask = valid & (pos < NK)
        kb = tl.load(KP + (kvhead * HD + dims[None, :]) * KBYTES + pos[:, None] // VPB,
                     mask=kmask[:, None], other=0)
        kc = (kb >> ((pos[:, None] % VPB) * BITS)) & MASK
        ks = tl.load(KS + (kvhead * HD + dims[None, :]) * KGROUPS + pos[:, None] // GS,
                     mask=kmask[:, None], other=0)
        km = tl.load(KM + (kvhead * HD + dims[None, :]) * KGROUPS + pos[:, None] // GS,
                     mask=kmask[:, None], other=0)
        kquant = (kc.to(tl.float32) * ks.to(tl.float32) + km.to(tl.float32)).to(tl.float16).to(tl.float32)
        kres = tl.load(KR + (kvhead * KRES + pos[:, None] - NK) * HD + dims[None, :],
                       mask=(valid & (pos >= NK))[:, None], other=0).to(tl.float32)
        key = tl.where(kmask[:, None], kquant, kres)
        scores = tl.sum(key * q[None, :], axis=1) * (HD ** -0.5)
        scores = tl.where(valid, scores, -float("inf"))
        nm = tl.maximum(m, tl.max(scores, axis=0))
        correction = tl.exp(m - nm)
        probs = tl.exp(scores - nm)
        vmask = valid & (pos < NV)
        vb = tl.load(VP + (kvhead * NV + pos[:, None]) * (HD // VPB) + dims[None, :] // VPB,
                     mask=vmask[:, None], other=0)
        vc = (vb >> ((dims[None, :] % VPB) * BITS)) & MASK
        vs = tl.load(VS + (kvhead * NV + pos[:, None]) * (HD // GS) + dims[None, :] // GS,
                     mask=vmask[:, None], other=0)
        vm = tl.load(VM + (kvhead * NV + pos[:, None]) * (HD // GS) + dims[None, :] // GS,
                     mask=vmask[:, None], other=0)
        vquant = (vc.to(tl.float32) * vs.to(tl.float32) + vm.to(tl.float32)).to(tl.float16).to(tl.float32)
        vres = tl.load(VR + (kvhead * VRES + pos[:, None] - NV) * HD + dims[None, :],
                       mask=(valid & (pos >= NV))[:, None], other=0).to(tl.float32)
        value = tl.where(vmask[:, None], vquant, vres)
        acc = acc * correction + tl.sum(probs[:, None] * value, axis=0)
        z = z * correction + tl.sum(probs, axis=0)
        m = nm
    tl.store(PM + head * SPLITS + split, m)
    tl.store(PL + head * SPLITS + split, z)
    tl.store(PA + (head * SPLITS + split) * HD + dims, acc)


def kivi_decode(query, layer, *, bits, group_size):
    if query.device.type != "cuda" or query.dtype != torch.float16 or query.ndim != 2:
        raise ValueError("KIVI decode requires CUDA FP16 [query_heads, head_dim]")
    query = query.contiguous()
    hq, hd = query.shape
    hkv = layer.key_residual.shape[1]
    if hq % hkv or hd & (hd - 1) or hd % group_size:
        raise ValueError("Unsupported GQA or head dimension")
    nk = layer.key.orig_shape[2] if layer.key else 0
    nv = layer.value.orig_shape[2] if layer.value else 0
    kr, vr = layer.key_residual.contiguous(), layer.value_residual.contiguous()
    total = nk + kr.shape[2]
    if nv + vr.shape[2] != total:
        raise ValueError("K and V lengths differ")
    dummy = torch.empty(1, device=query.device, dtype=torch.uint8)
    kp, ks, km = ((layer.key.packed, layer.key.scale, layer.key.mins) if nk else (dummy,) * 3)
    vp, vs, vm = ((layer.value.packed, layer.value.scale, layer.value.mins) if nv else (dummy,) * 3)
    splits = min(8, triton.cdiv(total, 16))
    split_len = triton.cdiv(total, splits)
    pm = torch.empty((hq, splits), dtype=torch.float32, device=query.device)
    pl = torch.empty_like(pm)
    pa = torch.empty((hq, splits, hd), dtype=torch.float32, device=query.device)
    _attention[(hq, splits)](query, kp, ks, km, vp, vs, vm, kr, vr, pm, pl, pa,
                             nk, nv, total, triton.cdiv(nk, 8 // bits), triton.cdiv(nk, group_size),
                             kr.shape[2], vr.shape[2], split_len, HD=hd, HGRP=hq // hkv,
                             BITS=bits, GS=group_size, SPLITS=splits, BN=16,
                             num_warps=4, enable_fp_fusion=False)
    _, denominator, numerator = _reduce_splits(pm, pl, pa)
    return (numerator / denominator[:, None]).to(torch.float16)
