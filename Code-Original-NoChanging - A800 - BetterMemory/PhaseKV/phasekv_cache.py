"""PhaseKV 的真实打包缓存：只读 prompt、冻结 cold 段、可写 FP16 hot 区。

首次 prefill：外部选择 prompt 下标 → from_dense_prefill() 选择并打包。
单步 decode：每层 append_hot() → 读取 layer_tiers() 做注意力；模型前向
返回后，调用方统一执行 finish_decode_step()，按需冷却，再推进真实位置。
多轮 prompt：begin_prompt_chunk() → 每层逐 token 追加并做注意力 →
finish_prompt_chunk()。这是已有压缩缓存上的增量输入，不是首次 dense prefill。

K 沿 token 轴分组（per-channel），V 沿 head_dim 轴分组（per-token）。
本模块管理存储和生命周期；不计算 RoPE、注意力或采样，也不执行预算规划。
完整调用链及生成循环见 docs/delivery/USAGE.md 第 6 节。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import Optional, Sequence

import torch
from transformers.cache_utils import Cache

from .kernels.packing import PackedTensor, quantize_pack, quantize_pack_fast, quantize_pack_into, allocate_packed
from .optimizations import OptimizationConfig, resolve_config
from .packed_storage import PackedPagePool, storage_identity, unique_storage_bytes


_PACKED_BITS = (2, 4, 8)
_INDEX_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@dataclass
class PackedKVSegment:
    """单个模型层的一段已打包 K/V；length 是保留 token 数，不含量化补齐。"""

    key: PackedTensor
    value: PackedTensor

    @property
    def length(self) -> int:
        return self.key.orig_shape[2]

    @property
    def nbytes(self) -> int:
        """此逻辑段的有效 payload，不含页内空闲容量。"""
        return self.key.nbytes + self.value.nbytes


@dataclass
class LayerTiers:
    """一个模型层的缓存，不是一组模型层。

    prompt 是首轮 prompt；prompt_extensions 保存后续输入块；cold 保存
    已冷却的生成 token。三个字段按用途组织，不代表多轮对话的全局时间顺序。
    hot_k/hot_v 为 [B, Hkv, capacity, hd]；从 hot_start 起的 hot_len 个逻辑位置有效。
    """

    prompt: PackedKVSegment
    prompt_extensions: list[PackedKVSegment] = field(default_factory=list)
    cold: list[PackedKVSegment] = field(default_factory=list)
    hot_k: Optional[torch.Tensor] = None
    hot_v: Optional[torch.Tensor] = None
    hot_len: int = 0
    hot_start: int = 0
    optimizations: OptimizationConfig = field(default_factory=OptimizationConfig)
    cold_pool: object = field(default=None, repr=False, compare=False)
    # Derived GPU descriptors/scratch, not KV storage. Rebuilt only when frozen
    # segments change; hot pointers and hot_len are passed at launch time.
    _decode_workspace: object = field(default=None, repr=False, compare=False)

    @property
    def cold_len(self) -> int:
        return sum(segment.length for segment in self.cold)

    @property
    def prompt_len(self) -> int:
        return self.prompt.length + sum(segment.length for segment in self.prompt_extensions)

    @property
    def physical_len(self) -> int:
        """当前保留的 token 数，不含热层预留空间和量化补齐。"""
        return self.prompt_len + self.cold_len + self.hot_len

    def hot_range(self, start=0, length=None):
        """Ordered reference/debug view; wrapping ranges allocate a temporary.

        Production CUDA decode reads hot_k/hot_v plus hot_start/hot_len directly.
        This helper is also used for compatibility selection and prompt resizing.
        """
        if self.hot_k is None or self.hot_v is None:
            raise RuntimeError("Hot buffers have not been allocated")
        length = self.hot_len - start if length is None else length
        if not 0 <= start <= self.hot_len or not 0 <= length <= self.hot_len - start:
            raise ValueError("Hot range exceeds valid logical tokens")
        capacity = self.hot_k.shape[2]
        physical = (self.hot_start + start) % capacity
        def ordered(tensor):
            if physical + length <= capacity:
                return tensor[:, :, physical:physical + length, :]
            first = tensor[:, :, physical:, :]
            second = tensor[:, :, :length - (capacity - physical), :]
            return torch.cat((first, second), dim=2)
        return ordered(self.hot_k), ordered(self.hot_v)

    def hot_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ordered logical hot KV; may copy on wrap. Never used by fast decode."""
        return self.hot_range()


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} 必须为正整数，实际为 {value!r}")


def _validate_config(prompt_bits, cold_bits, group_size, recent_window,
                     lazy_g, decode_keep_ratio) -> None:
    """构造入口共用校验；真实打包支持的位宽与 fake-quant 路径不同。"""
    for name, bits in (("prompt_bits", prompt_bits), ("cold_bits", cold_bits)):
        if isinstance(bits, bool) or not isinstance(bits, Integral) or bits not in _PACKED_BITS:
            raise ValueError(f"{name} 的真实打包位宽必须为 2、4 或 8，实际为 {bits!r}")
    for name, value in (("group_size", group_size), ("recent_window", recent_window),
                        ("lazy_g", lazy_g)):
        _require_positive_int(name, value)
    if (isinstance(decode_keep_ratio, bool) or not isinstance(decode_keep_ratio, Real)
            or not 0.0 < decode_keep_ratio <= 1.0):
        raise ValueError(f"decode_keep_ratio 必须落在 (0, 1]，实际为 {decode_keep_ratio!r}")


def _pack_kv(key: torch.Tensor, value: torch.Tensor, bits: int,
             group_size: int, *, fast=False, pool=None,
             token_start=0, token_count=None, channel_contiguous_k=False) -> PackedKVSegment:
    """Keep K/token and V/channel grouping while optionally writing a page slot."""
    if pool is None and token_count is None and not (channel_contiguous_k and bits == 4):
        pack = quantize_pack_fast if fast else quantize_pack
        return PackedKVSegment(pack(key, bits, axis=2, group_size=group_size),
                               pack(value, bits, axis=3, group_size=group_size))
    shape = list(key.shape)
    if token_count is not None:
        shape[2] = token_count
    if pool is not None:
        outputs = pool.reserve(shape, bits, group_size, key.device, channel_contiguous_k=channel_contiguous_k)
    else:
        outputs = tuple(allocate_packed(shape, bits, axis, group_size, key.device,
                                        channel_contiguous_k=channel_contiguous_k) for axis in (2, 3))
    for source, out in zip((key, value), outputs):
        quantize_pack_into(source, out, fast=fast, token_start=token_start, token_count=token_count)
    return PackedKVSegment(*outputs)


def _gather_kv(key: torch.Tensor, value: torch.Tensor,
               indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按各 batch、各 KV head 的 [B,Hkv,K] 下标保留 token。"""
    gather_index = indices.unsqueeze(-1).expand(*indices.shape, key.shape[-1])
    return key.gather(2, gather_index), value.gather(2, gather_index)




#一个 PhaseKVCache                  ← 整条对话的缓存
#新增prompt会在prefill时会暂时借用hot_k和hot_v
#│
#├─ layers[0]：第 0 个模型层的 LayerTiers
#│  ├─ prompt                       首轮 prompt 的压缩 KV
#│  ├─ prompt_extensions            后续新增 prompt 的压缩 KV
#│  ├─ cold                         已压缩的生成 token KV
#│  └─ hot_k / hot_v                 尚未压缩的生成 token KV
#│
#├─ layers[1]：第 1 个模型层的 LayerTiers
#│  ├─ prompt
#│  ├─ prompt_extensions
#│  ├─ cold
#│  └─ hot_k / hot_v
#│
#└─ ……其余模型层

class PhaseKVCache(Cache):
    """供自定义 attention 使用的低比特缓存，通常由 from_dense_prefill 创建。

    get_seq_length() 返回某层已保留的物理 token 数；true_seq_length 返回
    已完成前向并提交的真实 token 数。下次前向用后者生成 position_ids，
    用前者生成 cache_position。逐层写入期间，真实长度直到 finish_* 才推进。

    标准 Cache.update() 返回 dense K/V，本类不提供该接口语义。
    当前 Qwen2/Llama/OPT 适配器要求 CUDA、batch=1、无 padding，
    需通过各模型入口显式驱动生成循环。
    """

    def __init__(
        self,
        layers: list[LayerTiers],
        *,
        seen_tokens: int,
        prompt_bits: int,
        cold_bits: int,
        group_size: int,
        recent_window: int,
        lazy_g: int,
        decode_keep_ratio: float,
        optimizations=None,
    ) -> None:
        super().__init__()
        _validate_config(prompt_bits, cold_bits, group_size, recent_window,
                         lazy_g, decode_keep_ratio)
        _require_positive_int("seen_tokens", seen_tokens)
        if not layers:
            raise ValueError("layers 不能为空：PhaseKVCache 至少需要一层 KV")
        for layer_idx, layer in enumerate(layers):
            if layer.hot_k is None or layer.hot_v is None:
                raise ValueError(f"第 {layer_idx} 层未分配 hot_k/hot_v；请使用 from_dense_prefill() 创建缓存")
        self.layers = layers
        self.optimizations = resolve_config(optimizations)
        for layer in layers:
            layer.optimizations = self.optimizations
            if self.optimizations.paged_cold:
                layer.cold_pool = PackedPagePool(self.optimizations.page_slots)
        self._seen_tokens = int(seen_tokens)
        self.prompt_bits = prompt_bits
        self.cold_bits = cold_bits
        self.group_size = group_size
        self.recent_window = recent_window
        self.lazy_g = lazy_g
        self.decode_keep_ratio = decode_keep_ratio
        # One page slot can hold a cooling batch across all homogeneous layers.
        self._batched_cold_pool = None
        # 当前 decode 步中已经写入一个 token 的模型层。
        self._pending_layers: set[int] = set()   #当前这一次 decode 前向中，哪些模型层已经写入了新 token 的 K/V，但还没有统一结束这一步。
        # 0 表示没有新增 prompt 块；start 记录每层原有生成热 token 的数量。
        self._prompt_chunk_len = 0
        self._prompt_chunk_start: list[int] = []

    # 首轮 prefill：接收完整、尚未压缩的 KV，建立只读 prompt 层。
    @classmethod
    @torch.no_grad()
    def from_dense_prefill(
        cls,
        dense_cache,
        *,
        prompt_len: int,
        keep_indices: Optional[Sequence[torch.Tensor]] = None,
        prompt_bits: int = 2,
        cold_bits: int = 4,
        group_size: int = 32,
        recent_window: int = 32,
        lazy_g: int = 16,
        decode_keep_ratio: float = 1.0,
        optimizations=None,
        consume_dense_cache: bool = False,
    ) -> "PhaseKVCache":
        """选择原始 prefill K/V 并真实打包，默认不修改 dense_cache。

        dense_cache 需提供 key_cache/value_cache（如 Transformers 4.46 的
        DynamicCache），K 已施加真实位置的 RoPE。keep_indices[layer] 是
        [B,Hkv,K0] 的整数原始位置，各 head 可不同，但各层 K0 须相同；
        None 表示全部保留。注意力分数选择由调用方 select_prompt_indices 完成。
        构建后调用方应释放 dense_cache 与 prefill 输出中不再需要的引用。
        consume_dense_cache=True 显式移交所有权：每层打包后将源列表项置为
        None，降低新旧缓存同时驻留的峰值。源缓存随后不可再用；异常也可能
        留下部分已消费的缓存。调用者不得保留原 K/V 的额外引用。
        """
        optimizations = resolve_config(optimizations)
        _require_positive_int("prompt_len", prompt_len)
        _validate_config(prompt_bits, cold_bits, group_size, recent_window,
                         lazy_g, decode_keep_ratio)
        if not hasattr(dense_cache, "key_cache") or not hasattr(dense_cache, "value_cache"):
            raise TypeError("dense_cache 必须提供逐层 key_cache/value_cache，例如 DynamicCache")
        keys, values = dense_cache.key_cache, dense_cache.value_cache
        if consume_dense_cache and (not isinstance(keys, list) or not isinstance(values, list)):
            raise TypeError("consume_dense_cache requires mutable key_cache/value_cache lists")
        if len(keys) == 0 or len(keys) != len(values):
            raise ValueError(f"prefill 缓存需要非空且层数相同的 K/V，实际为 {len(keys)}/{len(values)} 层")
        if keep_indices is not None and len(keep_indices) != len(keys):
            raise ValueError(f"keep_indices 需要 {len(keys)} 层下标，实际为 {len(keep_indices)} 层")

        # 最多容纳 recent_window 个受保护 token + lazy_g 个待冷却 token。
        capacity = recent_window + lazy_g
        layers: list[LayerTiers] = []
        retained_length: Optional[int] = None
        for layer_idx in range(len(keys)):
            key, value = keys[layer_idx], values[layer_idx]
            context = f"第 {layer_idx} 层 prefill"
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError(f"{context} K/V 必须是 Tensor")
            if key.ndim != 4 or key.shape != value.shape or any(size == 0 for size in key.shape):
                raise ValueError(f"{context} K/V 必须为同形状的非空 [B,Hkv,S,hd]，实际为 {tuple(key.shape)}/{tuple(value.shape)}")
            if key.dtype not in (torch.float16, torch.bfloat16) or value.dtype != key.dtype:
                raise ValueError(f"{context} K/V 必须同为 float16 或 bfloat16，实际为 {key.dtype}/{value.dtype}")
            if key.device != value.device:
                raise ValueError(f"{context} K/V 必须在同一设备，实际为 {key.device}/{value.device}")
            if key.shape[2] != prompt_len:
                raise ValueError(f"{context} KV 长度应为 prompt_len={prompt_len}，实际为 {key.shape[2]}")
            if keep_indices is not None:
                indices = keep_indices[layer_idx]
                if not isinstance(indices, torch.Tensor) or indices.dtype not in _INDEX_DTYPES:
                    raise ValueError(f"第 {layer_idx} 层 keep_indices 必须是整数张量，不能将浮点位置截断为整数")
                if indices.ndim != 3 or indices.shape[:2] != key.shape[:2] or indices.shape[2] == 0:
                    raise ValueError(f"第 {layer_idx} 层 keep_indices 应为 [{key.shape[0]},{key.shape[1]},K0] 且 K0>0，实际为 {tuple(indices.shape)}")
                indices = indices.to(device=key.device, dtype=torch.long).sort(dim=-1).values
                if bool((indices < 0).any()) or bool((indices >= prompt_len).any()):
                    raise ValueError(f"第 {layer_idx} 层 keep_indices 必须落在 [0, {prompt_len})")
                if bool((indices[..., 1:] == indices[..., :-1]).any()):
                    raise ValueError(f"第 {layer_idx} 层 keep_indices 在同一 batch/head 内含重复位置")
                key, value = _gather_kv(key, value, indices)
            if retained_length is not None and key.shape[2] != retained_length:
                raise ValueError(f"各层必须保留相同数量的 prompt token：第 0 层为 {retained_length}，第 {layer_idx} 层为 {key.shape[2]}")
            retained_length = key.shape[2]
            # 模型 prefill 可以是 BF16；当前 packed attention 和热层使用 FP16。
            key, value = key.to(torch.float16), value.to(torch.float16)
            shape = (key.shape[0], key.shape[1], capacity, key.shape[3])
            #预分配hot容量
            layers.append(LayerTiers(
                prompt=_pack_kv(key, value, prompt_bits, group_size, fast=optimizations.fused_quantize,
                                channel_contiguous_k=optimizations.channel_contiguous_k),
                hot_k=torch.empty(shape, dtype=torch.float16, device=key.device),
                hot_v=torch.empty(shape, dtype=torch.float16, device=key.device),
            ))
            if consume_dense_cache:
                keys[layer_idx] = None
                values[layer_idx] = None
                # Do not carry gathered FP16 storage into the next packing.
                del key, value
        return cls(
            layers, seen_tokens=prompt_len, prompt_bits=prompt_bits,
            cold_bits=cold_bits, group_size=group_size,
            recent_window=recent_window, lazy_g=lazy_g,
            decode_keep_ratio=decode_keep_ratio,
            optimizations=optimizations,
        )

    # 读取接口：真实位置、物理长度、热区容量是三个不同概念。
    def __len__(self) -> int:
        return len(self.layers)

    def layer_tiers(self, layer_idx: int) -> LayerTiers:
        if (isinstance(layer_idx, bool) or not isinstance(layer_idx, Integral)
                or not 0 <= layer_idx < len(self.layers)):
            raise IndexError(f"layer_idx 必须是 [0, {len(self.layers)}) 内的整数，实际为 {layer_idx!r}")
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.layer_tiers(0 if layer_idx is None else layer_idx).physical_len

    def get_max_cache_shape(self) -> None:
        """总缓存可随冻结段增长；热区容量不是总缓存的最大长度。"""
        return None

    @property
    def true_seq_length(self) -> int:
        """已提交的真实长度；用于下一次前向的绝对 RoPE 起点。"""
        return self._seen_tokens

    @property
    def prompt_chunk_len(self) -> int:
        return self._prompt_chunk_len

    # 单步 decode：层内先写后读，模型前向结束后统一维护。
    @torch.no_grad()
    def append_hot(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """每层每步写入一个已施加 RoPE 的 K 和对应 V；此处不冷却、不推进真实长度。"""
        if self._prompt_chunk_len:
            raise RuntimeError("新增 prompt 块尚未结束：块内请用 append_prompt_token()，完成后调用 finish_prompt_chunk()")
        layer = self.layer_tiers(layer_idx)
        if layer_idx in self._pending_layers:
            raise RuntimeError(f"第 {layer_idx} 层在当前 decode 步已追加；每层每步只能写一次，模型前向结束后调用 finish_decode_step()")
        self._append_token(layer_idx, layer, key, value)
        self._pending_layers.add(layer_idx)

    @torch.no_grad()
    def finish_decode_step(self) -> None:
        """每次单 token 模型前向返回后调用一次；冷却后真实长度加 1。

        能检查各层是否写入，不能检查 GPU 注意力是否已执行；调用方负责顺序。
        """
        if self._prompt_chunk_len:
            raise RuntimeError("新增 prompt 块尚未结束：应调用 finish_prompt_chunk()，不能调用 finish_decode_step()")
        missing = sorted(set(range(len(self.layers))) - self._pending_layers)
        if missing:
            raise RuntimeError(f"finish_decode_step() 需要每一层各追加一个 token；尚未追加的层为 {missing}")
        # 先检查跨层边界，避免发现错误前已经冷却部分层。
        lengths = [(layer.physical_len, layer.hot_len) for layer in self.layers]
        if len(set(lengths)) != 1:
            raise RuntimeError(f"冷却前各层的 (物理长度, 热层长度) 不一致：{lengths}")
        aged = self.layers[0].hot_len - self.recent_window
        if aged >= self.lazy_g and not self._cool_batch(aged):
            for layer in self.layers:
                self._cool_aged_tokens(layer)
        self._seen_tokens += 1
        self._pending_layers.clear()

    def _cool_batch(self, aged: int) -> bool:
        """Pack the 16-to-13 CUDA case across layers, preserving per-head selection."""
        opt = self.optimizations
        if not (len(self.layers) > 1 and opt.hot_ring and opt.paged_cold and opt.fused_quantize
                and self.cold_bits == 4 and self.group_size == 32 and aged == 16
                and round(self.decode_keep_ratio * aged) == 13):
            return False
        first = self.layers[0].hot_k
        if first.device.type != "cuda" or first.shape[0] != 1 or first.shape[-1] != 128:
            return False
        if any(t.shape != first.shape or t.device != first.device or t.dtype != torch.float16
               for layer in self.layers for t in (layer.hot_k, layer.hot_v)):
            return False

        # Each leading row is one layer. Norm, topk and sort still operate on
        # the same dimensions as in the single-layer path (including tie handling).
        key_ranges, value_ranges = zip(*(layer.hot_range(0, aged) for layer in self.layers))
        keys = torch.cat(key_ranges, dim=0)
        values = torch.cat(value_ranges, dim=0)
        scores = torch.linalg.vector_norm(values.float(), dim=-1)
        indices = scores.topk(13, dim=-1).indices.sort(dim=-1).values
        keys, values = _gather_kv(keys, values, indices)
        if self._batched_cold_pool is None:
            self._batched_cold_pool = PackedPagePool(opt.page_slots)
        batch = _pack_kv(keys, values, self.cold_bits, self.group_size, fast=True,
                         pool=self._batched_cold_pool, channel_contiguous_k=opt.channel_contiguous_k)

        def layer_views(packed):
            # The batch axis here represents layers. Each segment must expose
            # batch=1 while retaining its view of the shared page.
            shape = (1, *packed.orig_shape[1:])
            return [replace(packed, packed=codes, scale=scale, mins=mins, orig_shape=shape)
                    for codes, scale, mins in zip(packed.packed.split(1), packed.scale.split(1),
                                                  packed.mins.split(1))]

        segments = [PackedKVSegment(k, v) for k, v in zip(layer_views(batch.key), layer_views(batch.value))]
        for layer, segment in zip(self.layers, segments):
            self._commit_cooling(layer, segment, aged)
        return True

    def _cool_aged_tokens(self, layer: LayerTiers) -> None:
        """仅处理热区最旧的一批；已冻结 prompt/cold 永不重新量化。"""
        aged = layer.hot_len - self.recent_window
        if aged < self.lazy_g:#要超过recent_window+lazy_g才会量化一批
            return
        keep_count = max(1, round(self.decode_keep_ratio * aged))
        token_start, token_count = 0, None
        if keep_count < aged:
            # Selection requires logical token order. Only the cooling batch is
            # materialized; the retained hot window is never moved in ring mode.
            key, value = layer.hot_range(0, aged)
            importance = torch.linalg.vector_norm(value.float(), dim=-1)
            indices = importance.topk(keep_count, dim=-1).indices.sort(dim=-1).values
            key, value = _gather_kv(key, value, indices)
        elif self.optimizations.hot_ring:
            key, value = layer.hot_k, layer.hot_v
            token_start, token_count = layer.hot_start, aged
        else:
            key, value = layer.hot_range(0, aged)
        segment = _pack_kv(key, value, self.cold_bits, self.group_size,
                           fast=self.optimizations.fused_quantize, pool=layer.cold_pool,
                           channel_contiguous_k=self.optimizations.channel_contiguous_k,
                           token_start=token_start, token_count=token_count)
        self._commit_cooling(layer, segment, aged)

    def _commit_cooling(self, layer, segment, aged):
        layer.cold.append(segment)
        if not self.optimizations.reuse_workspace:
            layer._decode_workspace = None
        remaining = layer.hot_len - aged
        if self.optimizations.hot_ring:
            layer.hot_start = (layer.hot_start + aged) % layer.hot_k.shape[2]
        else:
            layer.hot_k[:, :, :remaining, :].copy_(layer.hot_k[:, :, aged:layer.hot_len, :].clone())
            layer.hot_v[:, :, :remaining, :].copy_(layer.hot_v[:, :, aged:layer.hot_len, :].clone())
        layer.hot_len = remaining

    # 多轮输入：临时借用热区，块内逐 token 做因果注意力，块末冻结。
    @torch.no_grad()
    def begin_prompt_chunk(self, length: int) -> None:
        """为新增输入块预留空间；length 也可以为 1，调用前必须处于步间状态。"""
        _require_positive_int("length", length)
        self._require_idle("begin_prompt_chunk()")
        for layer in self.layers:
            needed = layer.hot_len + length
            if needed <= layer.hot_k.shape[2]:
                continue
            new_k, new_v = self._empty_hot(layer, needed)
            hot_k, hot_v = layer.hot_views()
            new_k[:, :, :layer.hot_len, :].copy_(hot_k)
            new_v[:, :, :layer.hot_len, :].copy_(hot_v)
            layer.hot_k, layer.hot_v = new_k, new_v
            layer.hot_start = 0
        self._prompt_chunk_start = [layer.hot_len for layer in self.layers]  #记录prefill前，已有的token数（就是hot_len）
        self._prompt_chunk_len = length

    @torch.no_grad()
    def append_prompt_token(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """每层块内逐 token 写入，写一个就计算该 query 的注意力，避免看见未来 token。"""
        if not self._prompt_chunk_len:
            raise RuntimeError("没有进行中的 prompt 块；请先调用 begin_prompt_chunk(length)")
        layer = self.layer_tiers(layer_idx)
        written = layer.hot_len - self._prompt_chunk_start[layer_idx]   #每写入一个token，hot_len都会+1,_prompt_chunk_start记录了初始时刻hot里面的token数量
        if written >= self._prompt_chunk_len:
            raise RuntimeError(f"第 {layer_idx} 层已写满本次 prompt 块的 {self._prompt_chunk_len} 个 token，不能继续追加")
        self._append_token(layer_idx, layer, key, value)

    @torch.no_grad()
    def finish_prompt_chunk(self) -> None:
        """块前向返回后调用：保留全部新增 token，按 prompt_bits 打包并冻结。

        当前多轮策略不做 prompt 选择。块开始前的生成热 token 原样保留为
        FP16，不因新增 prompt 而冷却或驱逐；其冷却仍由 finish_decode_step()
        按 recent_window 和 lazy_g 决定。旧冻结段保持不变。
        """
        if not self._prompt_chunk_len:
            raise RuntimeError("没有进行中的 prompt 块；请先调用 begin_prompt_chunk(length)")
        for layer_idx, (layer, start) in enumerate(zip(self.layers, self._prompt_chunk_start)):
            written = layer.hot_len - start
            if written != self._prompt_chunk_len:
                raise RuntimeError(f"第 {layer_idx} 层新增 prompt 尚未写完：期望 {self._prompt_chunk_len} 个，实际 {written} 个")
        for layer, start in zip(self.layers, self._prompt_chunk_start):
            #start:layer.hot_len保留的才是本论的prompt。0:start是上一轮留在热区的生成token
            prompt_segment = _pack_kv(
                layer.hot_k, layer.hot_v, self.prompt_bits, self.group_size,
                fast=self.optimizations.fused_quantize,
                channel_contiguous_k=self.optimizations.channel_contiguous_k,
                token_start=(layer.hot_start + start) % layer.hot_k.shape[2],
                token_count=self._prompt_chunk_len)
            # Restore ordinary capacity, retaining every pre-prompt generated
            # token (including an incomplete cooling batch) in logical order.
            old_k, old_v = layer.hot_range(0, start)
            hot_k, hot_v = self._empty_hot(layer, self.recent_window + self.lazy_g)
            hot_k[:, :, :start, :].copy_(old_k)
            hot_v[:, :, :start, :].copy_(old_v)
            layer.prompt_extensions.append(prompt_segment)
            layer._decode_workspace = None
            layer.hot_k, layer.hot_v = hot_k, hot_v
            layer.hot_len = start
            layer.hot_start = 0
        self._seen_tokens += self._prompt_chunk_len
        self._prompt_chunk_len = 0
        self._prompt_chunk_start = []

    # 写入共用逻辑：校验失败时不修改 hot_len，也不把该层标记为已写入。
    # 写入一个token
    def _append_token(self, layer_idx: int, layer: LayerTiers,
                      key: torch.Tensor, value: torch.Tensor) -> None:
        context = f"第 {layer_idx} 层新增 K/V"
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"{context} 必须是 Tensor")
        expected = (*layer.hot_k.shape[:2], 1, layer.hot_k.shape[-1])
        if key.shape != expected or value.shape != expected:
            raise ValueError(f"{context} 必须是单 token，期望形状 {expected}，实际为 {tuple(key.shape)}/{tuple(value.shape)}")
        if key.dtype != torch.float16 or value.dtype != torch.float16:
            raise ValueError(f"{context} 必须为 float16，实际为 {key.dtype}/{value.dtype}；模型可用 BF16，但适配器须在写入前转换")
        if key.device != layer.hot_k.device or value.device != layer.hot_v.device:
            raise ValueError(f"{context} 必须与热层在同一设备 {layer.hot_k.device}，实际为 {key.device}/{value.device}")
        if layer.hot_len >= layer.hot_k.shape[2]:
            raise RuntimeError(f"第 {layer_idx} 层热层容量不足：hot_len={layer.hot_len}，capacity={layer.hot_k.shape[2]}")
        position = (layer.hot_start + layer.hot_len) % layer.hot_k.shape[2]
        layer.hot_k[:, :, position:position + 1, :].copy_(key)
        layer.hot_v[:, :, position:position + 1, :].copy_(value)
        layer.hot_len += 1

    @staticmethod
    def _empty_hot(layer: LayerTiers, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*layer.hot_k.shape[:2], capacity, layer.hot_k.shape[-1])
        return layer.hot_k.new_empty(shape), layer.hot_v.new_empty(shape)

    def _require_idle(self, operation: str) -> None:
        if self._prompt_chunk_len:
            raise RuntimeError(f"{operation} 不能在新增 prompt 块内调用；请先完成块前向并调用 finish_prompt_chunk()")
        if self._pending_layers:
            raise RuntimeError(f"{operation} 不能在 decode 步内调用；已写入层为 {sorted(self._pending_layers)}，请先完成所有层并调用 finish_decode_step()")

    # 统计及 Cache 兼容接口。
    def physical_nbytes(self) -> int:
        """当前持有的 KV 张量字节数，含热层预留空间；不含临时量化/注意力工作区。"""
        return unique_storage_bytes(t for _, t in self.storage_tensors())

    def storage_tensors(self):
        """Enumerate KV owners, including full pages; consumers must deduplicate."""
        if self._batched_cold_pool is not None:
            yield from self._batched_cold_pool.storage_tensors()
        for layer in self.layers:
            for segment in (layer.prompt, *layer.prompt_extensions, *layer.cold):
                for packed in (segment.key, segment.value):
                    yield "codes", packed.packed
                    yield "metadata", packed.scale
                    yield "metadata", packed.mins
            if layer.cold_pool is not None:
                yield from layer.cold_pool.storage_tensors()
            yield "hot", layer.hot_k
            yield "hot", layer.hot_v

    def storage_breakdown(self):
        capacities = {"codes": 0, "metadata": 0, "hot": 0}
        seen = set()
        for kind, tensor in self.storage_tensors():
            identity = storage_identity(tensor)
            if identity not in seen:
                capacities[kind] += tensor.untyped_storage().nbytes()
                seen.add(identity)
        code_payload = metadata_payload = hot_payload = auxiliary = 0
        for layer in self.layers:
            for segment in (layer.prompt, *layer.prompt_extensions, *layer.cold):
                for packed in (segment.key, segment.value):
                    code_payload += packed.packed.numel()
                    metadata_payload += (packed.scale.numel() + packed.mins.numel()) * 2
            hot_payload += layer.hot_len * layer.hot_k.shape[0] * layer.hot_k.shape[1] * layer.hot_k.shape[3] * 4
            if layer._decode_workspace is not None:
                auxiliary += layer._decode_workspace.nbytes
        total = sum(capacities.values())
        return {"code_bytes": capacities["codes"], "metadata_bytes": capacities["metadata"],
                "fp16_residual_capacity_bytes": capacities["hot"], "physical_bytes": total,
                "code_payload_bytes": code_payload, "metadata_payload_bytes": metadata_payload,
                "hot_payload_bytes": hot_payload,
                "effective_payload_bytes": code_payload + metadata_payload + hot_payload,
                "page_free_bytes": capacities["codes"] + capacities["metadata"] - code_payload - metadata_payload,
                "hot_free_bytes": capacities["hot"] - hot_payload,
                "auxiliary_bytes": auxiliary, "total_owned_bytes": total + auxiliary}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        raise RuntimeError(
            "PhaseKVCache 不返回稠密 K/V，不能接普通 attention 的 Cache.update()；"
            "请安装 PhaseKV attention 适配器，由适配器调用 append_hot()（decode）或 "
            "append_prompt_token()（新增 prompt），再通过 layer_tiers() 读取分层缓存"
        )

    @torch.no_grad()
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """在步间重排 batch 维；此存储操作不代表 Qwen2 适配器已支持 beam search。"""
        self._require_idle("reorder_cache()")
        if (not isinstance(beam_idx, torch.Tensor) or beam_idx.ndim != 1
                or beam_idx.dtype != torch.long or beam_idx.numel() == 0):
            raise ValueError("beam_idx 必须是非空的一维 torch.long 张量")
        for layer_idx, layer in enumerate(self.layers):
            if bool((beam_idx < 0).any()) or bool((beam_idx >= layer.hot_k.shape[0]).any()):
                raise ValueError(f"beam_idx 超出第 {layer_idx} 层的 batch 范围 [0, {layer.hot_k.shape[0]})")

        def reorder_packed(packed: PackedTensor) -> PackedTensor:
            index = beam_idx.to(packed.packed.device)
            return replace(
                packed,
                packed=packed.packed.index_select(0, index),
                scale=packed.scale.index_select(0, index),
                mins=packed.mins.index_select(0, index),
                orig_shape=(index.numel(), *packed.orig_shape[1:]),
            )

        for layer in self.layers:
            layer._decode_workspace = None
            if layer.cold_pool is not None:
                layer.cold_pool = PackedPagePool(self.optimizations.page_slots)
            layer.prompt = PackedKVSegment(
                reorder_packed(layer.prompt.key), reorder_packed(layer.prompt.value))
            layer.prompt_extensions = [PackedKVSegment(
                reorder_packed(segment.key), reorder_packed(segment.value))
                for segment in layer.prompt_extensions]
            layer.cold = [PackedKVSegment(
                reorder_packed(segment.key), reorder_packed(segment.value))
                for segment in layer.cold]
            layer.hot_k = layer.hot_k.index_select(0, beam_idx.to(layer.hot_k.device))
            layer.hot_v = layer.hot_v.index_select(0, beam_idx.to(layer.hot_v.device))
        self._batched_cold_pool = None
