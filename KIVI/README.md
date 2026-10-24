# KIVI

- `packed_cache.py`：真实打包缓存，K/V 独立残留区更新。
- `kernels.py`：直接读取打包缓存的 Triton 解码内核。
- Qwen2 注意力适配位于 `models/Qwen2/kivi_attention.py`；`qwen2_attention.py` 仅为兼容转发。
- `legacy_fake_quant.py`：历史伪量化实现，只为旧接口保留，正式真实量化对比会禁止调用。
- `reference_snapshots/`：之前下载的官方实现参考文件，非完整仓库、未固定 Git 提交；来源信息与哈希见 `sources.json`。

参考仓库：https://github.com/jy-yuan/KIVI 。本地 Triton 实现不等同于官方 CUDA 扩展的性能复现。旧 `baselines.kivi*` 入口保留兼容。
