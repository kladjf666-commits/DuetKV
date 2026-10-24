# PyramidKV

- `compressor.py`：本地逐层预算与 prompt token 淘汰实现。
- `reference_snapshots/`：之前下载的官方参考源码，非完整仓库；见 `sources.json`。

参考仓库：https://github.com/Zefan-Cai/KVCache-Factory 。本地平均预算使用线性插值取整，与官方整数步长存在差异；不是逐行官方复现。该方法不做低位量化。旧 `baselines.pyramidkv` 入口兼容转发至本目录。
