# SCOPE

- `official/`：作者官方 Git 仓库，https://github.com/Linking-ai/SCOPE 。当前固定提交 `8cbae48042e34da630c7a044f535de7a991c2999`，稀疏检出根目录、`model/`、`scripts/`；大型数据和结果未检出。
- `compressor.py`：本地 ALLKV + Slide 的 GQA 适配。旧入口 `baselines.scope` 兼容转发至此。
- `AUDIT.md`：实现核对结论及支持范围。

官方代码与本地实现分开维护。目录整理完成后，本地 ALLKV + Slide 的 Qwen2-1.5B 触发验证已通过，见 `../results/compression_triggers_qwen2_20261024.md`。其 GQA 处理与官方重复query-head缓存存在差异，不等于完整 SCOPE 复现。
