# PhaseKV

本项目原创实现集中于本目录，不伪装为另外下载的官方仓库。

- `phasekv_cache.py`、`kernels/`：真实低位打包缓存与注意力内核。
- `prompt_tier.py`：prompt token 选择。
- 对应模型实现已集中到 `models/Qwen2/`、`models/Llama/`、`models/OPT/`；本目录下 `qwen2/`、`llama/`、`opt/` 仅保留兼容入口。
- `model_runner.py`：多个模型共用的运行逻辑。
- `compressor.py`、`quant.py`、`hetero_cache.py`、`decode_tier.py`：历史评估路径，含伪量化；禁止用于要求真实量化的正式对比。正式对比继续由 `scripts/compare_longbench_kv.py` 阻断伪量化入口。

推荐入口：`python -m PhaseKV.qwen2`。旧 `src.phasekv`、`python -m qwen2`、`python -m llama` 和 `python -m opt` 保留转发兼容。
