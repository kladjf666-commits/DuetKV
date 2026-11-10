# Qwen2-7B-Instruct

- `profile.json`：官方模型ID、架构族、默认质量测试设置。
- `architecture.json`：核对过的模型结构；加载权重前必须匹配。
- `compression.json`：full、KIVI、SnapKV、SCOPE、PyramidKV、PhaseKV 的显式参数。
- `run.py`：统一入口，默认仅执行环境预检。

```bash
.venv/bin/python models/Qwen2-7B-Instruct/run.py --mode check --model-path /path/to/Qwen2-7B-Instruct
.venv/bin/python models/Qwen2-7B-Instruct/run.py --mode smoke --model-path /path/to/Qwen2-7B-Instruct --fixed-output --max-new-tokens 80
.venv/bin/python models/Qwen2-7B-Instruct/run.py --mode longbench --model-path /path/to/Qwen2-7B-Instruct --longbench-dir .cache/longbench
```

完整环境准备、算法范围、结果口径和扩展说明见 [模型接入总说明](../README.md)。7B代码已接入，完整权重推理和质量评估留待租卡执行。
