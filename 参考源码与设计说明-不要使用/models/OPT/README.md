# OPT PhaseKV 入口

本目录适配 Transformers 4.46.3 的 OPT 架构，默认模型是 `facebook/opt-6.7b`。
OPT 使用旧式逐层 tuple 缓存和学习式绝对位置编码；适配器在首次原生 prefill 后
把旧缓存转换为 `PhaseKVCache`，并在后续生成时使用真实绝对位置，避免驱逐后
将物理缓存长度误当作位置。只支持单卡 CUDA、batch=1、无 padding、
`attn_implementation="eager"`；不接管 `model.generate()`。

在项目根目录运行：

```bash
python -m opt --check-cuda
python -m opt --model /path/to/opt-6.7b \
  --local-files-only --prompt "Explain KV caching briefly." --max-new-tokens 64
python opt/opt.py --model /path/to/opt-6.7b --local-files-only
```

短输入默认完整保留 prompt，并使用 2-bit 打包。长上下文选择实验可加
`--prefill-keep-ratio 0.3`。OPT 是基础语言模型，入口直接编码原始 prompt，
不套聊天模板。模型权重放在外部缓存或数据盘。

`opt_attention.py` 提供 `install_opt_phasekv_attention(model)`，同时安装注意力层
和 OPT decoder 桥接；`opt.py` 是可运行的生成入口。
