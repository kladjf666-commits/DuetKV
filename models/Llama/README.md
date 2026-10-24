# Llama 模型适配

本目录共用适配 [Llama-2-7b-chat-hf](../Llama-2-7b-chat-hf/README.md) 与 [Llama-3.1-8B-Instruct](../Llama-3.1-8B-Instruct/README.md)。六方法比较入口在各具体模型目录的 `run.py`，支持full、KIVI、SnapKV、SCOPE、PyramidKV和PhaseKV。

| 文件 | 用途 |
|---|---|
| `integration.py` | 六方法调度、MHA/GQA入口约束、原生聊天模板 |
| `kivi_attention.py` | 原生Llama投影/RoPE与真实打包KIVI decode |
| `llama_attention.py` | 原生Llama投影/RoPE与PhaseKV decode |
| `packed_runtime.py` | 注册两种注意力包装，调用共用打包循环 |
| `llama.py`、`__main__.py` | 保留历史PhaseKV单模型交互入口 |

共用循环在 `models/packed_runtime.py`，通用缓存/内核保留在对应算法目录。Llama适配不导入Qwen模型代码。限制为Transformers 4.46.3、eager、CUDA单卡、batch=1、无padding、`pretraining_tp=1`。Llama-3.1的RoPE使用Transformers原生实现，结束标记取模型generation config的完整列表。

本轮仅代码与CPU检查完成，完整权重GPU推理和质量/性能评估待租卡执行，详见[统一部署说明](../README.md)。

## 历史PhaseKV入口

同一 `llama/` 适配器面向 Transformers 4.46.3 的 Llama 架构，默认模型是
`meta-llama/Llama-3.1-8B-Instruct`；Llama-2-7B 可通过 `--model` 指定。
首次 prefill 使用原始 Llama 注意力，随后将 KV 打包到 `PhaseKVCache`，
逐 token 生成使用融合 Triton 注意力。只支持单卡 CUDA、batch=1、无 padding、
`pretraining_tp=1`；不接管 `model.generate()`，由本目录入口显式推进缓存。

在项目根目录运行：

```bash
python -m llama --check-cuda
python -m llama --model /path/to/Llama-3.1-8B-Instruct \
  --local-files-only --prompt "Explain KV caching briefly." --max-new-tokens 64
python llama/llama.py --model /path/to/Llama-2-7B \
  --local-files-only --prompt "Explain KV caching briefly."
```

短输入默认完整保留 prompt，并使用 2-bit 打包。长上下文选择实验可加
`--prefill-keep-ratio 0.3`。输入模型路径可指向已有的模型缓存目录；本目录不存权重。

`llama_attention.py` 提供 `install_llama_phasekv_attention(model)`；
`llama.py` 提供命令入口和单 token 解码函数；`__main__.py` 支持 `python -m llama`。
