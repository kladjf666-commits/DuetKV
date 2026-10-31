# PhaseKV

本项目原创实现集中于本目录，不伪装为另外下载的官方仓库。

- `phasekv_cache.py`、`kernels/`：真实低位打包缓存与注意力内核。
- `prompt_tier.py`：prompt token 选择。
- 对应模型实现已集中到 `models/Qwen2/`、`models/Llama/`、`models/OPT/`；本目录下 `qwen2/`、`llama/`、`opt/` 仅保留兼容入口。
- `model_runner.py`：多个模型共用的运行逻辑。
- `compressor.py`、`quant.py`、`hetero_cache.py`、`decode_tier.py`：历史评估路径，含伪量化；禁止用于要求真实量化的正式对比。正式对比继续由 `scripts/compare_longbench_kv.py` 阻断伪量化入口。

推荐入口：`python -m PhaseKV.qwen2`。旧 `src.phasekv`、`python -m qwen2`、`python -m llama` 和 `python -m opt` 保留转发兼容。

## 长输出解码优化：统一读取冻结分段

`fused_phasekv_decode()` 现在使用 `kernels/segmented_attention.py`：一次 Triton
启动并行读取 prompt、prompt extensions、所有 cold 段及 FP16 hot 区，第二次启动
用稳定的 softmax 归约合并结果。旧逐段实现保留为
`segmented_phasekv_decode_reference()`，用于正确性及速度对照。

这是对 KIVI 统一读取方式的借鉴，不采用每次 `torch.cat` 复制整份历史缓存。
量化码、scale/min、各段量化分组和 token 选择规则保持不变，也没有全量反量化到显存。
地址表及 FP32 partial 工作区按层复用，仅在新增冻结段、重排缓存或切换执行配置/流时重建。
hot 长度和地址在每次调用时传入；多轮输入扩容也能读取正确的热区。
浮点归约顺序变化可能导致微小数值差异，不承诺输出逐位相等。

每层注意力主计算为两次启动；缓存追加、冷却量化和模型其他算子仍有各自开销。
GPU 读取量、线程块数量和 partial 工作区仍随缓存增长。`physical_nbytes()` 延续原口径，
只统计 KV，不含地址表与注意力工作区；基准的实际 allocated 峰值包含这些开销。
外部代码若直接替换冻结段，应清除该层的 `_decode_workspace`；正常 Cache API 会自动清除。

在 Linux CUDA 环境先验证再测速（这些入口兼容本仓库缺失 `eval/` 的独立测试布局）：

```bash
.venv-4090d/bin/python -m unittest discover -s tests -p test_phasekv_segmented_decode.py -v
.venv-4090d/bin/python scripts/bench_phasekv_segments.py \
  --segments 0 16 125 189 253 --output results/phasekv_segments.json
```

该微基准对照新旧实现，计时包含 Python 提交开销，不使用 CUDA Graph 隐藏调度成本；
同时单列地址表重建耗时。它只测单层注意力，不代表完整模型 token/s。
本地已完成 CPU 生命周期测试、Triton 解释器数值对照、sm89 编译以及 RTX 3070
上的原生 CUDA 内核数值对照；PyTorch CUDA 调用路径和完整 7B/4090D 吞吐仍需在租卡环境验证。
其中原生 CUDA 对照使用 CPU PyTorch 参考值和 CUDA Driver API 启动编译后的内核，
不能替代完整模型适配器测试。旧 KIVI 测试的 `baselines.kivi_packed` 导入失效问题不在本次修改范围。

已有远程测试应先完成并保留原结果，再上传新代码。正式测试必须使用新的 `--out` 目录，
运行入口自带 GPU 数值检查；不要用修改后的代码 `--resume` 旧结果目录。
