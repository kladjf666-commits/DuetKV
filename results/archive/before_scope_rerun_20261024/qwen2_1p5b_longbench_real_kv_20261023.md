# Qwen2-1.5B · LongBench KV 对比

状态：完成。模型：`Qwen/Qwen2-1.5B-Instruct`。
每任务 32 条；正文截断 1536 token；BF16 模型、eager、贪心生成。

Full/驱逐方法的 KV 为 BF16；打包方法的 scale/min 和高精度残差为 FP16（均为 16-bit），沿用已有 packed kernel 精度。
KIVI 和 PhaseKV 使用真实低比特打包；伪量化入口被设为立即报错。其他压缩方法使用 BF16 token 驱逐。
KIVI 的 K 按通道分组，V 按 token 分组；采用官方独立 K/V 残差更新时机。PhaseKV 使用现有真实打包缓存。
KV 节省以同一条生成轨迹已经写入缓存的 token 数为基准（prompt + generated - 1），修正旧表的一 token 偏差。
峰值为 PyTorch allocated，包含模型及注意力中间量；KV 字节不含 SCOPE 的重要度等辅助状态。
墙钟时间包含压缩维护和测量开销；不代表单独 attention kernel 的速度。

| 方法 | 样本 | QA F1 | 逻辑 KV 节省% | 实际 KV 节省% | 平均 KV MiB | 峰值 MiB | 平均每题 ms | 对 full token 一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 160 | 24.85 | 0.00 | 0.00 | 42.69 | 3456.0 | 461.7 | — |
| kivi | 160 | 24.89 | 78.90 | 78.90 | 9.01 | 3466.2 | 508.8 | 0.8249 |
| snapkv | 160 | 25.05 | 69.64 | 69.64 | 12.97 | 5001.5 | 470.5 | 0.8553 |
| scope | 160 | 24.85 | 0.00 | 0.00 | 42.69 | 5007.2 | 480.5 | 1.0000 |
| pyramidkv | 160 | 25.09 | 69.65 | 69.65 | 12.96 | 5001.5 | 468.1 | 0.7931 |
| phasekv | 160 | 24.17 | 91.29 | 91.29 | 3.72 | 5004.5 | 516.7 | 0.7987 |

逐题预测、答案、生成 token 和字节数见同名 `.predictions.jsonl`。
SCOPE 的 256-token 生成段预算高于本组任务最多 127 次 decode KV 追加，因此本配置未触发驱逐；不能据此评价 SCOPE 的有效压缩效果。
SnapKV、SCOPE、PyramidKV 使用仓库中的 GQA 适配实现。PyramidKV 为保持平均预算采用线性插值取整，与官方实现的整数步长取整有所不同。
量化方法使用本项目的 Triton 内核，不是官方 KIVI CUDA 扩展的性能复现；内核和模型适配已单独通过 GPU 数值验证。
原 7B 结果不能用于推断本次 1.5B 的精度变化。重新取得同一数据集，并保存了文件及输入 token 哈希；旧实验没有数据哈希，无法证明下载版本逐字节相同。

## 核对结果

960 条预测齐全，各方法使用逐条相同的输入 token 哈希；320 条量化方法预测全部记录了 uint8 载荷、元数据和残差的实际存储字节，未发现 dense KV 副本。

KIVI 残差更新规则核对自[官方实现](https://github.com/jy-yuan/KIVI/blob/main/models/llama_kivi.py)。本地 packed 内核、BF16 Qwen 适配器与 PhaseKV 数值测试通过 7 项；驱逐行为和伪量化禁用测试通过 4 项。

| 方法 | 相对 full 的 QA F1 差值 | 配对 bootstrap 95% 区间 |
|---|---:|---:|
| kivi | +0.05 | [-1.65, +1.95] |
| snapkv | +0.21 | [-1.59, +2.19] |
| scope | +0.00 | [+0.00, +0.00] |
| pyramidkv | +0.24 | [-1.59, +2.27] |
| phasekv | -0.67 | [-3.24, +2.03] |

此区间仅描述当前 160 条样本的变化，不能替代完整 LongBench 测试。此前 FP16 输出异常的一轮已标记为 INVALID_FP16，未参与本表。

## 复跑

在项目根目录使用现有 `.venv`，GPU 需对运行进程可见：

```bash
HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python scripts/compare_longbench_kv.py --longbench-dir .cache/longbench --out results/qwen2_1p5b_real_kv_new
```
