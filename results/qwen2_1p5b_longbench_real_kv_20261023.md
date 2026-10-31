# Qwen2-1.5B · LongBench KV 对比

> 2026-10-24 已更新：主表 SCOPE 替换为修正后的 ALLKV + Slide（GQA适配）同参复测，其他五种方法保留2026-10-23结果。另附32-token生成预算敏感性测试。

状态：完成。模型：`Qwen/Qwen2-1.5B-Instruct`。
每任务 32 条；正文截断 1536 token；BF16 模型、eager、贪心生成。

Full/驱逐方法的 KV 为 BF16；打包方法的 scale/min 和高精度残差为 FP16（均为 16-bit），沿用已有 packed kernel 精度。
KIVI 和 PhaseKV 使用真实低比特打包；伪量化入口被设为立即报错。其他压缩方法使用 BF16 token 驱逐。
KIVI 的 K 按通道分组，V 按 token 分组；采用官方独立 K/V 残差更新时机。PhaseKV 使用现有真实打包缓存。
KV 节省以同一条生成轨迹已经写入缓存的 token 数为基准（prompt + generated - 1），修正旧表的一 token 偏差。
峰值为 PyTorch allocated，包含模型及注意力中间量；KV 字节不含 算法辅助状态。
墙钟时间包含压缩维护和测量开销；不代表单独 attention kernel 的速度。

| 方法 | 样本 | QA F1 | 逻辑 KV 节省% | 实际 KV 节省% | 平均 KV MiB | 峰值 MiB | 平均每题 ms | 对 full token 一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 160 | 24.85 | 0.00 | 0.00 | 42.69 | 3456.0 | 461.7 | — |
| kivi | 160 | 24.89 | 78.90 | 78.90 | 9.01 | 3466.2 | 508.8 | 0.8249 |
| snapkv | 160 | 25.05 | 69.64 | 69.64 | 12.97 | 5001.5 | 470.5 | 0.8553 |
| scope | 160 | 24.85 | 0.00 | 0.00 | 42.69 | 5007.2 | 461.7 | 1.0000 |
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

## SCOPE 修正后的复测（2026-10-24）

两组均使用相同的5任务×32条输入，逐条核验输入token、答案和数据文件哈希。正文截断1536，添加模板后1211–1555 tokens，平均1552.81；BF16、贪心、原任务输出上限、正常遇EOS停止。
评分已由历史累计注意力改为当前query注意力；当前支持ALLKV + Slide的GQA组内聚合适配，未实现Adaptive、Discontinuous或压缩prompt插件。参考 [SCOPE核对报告](../SCOPE/AUDIT.md)。

| 生成段预算 / 近期窗口 | QA F1 | 实际KV节省% | 平均KV MiB | 触发样本 | 淘汰步数 | 输出与full完全相同 |
|---|---:|---:|---:|---:|---:|---:|
| 256 / 16 | 24.8465 | 0.0000 | 42.6863 | 0/160 | 0 | 160/160 |
| 32 / 16 | 24.8876 | 0.0295 | 42.6728 | 5/160 | 76 | 159/160 |

256预算组仍未触发淘汰，160条输出与旧full和旧SCOPE完全一致；这不是有效压缩效果的证据。
32预算组有5条真正发生淘汰；样本：multifieldqa_en:3, multifieldqa_en:4, multifieldqa_en:5, multifieldqa_en:17, multifieldqa_en:30。由于完整保留prompt且大部分回答很短，总KV节省很有限，不能据此评价长输出场景。该组参数不同，单独展示，不替换主表。
32预算组相对full的F1差值为+0.0411，按任务配对bootstrap的95%区间为[+0.0000, +0.1234]；仅用于当前样本的描述。

### 速度口径

| 生成段预算 | 平均输出tokens（含EOS） | 端到端输出tokens/s | 平均每题ms |
|---|---:|---:|---:|
| 256 | 9.29 | 20.13 | 461.70 |
| 32 | 9.28 | 19.56 | 474.17 |

以上速度包含prefill、缓存维护和触发日志采集，不是纯decode速度；其他方法的旧耗时没有同样的触发日志开销，也不是本轮同时重测，不宜据此排名速度。

### 原始记录与追溯

- [同参256预算逐题记录](qwen2_1p5b_scope_b256_20261024.predictions.jsonl)
- [补充32预算逐题记录](qwen2_1p5b_scope_b32_20261024.predictions.jsonl)
- [复测验证和汇总](scope_rerun_validation_20261024.json)
- [更新前报告备份](archive/before_scope_rerun_20261024/qwen2_1p5b_longbench_real_kv_20261023.md)

主报告JSON及逐题文件同步替换SCOPE行，保留各条记录真实的protocol_hash，并在meta中列出各方法所属运行。旧bootstrap区间经逐题核验未改变；其他方法没有补造触发记录。
