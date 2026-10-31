> **无效结果：FP16 下输出退化，已排除，不能作为算法质量结果。正式结果见 qwen2_1p5b_longbench_real_kv_20261023.md。**

# Qwen2-1.5B · LongBench KV 对比

状态：进行中。模型：`Qwen/Qwen2-1.5B-Instruct`。
每任务 32 条；正文截断 1536 token；FP16、eager、贪心生成。

KIVI 和 PhaseKV 使用真实低比特打包；伪量化入口被设为立即报错。其他方法使用 FP16 token 驱逐。
KIVI 的 K 按通道分组，V 按 token 分组；采用官方独立 K/V 残差更新时机。PhaseKV 使用现有真实打包缓存。
KV 节省以同一条生成轨迹已经写入缓存的 token 数为基准（prompt + generated - 1），修正旧表的一 token 偏差。
峰值为 PyTorch allocated，包含模型及注意力中间量；KV 字节不含 SCOPE 的重要度等辅助状态。
墙钟时间包含压缩维护和测量开销；不代表单独 attention kernel 的速度。

| 方法 | 样本 | QA F1 | 逻辑 KV 节省% | 实际 KV 节省% | 平均 KV MiB | 峰值 MiB | 平均每题 ms | 对 full token 一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 160 | 0.00 | 0.00 | 0.00 | 44.53 | 3455.9 | 1454.0 | — |
| snapkv | 16 | 0.00 | 64.74 | 64.74 | 16.21 | 5019.2 | 2061.3 | 1.0000 |

逐题预测、答案、生成 token 和字节数见同名 `.predictions.jsonl`。
原 7B 结果不能用于推断本次 1.5B 的精度变化。重新取得同一数据集，并保存了文件及输入 token 哈希；旧实验没有数据哈希，无法证明下载版本逐字节相同。
