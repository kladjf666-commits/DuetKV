# Qwen2-1.5B 压缩触发验证（2026-10-24）

用途：目录迁移后验证实际缓存维护和日志。使用同一条 narrativeqa 输入，强制输出到固定长度、忽略结束符；不是质量或速度评估。
GPU：NVIDIA GeForce RTX 3070 Ti Laptop GPU；实际输入：1555 tokens；模型 BF16。

| 方法 | 输出长度 | prefill触发步数 | decode触发步数 | 首次decode触发 | 状态 |
|---|---:|---:|---:|---:|---|
| full | 80 | 0 | 0 | — | baseline_no_compression |
| kivi | 80 | 1 | 79 | 1 | triggered |
| snapkv | 80 | 1 | 0 | — | triggered |
| scope | 80 | 0 | 0 | — | not_triggered |
| pyramidkv | 80 | 1 | 0 | — | triggered |
| phasekv | 80 | 1 | 2 | 48 | triggered |
| scope | 258 | 0 | 1 | 257 | triggered |

触发步数按整个模型的一次维护边界统计，不乘层数。逐层 token 数、K/V 量化量、冷区批次数、压缩前后存储量和最终各层布局，均保存在同名 JSON 的 compression_trace。

SCOPE 保持256的生成段预算和16的近期窗口；80输出只写入79个生成缓存，未触发；258输出写入257个生成缓存，恰好触发一次。
SCOPE 当前是 ALLKV + Slide 的 GQA 组内聚合适配，尚未实现 Adaptive/Discontinuous 和压缩 prompt 插件，不能称为官方所有策略的完整复现。

2026-10-23 的旧质量结果不改写，不补造其未采集的事件；正式比较脚本已接入新日志，并升级协议标识，拒绝把新日志与旧预测混合续跑。
