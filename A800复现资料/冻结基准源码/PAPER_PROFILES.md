# 正式对比采用的论文配置

本表用于解释单卡 A800、统一 Llama-3.1-8B-Instruct 工作负载上的结果。每种方法选择论文实际报告的一档。原文没有覆盖的模型、数据集或输入输出长度，运行记录表示该配置在本实验的迁移。此次不做等显存预算对齐。

| 方法 | 本包采用的数值 | 依据和边界 |
|---|---|---|
| Full | FP16，全量 KV，FA2 | 共享模型、冻结输入和贪心解码的对照 |
| PhaseKV | 质量保留 95% prompt，速度保留 100%；冷区名义 80%（每批 16 留 13）；prompt/cold 均 4-bit，热窗 128 | 使用此次要求比较的项目定版，16 个核心源码来自 Code-Original-NoChanging；属于本项目参数 |
| KIVI | K/V 均 2-bit，group 32，FP16 residual 128 | KIVI §4.1 主配置。论文也报告 4-bit 和 R32，此包固定主配置 |
| SnapKV | 常规质量与测速取论文预算 2048；常规观察窗 32、maxpool 7。两项 NIAH 切到 1024、16、maxpool 5 | SnapKV §5.1.1、§5.1.2、LongBench 表 1。速度章节明确给出 2048，其余沿用 LongBench 设置 |
| PyramidKV | 平均预算 512，观察窗 α=8，β=20；maxpool 7 | 原文 §5.4、5.5；池化核采用发布实现的设置。沿用项目的逐 KV-head GQA 移植，尾部观测后执行分层筛选 |
| SCOPE | H2O prefill 2048，Discontinuous 解码预算 512，recent 256；观察窗 8，核 7 | 对应作者 LongGenBench-4K 脚本及附录 B.2。本项目 48 条复合题为 4K 输出集。短输出达不到预算时不会发生解码驱逐，记录为 prefill_only_short_output_boundary |
| MILLION | 4-bit/标量（d=128，M=64，每子空间 8 bit，256 个码字），K/V 分别校准 | 表 IV 的 4-bit 系统档，速度走作者 CUDA 路径，FP16 residual 128 由发布代码决定。质量采用 §IV-B LongBench 的 residual=0，通过新 token 立即编码、空残差项对 softmax 贡献为零实现 |
| ClusterKV | 初始 C0=floor(L/80)，sink 16，每 320 token 追加 4 簇，前两层 Full，余弦聚类与检索，CPU offload 开启，保留上一步的召回缓存。速度预算 1024、质量预算 256 | §III、IV、V 与表 I、图 12/13。两档均来自原文；质量选择表 I 最小档，使所有冻结样本的 prefill 都能满足上游 offload 入口预算。迭代 20 取自作者命令行默认值。完整 32,768 token 对应 409 个初始簇，文中的 32,000 对应 400 个。质量论文用 GLM4，本文测试将参数移植至 Llama |

## 有关实现的记录

- ClusterKV 的固定版本 CMake 默认编译 SM89，Llama 封装不识别 llama3 RoPE，近邻选择内核的共享缓冲区为 256。本包在固定提交上应用可重复补丁，编译 SM80，按 checkpoint 配置应用 RoPE，并将 CUDA 检索扩至 512 个簇。补丁修正排序方向和第二个扫描块前缀，将质心归一化后按余弦顺序检索，补齐计算与 CPU offload 的流依赖，并跳过未被原生 causal prefill 使用的稠密掩码。聚类、召回、decode 仍调用作者实现。验收对 26、409、457 个簇逐个核对近邻排序及 token 索引，并检验 32K RoPE。速度反映这一移植版。
- MILLION 的速度入口原本支持 SM80。为了实现论文的零残差质量档，小补丁只对空 residual 的归并项置零，质量驱动每步调用原始编码器。速度档继续调用作者的 decode 方法。校准采用作者脚本使用的 WikiText-2-raw-v1，取 train split，按作者 256×256 的目标每侧抽 65,536 个真实向量，共享字典、FAISS 25 轮。为统一所有任务，具体抽样是本包固定的 16 段、seed 42。用 Instruct 权重重新标定，结果记录语料、权重索引和两份码本的 SHA256。
- 批量测速使用无 padding、等长但不同内容的请求。技术适配、模型、硬件、输出长度或校准样本不同，不能将本次实测称为原论文成绩的逐项复制。以各样本结果中的 effective_options、mode、trace 和 protocol.json 复核实际配置。ClusterKV 的 final_kv_bytes 只表示 GPU KV 张量容量，另有 CPU pinned KV 和检索元数据；比较 GPU 峰值优先看 peak_allocated_bytes（PyTorch allocator）。

## 来源

[KIVI 原文](https://arxiv.org/abs/2402.02750) · [SnapKV 原文](https://arxiv.org/html/2404.14469v2) · [PyramidKV 原文](https://arxiv.org/html/2406.02069v2) · [SCOPE 原文](https://openreview.net/attachment?id=nMXtOspmrD&name=pdf) · [ClusterKV 原文](https://arxiv.org/abs/2412.03213) · [ClusterKV 固定源码](https://github.com/sjtu-zhao-lab/ClusterKV/tree/c7380819588771e8c5dadacdbd33942d7c53adea) · [MILLION 论文与固定源码](https://github.com/ZongwuWang/MILLION/tree/6d766701bdf6d3c0406fdc0fd92f530caa0359dc) · [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) · [LongBench v2](https://huggingface.co/datasets/THUDM/LongBench-v2)
