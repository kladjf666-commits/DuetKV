# PhaseKV

PhaseKV 将 prompt、最近生成的 token 和较早生成的 token 分开存储。解码时，分段注意力内核直接读取压缩的 KV。当前定版基线使用单张 RTX 4090 和 Llama-3.1-8B-Instruct（FP16）。在 8192 输入、2048 输出下，batch 1 的解码速度与 Full FA2 接近，batch 6 的整批吞吐从 132.01 提高到 164.99 token/s。[基线提交 `e9e898e`](https://github.com/kladjf666-commits/PhaseKV/commit/e9e898e)

## 缓存策略

- **Prompt 保留率**决定预填充后每层、每个 KV 头留下多少输入位置。质量实验扫描 70%–100%；下面的吞吐实验统一保留 100%。两组实验都将保留的 prompt KV 量化为 4-bit。
- **冷区保留率**作用于生成阶段。最近 128 个位置保留 FP16，每积累 16 个可冷却位置，各 KV 头按 V 向量范数留下 13 个并量化为 4-bit。因此配置中的 80% 对应每个冷块实际保留 81.25%。
- 在上述 16 选 13 路径中，缓存将多层的选择和打包合并执行。分段注意力的描述表直接更新目标行，以减少临时 GPU 张量和小规模复制。[缓存实现](PhaseKV/phasekv_cache.py) · [描述表更新](PhaseKV/kernels/segmented_plan.py)

## RTX 4090 解码吞吐与显存

表中是**整批输出 token/s**，从首 token 之后开始计时。Full 与 PhaseKV 使用相同输入和固定输出长度。batch 1 取三轮正式测试的中位数；batch 2–6 每个配置完整预热一次、正式测一次，双方均按 512 个输入 token 分块预填充。

| Batch | 9192+256 Full | 9192+256 PhaseKV | 8192+2048 Full | 8192+2048 PhaseKV |
|---:|---:|---:|---:|---:|
| 1 | 44.50 | 44.19 | 44.56 | 44.72 |
| 2 | 74.46 | 78.26 | 74.32 | 78.88 |
| 3 | 93.32 | 105.00 | 94.34 | 106.49 |
| 4 | 109.26 | 126.73 | 109.69 | 129.86 |
| 5 | 120.98 | 144.73 | 122.45 | 148.67 |
| 6 | 130.59 | 打包 prompt 时显存不足 | 132.01 | **164.99** |

在 8192+2048、batch 6 下，PhaseKV 的吞吐比 Full 高 **24.98%**，最终 KV 从 **7.50 GiB** 降至 **2.43 GiB**，解码阶段的峰值显存从 **22.59 GiB** 降至 **17.65 GiB**。PhaseKV 在预填充及打包阶段的峰值为 **22.99 GiB**。9192+256 的 batch 6 在完整 KV 转为压缩 KV 时耗尽显存，因此这个点没有 PhaseKV 解码吞吐。[单序列计时](PhaseKV与Full消融/4090_prompt100_speed_2cases_VjjdG5Fk/summary.json) · [批量吞吐与峰值显存](PhaseKV与Full消融/4090_batch2_6_fAryLQ4D/summary.json)

## Prompt 保留率与作答质量

LongGenBench 的 GSM8K、CSQA、MMLU 各使用 16 条固定的多题提示，共 1600 道子题。batch 为 1，使用贪心解码，遇到 EOS 停止，最多生成 4096 个 token。七档实验只改变 prompt 保留率。

下表按**题号**核对答案。漏答或无法解析的题计错，三项平均对数据集等权。结果文件同时保留官方按答案出现顺序计算的分数；连续作答漏掉一题时，两种口径可能相差较大。[评分实现](official_llama/benchmark_metrics.py)

| Prompt 保留率 | GSM8K | CSQA | MMLU | 三项平均 |
|---:|---:|---:|---:|---:|
| 70% | 31.88% | 19.69% | 37.50% | 29.69% |
| 75% | 37.29% | 45.78% | 44.58% | 42.55% |
| 80% | 44.17% | 54.22% | 45.42% | 47.93% |
| 85% | 53.54% | 71.72% | 46.04% | 57.10% |
| 90% | 56.88% | 71.72% | 45.63% | 58.07% |
| 95% | **64.17%** | **75.31%** | 47.08% | **62.19%** |
| 100% | 60.00% | 74.84% | **48.75%** | 61.20% |

在这批提示上，95% 比 100% 净多答对 **15 题**（1016 对 1001）：GSM8K 多对 20 题，CSQA 多对 3 题，MMLU 少对 8 题。两档均未出现反复输出同一答案编号、整条提示无可解析答案的严重循环。测试预先以官方得分和严重循环数判定“接近全保留”：从该档到 100%，每档官方得分均不低于本任务 100% 得分减 3 分，且严重循环为零。**95% 是三个任务同时达标的最低已测档位**；每个任务仅有 16 条提示，尚不足以判断 95% 的准确率普遍优于 100%。[七档质量结果](PhaseKV与Full消融/4090_prompt_quality_scan/summary.json)

## 复现基线

在已配置好的 Linux 4090 环境中运行。基线使用 PyTorch 2.5.1+cu121、FlashAttention 2.7.4.post1、Transformers 4.46.3 和 Triton 3.1.0。模型目录默认为 `/root/data/Llama-3.1-8B-Instruct/snapshots/master/`，可用 `PHASEKV_MODEL` 指向其他位置。运行器从实验包加载相应版本的代码。

| 实验 | 上传到服务器的 ZIP | 本地构建脚本 |
|---|---|---|
| batch 1 两组解码长度 | `PhaseKV_4090_prompt100_speed_2cases.zip` | [build_prompt100_speed.py](workspace_migration/quick_4090d/build_prompt100_speed.py) |
| batch 2–6 吞吐与显存 | `PhaseKV_4090_batch2_6.zip` | [build_batch_speed.py](workspace_migration/quick_4090d/build_batch_speed.py) |
| 七档 prompt 质量扫描 | `PhaseKV_4090_prompt_quality_scan.zip` | [build_quality_scan.py](workspace_migration/quality_coverage/build_quality_scan.py) |
| 事后按未来注意力固定保留 70% | [PhaseKV_4090_future70.zip](PhaseKV与Full消融/PhaseKV_4090_future70.zip) | [build_future_oracle.py](workspace_migration/quality_coverage/build_future_oracle.py) |

完整工作区的实验包在 `PhaseKV与Full消融/`。Git 默认忽略 ZIP 和模型权重。重新打包时，单序列测速和质量扫描需要归档的 `PhaseKV_4090_prompt100_quality.zip`，批量测速需要 `PhaseKV_4090_prompt100_speed_2cases.zip`；构建脚本还会校验对应的历史结果和冻结输入。在仓库根目录用 Python 执行表中的脚本即可构建相应实验包。

将所需 ZIP 放到服务器的 `/root/data`，按实验选择一组命令：

```bash
cd /root/data

# 单序列解码速度
unzip -o PhaseKV_4090_prompt100_speed_2cases.zip
bash phasekv_4090_prompt100_speed_2cases/start.sh

# 批量吞吐与显存
unzip -o PhaseKV_4090_batch2_6.zip
bash phasekv_4090_batch2_6/start.sh

# 质量扫描（复用 70% 与 100% 端点，新生成 240 条）
unzip -o PhaseKV_4090_prompt_quality_scan.zip
bash phasekv_4090_prompt_quality_scan/start.sh

# 未来访问诊断（复用三个历史对照，只新增 48 次自由生成）
unzip -o PhaseKV_4090_future70.zip
nohup bash phasekv_4090_future70/start.sh > /root/data/future70.log 2>&1 &
tail -f /root/data/future70.log
```

运行结束时终端会打印 `/root/data/phasekv-results/` 下的具体结果目录，其中包含 `summary.json` 和 `run.log`。质量扫描可加 `--coarse` 先测 80% 和 90%，之后去掉该参数、在同一结果目录续跑其余三档。

### 未来访问诊断

未来访问诊断复用 48 份题单的普通 70%、85%、100% 历史成绩和 100% 组的完整输出 token。每份只强制回放一次已有输出，取出此前未保存的注意力。评分计入所有解码步以及同组查询头，softmax 分母包含 prompt 和生成历史。每层每个 KV 头固定保留末尾 32 个位置，其余预算按累计注意力取到 70%。重新预填充原始输入，用固定位置自由生成并评分。100% 组的 prompt 也是 4-bit，四组沿用相同的生成冷区设置。

结果在 `/root/data/phasekv-results/4090_future70_v1/`。运行中断可执行同一命令续跑，已落盘的题单自动跳过。完成时 `summary.json` 显示 `status=complete`、`new_free_generations=48`。每个任务的 `indexed_gain_pp` 是未来访问 70% 相对普通 70% 的按题号准确率增益（百分点）。`fraction_of_70_to_85_gap_recovered` 是该增益除以普通 85% 相对 70% 的增益，分母非正时留空。协议记录云机与历史对照的 GPU UUID 是否相同。

## 源码与历史实验

- [PhaseKV 算法源码](PhaseKV/) · [模型适配](models/) · [Llama 评测与冻结输入](official_llama/) · [测试](tests/)
- 早期 RTX 4090 全量复测采用 prompt 70%、冷区 80%，共完成 1126 条正式记录，涵盖 LongBench、RULER、LongGenBench 和性能测试。[80% 冷区复测报告](PhaseKV与Full消融/80%25复测结果对比.md) · [复现脚本](workspace_migration/keep80/) · [实验交接记录](PhaseKV与Full消融/READ.md)
- [RTX 4090D 历史多算法对比](不同算法的不同数据集测试/analysis_20261031/分析报告.html) · [算法原理及早期 Qwen、Llama 实验](PROJECT_OVERVIEW.md)
