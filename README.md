# PhaseKV 项目交接

截至 2026-11-01，当前任务是在单张 RTX 4090 上用 Llama-3.1-8B-Instruct、FP16、batch=1 复测新版 PhaseKV。80% 离线包已准备好，本地 GPU 数值检查已通过。本目录尚无本轮正式结果。已回传的 `PhaseKV_4090_channel_only_result` 是上一轮 **100%** 的批量测试。

## 本轮改动

- 优化分段注意力读取打包 K 的布局，使相邻通道在内存中连续。生成时最近 128 个位置保留 FP16，每积累 16 个可冷却位置，按各 KV 头的 V 向量范数保留 13 个。配置保留率为 0.80，冷块实际保留 81.25%。内核已适配 13-token 冷块，prompt 仍保留 70%，prompt 和冷块均为 4-bit。
- 增加只跑新版 PhaseKV 的入口，沿用冻结输入、评分、生成停止条件和计时方式。共 1,126 条正式记录，其中 LongBench 640 条、RULER 384 条、LongGenBench 48 条，另有 6 种性能形状 × 3 个提示 × 3 轮。上次单卡排除的 3 种性能形状继续跳过。

## 文件结构

| 位置 | 用途 |
|---|---|
| [PhaseKV/](PhaseKV/) | 算法源码。`phasekv_cache.py` 管理缓存和冷却，`kernels/` 包含打包、分段注意力与内核选择。 |
| [official_llama/](official_llama/) | Llama 冻结基准、运行时和评分器。`bench_assets/frozen/` 保存输入，`delivery/` 保存离线环境包。 |
| [workspace_migration/keep80/](workspace_migration/keep80/) | 当前实验。`build.py` 打包，`campaign.py` 运行，`compare.py` 与历史记录配对。相邻的 `channel_retest/` 和 `decode_probe/` 属于前两轮实验。 |
| [PhaseKV与Full消融/](PhaseKV与Full消融/) | 当前上传包、过往 4090 实测和诊断。实验参数、图表和结论见 [READ.md](PhaseKV与Full消融/READ.md)。 |
| [不同算法的不同数据集测试/](不同算法的不同数据集测试/) | `native_llama_20261029/` 保存 4090D 原始结果，`analysis_20261031/` 保存[分析报告](不同算法的不同数据集测试/analysis_20261031/分析报告.html)和[质量对比](不同算法的不同数据集测试/analysis_20261031/质量对比详解.html)。 |
| [models/](models/)、[scripts/](scripts/)、[tests/](tests/) | 模型适配、基准脚本和回归测试。KIVI、SnapKV、PyramidKV、SCOPE 的实现各在同名目录。 |
| [results/](results/)、[reports/](reports/)、[docs/experiments/](docs/experiments/) | 早期 Qwen 等实验的记录、报告和说明。算法原理与早期复现命令见[原项目总览](PROJECT_OVERVIEW.md)。 |

## 接手后怎么跑

使用已配置好的 4090 工作空间，将[本轮上传包](PhaseKV与Full消融/PhaseKV_4090_keep80_full.zip)传到服务器的 `/root/data`，执行以下命令。

```bash
cd /root/data
unzip -o PhaseKV_4090_keep80_full.zip
nohup bash migration/keep80/run.sh >> /root/data/keep80_full.log 2>&1 &
tail -f /root/data/keep80_full.log
```

在同一设备、相同协议下可用同一命令续跑。日志出现 `FINISHED` 后，取回 `/root/data/phasekv-results/rtx4090_channel_keep80_full_v1/`。先查 `summary.json` 的成功、OOM 和待跑条数，再读 `comparison.json`。

速度优先对照历史 RTX 4090 Full，全量质量对照来自 4090D。与原生 PhaseKV 的历史比较同时包含内核和保留率的变化。优化版 100% 在 B=1 只有一个 8192+256 提示可作参照。1,072 条质量样本中有 832 条的输出上限不超过 128，分析冷区裁剪影响时，另按 `final_trace.cold_segments > 0` 统计。

**重新打包或换机交接**时，保留完整工作目录中的离线包和原始结果。Git 默认忽略 ZIP 和权重。当前构建要读取 `PhaseKV_4090_channel_only.zip`、`rtx4090_partial_results.tar.gz`、`PhaseKV_4090_channel_only_result/` 和 4090D 原始记录。在项目根目录运行 `python workspace_migration/keep80/build.py` 即可重建上传包。新工作空间需要的依赖已保存在 `official_llama/delivery/` 的两个大包 `PhaseKV_new_workspace_cp310_cu121.zip` 和 `PhaseKV_torch_cuda121_offline.zip` 中。
