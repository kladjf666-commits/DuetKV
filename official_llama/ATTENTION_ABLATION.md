# Llama 注意力与 KV 缓存四组消融

本轮使用独立入口、独立输出目录。原来的 `benchmark.py`、`native_runtime.py`、
`runtime/`、冻结数据和正式实验结果均不修改。

## 四组定义

| 标识 | 常驻缓存 | 解码注意力 |
|---|---|---|
| `full_fa2` | Full FP16 DynamicCache | FlashAttention 2 |
| `phasekv_fa2` | 原 PhaseKV packed prompt/cold + FP16 hot | 每步逐层解包成临时 FP16 KV，再调用同一 FlashAttention 2 |
| `full_triton` | Full FP16 DynamicCache | 共用分段 Triton 内核的 FP16 分支 |
| `phasekv_triton` | 原 PhaseKV packed prompt/cold + FP16 hot | 共用分段 Triton 内核的压缩分支，延续原融合算法 |

全部使用同一模型权重、FP16、batch=1、GQA 32/8、绝对 RoPE、贪心解码和冻结输入。
四组解码共用 Q/K/V 投影、位置编码和输出投影代码。PhaseKV 参数保持原配置：
prompt 保留 0.7，prompt/cold 4bit，hot 128，lazy 16，group 32，decode keep=1。
prompt 选择、冷却、页式存储、hot ring 和优化开关不变。

Full 的 Triton 路径与 PhaseKV 共用分段注意力、FP32 在线 softmax 和归并代码，
使用同一切块规划器。两种缓存的有效长度、段数和内存布局不同，因此实际切块数不同。
该组用于控制计算框架，不能据此替代高效的 Full+FA2 基线或宣称超过 FA2。

## 公平性边界

- 主要速度对比使用 performance 固定输入/输出长度；自然停止的质量测试速度仅供描述。
- `phasekv_fa2` 的解包、FP16 写回、hot 拷贝、缓存追加/冷却都包含在解码计时内。
  解包使用每冻结段一次 Triton launch，直接写最终临时缓冲区，未维护完整 FP16 影子缓存。
  这是可审计的接入实现，不能视为 FA2 解包路径的最佳性能上界。
- 临时 FP16 KV 逐层释放；显存分别报告常驻 KV、常驻辅助空间、单层临时 KV 大小和
  CUDA 总分配峰值。总峰值已经包含临时空间，不应再次相加。
- FA2 的输入必须为 FP16，本组解包后会发生 FP16 舍入；原融合路径在寄存器中以 FP32
  反量化和累加。因此两个 PhaseKV 后端不承诺逐位相同，需通过数值门禁并实际重测质量。
- 被保留 K 已包含原绝对位置的 RoPE，禁止重新编号后再次旋转。单 token 解码中所有
  retained KV 均来自过去或当前，FA2 使用右下对齐 causal 语义。
- Full prefill 延续原 FA2；两组 PhaseKV prefill 延续 FA2 前缀、eager 末尾观察窗口、
  选择和打包。TTFT 包含这些缓存策略成本，不能声称四组 prefill 内核完全相同。
- `full_fa2` 解码使用共用适配器直接调用 FA2，GPU 门禁会对照原 HF Full 路径。
  `phasekv_triton` 也会与原融合路径比较。旧实验结果仅作历史参考，不混入本轮汇总。

## 上传与启动

上传 `PhaseKV_attention_ablation_20261031.zip` 到 `/root/KV/PhaseKV/`，在该目录解压。
压缩包自带 `official_llama/` 前缀，只增加本轮文件；需要之前已验证的完整服务器部署。

```bash
cd /root/KV/PhaseKV
unzip PhaseKV_attention_ablation_20261031.zip
bash official_llama/start_attention_ablation.sh
```

不重装 torch、CUDA、Transformers 或 flash-attn。默认环境与路径：

- Python：`/data/phasekv-native-venv/bin/python`
- 模型：`/data/Llama-3.1-8B-Instruct/snapshots/master`
- 新结果：`/data/phasekv-results/attention_ablation_20261031`

启动脚本会验证增量包、原源码指纹、冻结输入和模型配置，再后台运行；与旧实验共享
进程锁，并拒绝在 GPU 已有计算进程时竞争运行。不要在旧任务还在运行时启动。

```bash
tail -f "$(cat /data/phasekv-results/attention_ablation_20261031/latest_log.txt)"
# 数值门禁进行中，详细进度写在这里：
tail -f /data/phasekv-results/attention_ablation_20261031/gpu_gate.log
```

## 门禁与正式测试

本机是 CPU 环境，**交付时 GPU 门禁和完整实验尚未执行**。服务器自动先运行：

1. FP16 Full Triton 对照独立 FP32 注意力参考和 FA2；覆盖 MHA/GQA、非整齐长度及 16K。
2. 2/4/8bit 解包对照独立 unpack 参考，要求逐值一致；覆盖 hot 环绕、冷段分组、页扩容。
3. 新旧 PhaseKV fused 内核逐值比较；再与物化 FA2/FP32 参考比较。
4. tiny Llama 固定 teacher tokens 的多步 logits 对照，覆盖绝对 RoPE、裁剪和冷却。
5. 实际 8B 模型逐组完成 257 输入 + 161 输出 smoke；不计入正式性能。

数值门禁沿用原 CUDA 检查容差 `atol=0.005, rtol=0.01`；纯解包和新旧融合路径要求
逐值一致。失败即停止，详情保留在 `gpu_gate.json`，不会继续跑出一批可疑结果。

通过后默认测试全部 4 组，共 **4,612 条正式记录**：

- LongBench：640 条输入 ×4 = 2,560。
- RULER：384 条输入 ×4 = 1,536。
- LongGenBench：48 条输入 ×4 = 192。
- Performance：27 条输入 ×3 次测量 ×4 = 324；另有独立诊断预热。

每条性能输入先做完整预热/显存诊断，然后重复三次计时。首次 JIT 编译不进入性能汇总。
OOM 记录单独保留，其他异常中止后续工作。OOM 并不等于整个实验未完成。

可先只检查门禁，随后以相同输出目录启动完整实验：

```bash
bash official_llama/start_attention_ablation.sh --gate-only
# 上一命令所启动的后台门禁完成后，再运行：
bash official_llama/start_attention_ablation.sh
```

中断后重复完整启动命令可续跑。每次启动重跑门禁，已完成正式记录跳过。
`--retry-failed` 会重试失败记录。若修改源码、方法列表或测试套件，必须换新输出目录。
仅重建汇总（不使用 GPU）：

```bash
/data/phasekv-native-venv/bin/python official_llama/attention_ablation.py \
  --out /data/phasekv-results/attention_ablation_20261031 --summarize-only
```

## 结果含义

`summary.csv`：每任务质量分数、成功/失败覆盖率、速度和显存。
`ablation_pairs.json`：四种成对对比，速度先按同一输入三次取中位数；任何一侧缺次或
失败都明确列为未配对，不填零、不用不同输入均值相除。
`quality_paired_vs_full.json`：相对本轮 Full+FA2 的成功样本配对质量差与文档级 bootstrap。
`records.jsonl`、`records/`、`diagnostics/`、`environment/`：完整可追溯记录。

优先看 PhaseKV+FA2 对 Full+FA2，以及 PhaseKV fused 对 Full Triton。
再看 Full Triton 对 Full+FA2 的后端差距，以及 PhaseKV fused 对物化 FA2 的融合收益。
质量指标按任务解释，不把 F1、ROUGE、检索命中率和选择题准确率混称为统一准确率。

运行完成后可把结果压缩到指定易访问目录：

```bash
tar -czf /root/KV/PhaseKV/attention_ablation_20261031_results.tar.gz \
  -C /data/phasekv-results attention_ablation_20261031
```

## 回滚

本地修改前提交：`e0179c3baf6c6ea5eb047d3a3e209b84afc3ffb9`。
本地回滚标记：`pre-attention-ablation-20261031`；新增工作分支：`codex/attention-ablation`。
Git 只能回滚已跟踪并提交的文件，未跟踪的 vendor/bench_sources 独立仓库不自动受保护。
本轮不改这些目录。服务器恢复旧实验只需使用原 `start_benchmark.sh`，无需替换 runtime。
本轮增量包的云端备份应以最终提交和远端确认结果为准，不能由 Desktop 的 Changes 面板推断。
