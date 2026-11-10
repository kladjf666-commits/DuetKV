# PhaseKV 单卡 A800 80GB 复测

解压后进入本目录。数据、旧评分器、优化版 PhaseKV、四个已有对比算法和 MILLION 的源码已入包。ClusterKV 及其递归子模块在安装时从作者固定提交获取。准备 Python 3.10 或 3.11、Linux CUDA 开发镜像、nvcc、git、flock、编译工具。ClusterKV 推荐作者测试的 CUDA 12.8 工具链，需 /usr/bin/gcc-11 和 g++-11，以及够用的磁盘、主机内存和可锁页内存。安装需访问 PyPI、PyTorch、GitHub 及其依赖源，正式测试和码本校准读取本地数据。

```bash
export PHASEKV_MODEL=/data/Llama-3.1-8B-Instruct/snapshots/master
export A800_OUT=/data/phasekv-results/a800_paper_v1
export CUDA_VISIBLE_DEVICES=0
# 若系统的 python3 不是 3.10/3.11，例如 export PYTHON_BIN=python3.10
bash run.sh plan
bash setup.sh main
bash run.sh check
bash setup.sh million
bash run.sh calibrate
bash setup.sh clusterkv
bash run.sh new-check
```

check 先用短输入检查模型、非空压缩、输出有限性、同请求单跑与合批的 logits，跨过首次冷却，并用质量配置生成 3 个 token。两个新增方法单独 B1，ClusterKV 的 smoke 还跨过一次 320 步重聚类。检查不通过会返回非零并写 run.log；同组其余算法继续验收，速度和质量阶段只接受已通过验收的算法。首次 JIT 或编译的墙钟时间单独看安装日志。指定 `A800_MAIN_METHODS='full phasekv'` 或 `A800_NEW_METHODS=clusterkv` 可单独重试。已有同版本环境可用 A800_MAIN_PYTHON、A800_MILLION_PYTHON、A800_CLUSTER_PYTHON 指向对应 Python 可执行文件，MILLION 的零 residual 扩展及 ClusterKV 的补丁仍须由此包安装。切换依赖、补丁或权重时用新输出目录。

## 测速

```bash
# 先快速得到 9192+256 的 B1、B2、B3、B4、B5、B6、B7、B8 曲线
bash run.sh speed --shapes 9192+256
bash run.sh new-speed --shapes 9192+256
# 补全四个长度；相同轮数下已完成或 OOM 的单元会跳过
bash run.sh speed
bash run.sh new-speed
# 正式三轮中位数可选。按轮数独立建档，只与相同轮数的 Full 比较
# bash run.sh speed --rounds 3
# bash run.sh new-speed --rounds 3
```

四档为 9192+256、8192+2048、32768+256、32768+2048。每个测点先完整预热一轮，再正式计时（默认一轮）。一次只运行一个算法。Full 和 PhaseKV 的测速 prompt 均 100%，分块 512 prefill；其他方法按各自的原生 prefill 或观测窗口执行。固定输出不提前 EOS。总解码吞吐 = batch × (输出长度−1) / 同步后的解码秒数；同时输出每序列吞吐、TTFT、prefill/decode 峰值及 KV 大小。预填充、首次输出 token 和正常解码分段记录。显存不足记 OOM，原生算子异常保留错误。按协议继续运行不会清除既有数据。

## 真实题面的批量吞吐

```bash
bash run.sh speed --data-groups longgenbench longbenchv2
bash run.sh new-speed --data-groups longgenbench longbenchv2
# 需要全部任务时可接续运行，已成功的相同测点直接复用
bash run.sh speed --data-groups all
bash run.sh new-speed --data-groups all
```

每个旧任务和新增 RULER 任务各组 8 条不同题面，LongBench v2 从五个领域合并选 8 条；每组按输入长度由长到短固定选择。取组内最短长度，其余行从中间删去差额，保留首尾；每行原题 ID、原始哈希、删去 token 数和最终哈希可追溯。B1 到 B8 按同一顺序逐行增加请求，固定生成 256 token，单独对照同组 Full。质量环节始终读取完整的冻结样本。

## 质量

```bash
# 每任务先取 1 条，新增 32K RULER 与旧 RULER 分开取
bash run.sh quality --limit 1
bash run.sh new-quality --limit 1
# 全量质量，每个算法 1150 条、B1，自然 EOS
bash run.sh quality
bash run.sh new-quality
bash run.sh summary
```

可先跑 `bash run.sh quality --suites longgenbench ruler32`，新增算法对应 `new-quality`。旧样本保留 LongBench 10 项 ×64，RULER 4 项共 384 条，LongGenBench 的 GSM8K、CSQA 和 MMLU 共 48 条。新增 RULER 四项 ×16 条（32K 输入加答案预算），以及 LongBench v2 按完整题面长度选择的 14 道多选题（实际 28,776–32,695 token）。后者使用本地直接作答提示与解析规则。旧样本的输出上限、题序、EOS 和评分器随包冻结。质量与固定步数测速分开；不把合成吞吐输入用于评价正确率。

输出目录含 summary.json、run.log、install/ 安装日志、各方法 gate.json、protocol.json、逐题模型原文和测速原始记录。汇总中正的 vs_full_throughput_pct 表示比 Full 快；质量分差只算与 Full 都成功的相同题 ID，同时给出覆盖率、OOM 和未完成数。各算法的论文设置和源码适配见 PAPER_PROFILES.md。
