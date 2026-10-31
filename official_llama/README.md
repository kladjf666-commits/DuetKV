# PhaseKV 原生 Llama 正式测试与交接

更新：2026-10-29。接手 agent 请先读本文件，再看 `config.json`、`benchmark.py` 和 `bench_assets/frozen/manifest.json`。

## 当前结论与第一步

**服务器已经完成环境安装与部署验证。下一步运行正式 benchmark，不要再次运行安装脚本或把 smoke 当成正式实验。**

用户报告已看到：

```text
Deployment smoke passed. This is not dataset accuracy or publication-ready throughput.
Validation artifacts: /root/KV/PhaseKV/official_llama/results/validation_20261029_103248
```

这是用户从远端提供的完成信息；本地 agent 没有远程 shell 连接，尚未下载审查该目录。本次新增正式脚本的完整 8B GPU 运行尚待服务器执行，不能写成已获得正式结果。

将 `delivery/PhaseKV_formal_benchmark_20261029.zip` 上传到服务器 `/root/KV/PhaseKV/`，执行：

```bash
cd /root/KV/PhaseKV
unzip -o PhaseKV_formal_benchmark_20261029.zip
cd official_llama
bash start_benchmark.sh
```

**这次入口会真正运行性能测试和三个数据集，不会再停在部署验证。** 脚本自行后台启动；不要额外再套一层 nohup，不要重复启动多个终端任务。既有代码、runtime、权重和环境必须保留，增量包不包含它们。

查看实时日志：

```bash
tail -f "$(cat /data/phasekv-results/native_llama_20261029/latest_log.txt)"
```

退出日志查看按 Ctrl+C。关闭 VSCode 不停止后台任务。若安装包完整性或评分依赖检查失败，启动脚本会立即报错；若后台失败，错误在上述日志里，不能仅凭启动时打印的 PID 判定成功。

## 服务器位置

| 内容 | 路径 |
|---|---|
| 代码根目录 | `/root/KV/PhaseKV` |
| 原生测试目录 | `/root/KV/PhaseKV/official_llama` |
| 已安装 Python 环境 | `/data/phasekv-native-venv` |
| 实际模型目录 | `/data/Llama-3.1-8B-Instruct/snapshots/master` |
| 最近成功的部署验证 | `/root/KV/PhaseKV/official_llama/results/validation_20261029_103248` |
| 本次正式结果 | `/data/phasekv-results/native_llama_20261029` |
| 本地项目 | `C:\Users\kladjf\Desktop\PhaseKV` |

模型上层 `/data/Llama-3.1-8B-Instruct` 不是 Transformers 权重目录。必须用包含 config、tokenizer 和四个 safetensors 分片的 `snapshots/master`。

`/root` 曾只剩约 2 GB，正式输出放 `/data`。`/data` 是共享存储，权重读取曾耗时数分钟至二十多分钟；进度条每完成一个分片才跳一次。模型按算法和测试阶段加载，绝不会每条样本重新加载。

## 已完成的工作

1. 固定 KIVI、SnapKV、SCOPE 官方提交，保存原始 `vendor/`、兼容后的 `runtime/`、补丁与 941 个源文件的哈希。算法压缩策略和原生内核未换成本项目旧的重实现。
2. 接入 Llama-3.1-8B、FP16、单卡、batch=1；Full 使用原生 FlashAttention2。模型权重不量化、不做 CPU offload。
3. 服务器已安装 PyTorch 2.5.1+cu121、Transformers 4.46.3、FlashAttention 2.7.4.post1 以及 KIVI 原生扩展。
4. 本地 CPU 接口/语义检查此前通过；用户报告服务器 CUDA kernel、tiny 和实际 8B 部署门禁通过。正式实验没有完成。
5. 新增正式 runner、自然停止质量生成、固定长度性能生成、官方评分、配对统计、OOM 留档、断点续跑和防重复启动。
6. 本地已生成冻结输入：LongBench 640、RULER 384、LongGenBench 48、性能形状 27。服务器使用随包输入，不重新下载或随机抽样。
7. 新增 CPU 回归测试，覆盖自然 EOS 与固定输出计数、评分缺答分母、断点续跑和失败统计。验收记录见 `validation/benchmark_delivery_checks.json`。

前一次 OOM 原因已经确认：用户的 shell 同时启动了三份验证任务，三份模型争抢显存，不是压缩算法的容量结果。这三份任务已全部退出。新入口使用两个锁，并在开始 GPU 工作前检查现存计算进程；它不会主动杀死任何进程。

## 代码结构

```text
official_llama/
├── README.md                       本交接文档（正式测试优先读它）
├── README.txt                      旧部署说明，保留以追溯
├── config.json                     已验证的模型环境与算法参数
├── sources.json                    原生源码与兼容副本哈希
├── vendor/{KIVI,SnapKV,SCOPE}/      固定提交的官方原始文件
├── runtime/                        实际导入的兼容副本与 PhaseKV 快照
├── patches/                        最小接口兼容补丁
├── native_runtime.py               Llama 加载、原生 forward/cache 适配与计量
├── prepare.py                      源码核验；服务器只用 --verify
├── setup_linux.sh                  旧环境安装入口，现在不要重跑
├── validate_linux.sh               部署验证入口，不自动进行正式测试
├── run.py                          旧 smoke/手动冻结输入诊断入口
├── check_model.py                  本地模型分片和索引结构检查
├── check_run.py                    部署结果门禁
├── start_benchmark.sh              正式测试一键后台启动/续跑入口
├── benchmark.py                    正式调度、隔离 worker、结果存储与汇总
├── benchmark_generate.py           统一生成、计时及实际 KV/辅助空间采集
├── benchmark_metrics.py            调用官方评分器；额外诊断单独命名
├── prepare_benchmark.py            本地数据选择、历史排除、分词与冻结
├── fetch_ruler_essays.py            获取官方 RULER URL 清单中的文章
├── bench_sources/RULER/            固定提交的官方数据生成/评分源码
├── bench_assets/
│   ├── frozen/samples.jsonl.gz     所有方法共用的输入 ID、标签、停止规则
│   ├── frozen/manifest.json        输入哈希、历史证据、数据选择与来源
│   ├── frozen/ruler_raw/           官方生成器原始输出，含 answer_prefix
│   ├── essay_sources.json          RULER 背景文章的来源/哈希
│   └── wheels/                    四个纯 Python 离线评分包
├── benchmark_bundle.json           新增交付文件哈希
├── verify_benchmark_bundle.py       增量包完整性检查
├── build_benchmark_bundle.py        构建增量交付包
├── tests/test_benchmark.py          新 runner 的 CPU 回归测试
├── validation/                     本地检查记录，不是正式 GPU 测试结果
├── results/                        旧部署验证结果（远端），保留
└── delivery/                       可上传增量 zip 与 sha256
```

项目根目录旧 `scripts/benchmark_4090d.py`、旧 Qwen 环境、Figure 和历史报告，不参与本轮运行。不要将历史速度或显存拼入本轮主表。不要修改 `runtime/` 来绕过 OOM 后仍称为同一个原生实现。

## 本轮配置与规模

主模型为 Llama-3.1-8B-Instruct，RTX 4090 D 23.52 GiB；所有方法 FP16、greedy、batch=1、相同输入 ID。参数从 `config.json` 读取，不由测试分数选择。

| 方法 | 配置 |
|---|---|
| Full | 完整 FP16 KV，FlashAttention2 |
| KIVI | K/V 4-bit，group=32，residual=128，官方内核 |
| SnapKV | prompt 容量 2048，window=32，maxpool，kernel=7 |
| SCOPE | H2O prefill 2048/recent8，Discontinuous decode 512/recent256 |
| PhaseKV | prompt 70%/4-bit；hot128；cold4-bit；decode 保留1.0；P5 开关见配置 |

SCOPE-Slide、SCOPE-Adaptive 仅补充 LongGenBench，每个变体 48 条。主算法始终是五个，不以最终分数挑最优 SCOPE 变体。短输出场景的 SCOPE 调度边界沿用已验证适配并记录 `scope_mode`。

### 数据集质量

| 套件 | 样本定义 | 主方法生成数 |
|---|---|---:|
| LongBench v1 | 10任务×32文档×8192/16384 两档 | 3200 |
| RULER | 4任务×4096/8192/16384×32 | 1920 |
| LongGenBench-4K | 3任务×16复合prompt | 240 |
| SCOPE 两个额外变体 | 仅 LongGenBench | 96 |
| 合计 | 质量生成一次，遵循自然 EOS | 5456 |

LongBench：narrativeqa、qasper、multifieldqa_en、hotpotqa、2wikimqa、gov_report、qmsum、multi_news、lcc、repobench-p。输出上限分别依官方配置 128/128/64/32/32/512/512/512/64/64。代码任务不套聊天模板；其他任务使用本地 tokenizer 的 chat template。长文本先按 token 头尾截断，再套模板并核验预算；记录是否截断。

RULER：niah_multikey_1、niah_multivalue、vt、cwe。官方 generator 未改动，调用参数列表记录在 manifest；使用 `meta-llama3` 模板，保留官方 answer_prefix。长度档包含预留输出额度，分别 128/128/30/120，不与 LongBench 的最大输入预算混淆。

LongGenBench：使用固定 SCOPE 仓库已构造的 GSM8K+/MMLU+/CSQA+，30/30/40道题，统一上限4096。采用官方 llama3_modify 模板，不继承官方 runner 删除特定 MMLU 索引的操作。同任务选中复合 prompt 的底层题目哈希不重叠。

### 关于历史数据与方案调整

原计划要求全部使用未见文档。核对本地历史 JSONL 后发现，narrativeqa 的原文上下文已覆盖；qmsum 按上下文排除后只剩5份可用未见文档。因此本轮采用**明示复测**，不是默默降低标准：

- narrativeqa：32份历史/共享上下文复测。
- qmsum：5份未见、27份历史/共享上下文复测。
- 其他任务：每任务32份按现有历史记录排除后的未见文档。
- 每条输入携带 `evaluation_split` 和 `heldout_from_available_history`。汇总按 split 分开；不得在论文中把所有640条称作独立留出集。
- 可用历史证据的路径和 SHA256 都在 manifest；不能据此证明没有未记录的手工测试。
- 若未来必须严格独立评估，由接手 agent 新增未使用任务/文档并另开输出目录。不可事后把本轮样本重新命名为 heldout。

### 固定长度性能

8个主形状：4096+256、8192+256、16384+256、1024+1024、1024+4096、1024+8192、8192+2048、16384+4096。另有32000+256容量扩展。

每形状3个冻结 prompt，每方法1次预热/存储诊断、3次正式测速：主形状360条，扩展45条，共405条正式测量；最多135条预热。性能测试忽略 EOS，固定生成长度，不能用于准确率。

方法按测试阶段轮换顺序；同阶段一个方法处理该阶段全部样本，减少共享盘反复加载权重。**没有逐样本交替五个方法**，这项取舍写入报告，温度/频率/功率记录用于检查运行漂移。

默认顺序：performance → longbench → ruler → longgenbench。每阶段每方法只加载一次模型。warmup 同时承担每步持久 KV 存储诊断，其时间不会进入吞吐主表。

## 如何运行、续跑与检查

环境已安装。`start_benchmark.sh` 只从随包 wheels 安装 rouge、jieba、fuzzywuzzy、six，不访问网络、不安装 torch、不编译 CUDA。不手动改算法参数。

启动/同目录续跑：

```bash
cd /root/KV/PhaseKV/official_llama
bash start_benchmark.sh
```

单条成功、OOM 或普通错误结果采用临时文件再原子替换；重启默认跳过已有结果。运行中断未完成的一条会重新执行。OOM 会记录并继续下一条；非预期异常会保存失败条目并停止本轮，以免错误污染后续。

修复运行条件后显式重试失败项：

```bash
bash start_benchmark.sh --retry-failed
```

只改运行条件可在同目录续跑；**代码、配置、冻结数据或模型路径变化会触发协议不一致并拒绝混跑**。要评估修复版本请使用新的 `PHASEKV_BENCHMARK_OUT`。保留旧输出，不删除失败证据。

```bash
PHASEKV_BENCHMARK_OUT=/data/phasekv-results/native_llama_fix1 bash start_benchmark.sh
```

显式运行子集时也使用新的输出目录，不能与默认全协议混合：

```bash
PHASEKV_BENCHMARK_OUT=/data/phasekv-results/native_llama_perf_only \
  bash start_benchmark.sh --suites performance --methods full kivi snapkv scope phasekv
```

查看当前条目与完成记录数：

```bash
cat /data/phasekv-results/native_llama_20261029/progress.json
find /data/phasekv-results/native_llama_20261029/records -name '*.json' | wc -l
nvidia-smi
```

统计汇总（不启动 GPU 工作；建议测试退出后运行，避免与自动汇总同时写文件）：

```bash
/data/phasekv-native-venv/bin/python benchmark.py \
  --out /data/phasekv-results/native_llama_20261029 --summarize-only
```

结束标志 `FORMAL BENCHMARK FINISHED` 后查看 ok/oom/error/pending。退出码0为全成功，2表示调度完成但存在失败/未完成；运行异常也会非零退出。不要只看到 FINISHED 就写所有算法成功。

## result 目录与字段

这里服务器目录名为 `results` 或 `/data/phasekv-results`，不是固定叫 `result`。本轮实际结构：

```text
/data/phasekv-results/native_llama_20261029/
├── run_manifest.json                  协议哈希、配置/源码指纹、冻结数据清单
├── pip-freeze.txt                     实际环境依赖
├── gpu_initial.json                   初始 GPU 信息
├── launcher_<时间>_<PID>.log           总日志
├── launcher.pid / latest_log.txt      启动 PID / 最新日志路径
├── progress.json                      当前阶段、方法、样本、repeat
├── jobs/<suite>__<method>.json         子进程作业配置
├── logs/<suite>__<method>.log          各阶段方法日志，续跑追加
├── environment/<suite>__<method>.json  版本、原生类、实际算法配置、dtype
├── records/<method>/<ID哈希>_rN.json   原子结果，是续跑的唯一依据
├── diagnostics/<method>/<ID哈希>.json  性能预热的每步持久KV峰值诊断
├── records.jsonl                      自动导出的全部结果/待运行行
├── summary.csv / summary.json         各任务、长度、方法、split 的统计
├── summary.md                        简短完成状态说明
├── performance_by_prompt.json         每prompt重复中位数/范围，保留失败分母
└── quality_paired_vs_full.json         文档聚类配对bootstrap 95%区间
```

`records/*.json` 保存输出 token、解码文本、官方质量得分、EOS/输出上限状态、输入哈希、方法参数、速度、存储、GPU采样和错误信息。方法間核对 `input_sha256`，不得只按显示名称配对。

指标口径：

- `score`：0—100。QA F1、ROUGE-L、代码相似度、RULER字符串匹配、复合题准确率是不同指标，不能合成一个“总准确率”。
- `score_success_only`：成功项均值；`score_system_failures_zero`：全部预定项作为分母、失败作0的系统分数。pending 未清零时不输出最终系统分数。
- `quality_paired_vs_full.json`：仅成功配对；同一文档不同长度先在文档内平均差值，再以文档为单位重采样。必须同时呈现失败表，不把成功子集偏差隐藏。
- `ttft_s` / `prefill_and_compress_s`：从 prefill 开始到首 token，包含原生压缩初始化。当前无法单独分离二者，不把它伪称纯 prefill。
- `decode_tokens_per_s`：`(output_tokens−1)/decode_s`。计时有CUDA同步边界，固定输出不做每token `.item()`；自然EOS质量生成会检查结束token，其速度不代替固定长度性能。
- `decode_chunks`：每128步以及512/2048/4096等边界的分段计时，可聚合分析长输出退化。
- `prefill_storage` / `final_storage.physical_bytes`：实际底层storage去重容量。PhaseKV 的code/scale/min/hot/空闲页与 `auxiliary_bytes` 单列。
- `persistent_kv_peak_bytes_diagnostic`：**仅**性能预热诊断中逐步采集的持久KV峰值；临时attention矩阵由整进程 allocator 峰值反映。
- `peak_allocated_bytes`、`peak_reserved_bytes`：PyTorch整进程峰值，包含权重与临时张量，不能用它减权重代替KV。
- `gpu_before` / `gpu_after`：nvidia-smi端点采样（总GPU占用、温度、功率、SM时钟），**不是NVML整段峰值采样**。
- `generation_wall_s_including_instrumentation`：含日志/统计开销的wall time，不冒充纯推理端到端延迟。
- LongGenBench 保留官方regex评分；额外给出按Answer编号对齐的准确率、未答数、完成率、末1/3准确率。官方列表式parser可能漏解析/错位，不将补充诊断冒称官方分数。

SCOPE-H2O在长prefill构造完整注意力矩阵；SnapKV/SCOPE会在query-head布局保存KV。这些原生实现特点可能导致24GB卡OOM或实际KV不如名义预算小，需如实报告，不能改成非原生紧凑实现后沿用旧标签。

## 下载结果

测试结束后在服务器：

```bash
tar -czf /data/native_llama_results_20261029.tar.gz \
  -C /data/phasekv-results native_llama_20261029
```

通过现有文件管理器下载 `/data/native_llama_results_20261029.tar.gz`。保留所有 records、环境、清单和日志；不要仅下载 summary.csv。本地建议解压到项目 `results/downloads/20261029_native_llama/`，再由接手 agent 生成独立报告。

## 接手 agent 的剩余工作

1. 上传增量包、启动正式测试；本地 agent 无远程连接，尚未实际启动服务器任务。不要再次征求已授权的安装/启动确认；不要重复安装CUDA环境。
2. 查看第一批性能/质量结果，确认真实输出、压缩触发、停止规则、FP16和实际KV布局。失败按代码/数据/硬件分类；不扩大数值容差、不更换算法来“让它跑过”。
3. 长SCOPE输入OOM是可预期结果，保留并继续。若需要统一更低公共预算，另立实验配置和输出目录，不能悄悄改冻结输入。
4. 本脚本覆盖首轮5456条质量和独立性能；未实现原方案后续Qwen泛化、KIVI-2/SnapKV预算曲线、PhaseKV 100%候选、P5开关A/B、LongGenBench8K。
5. 本轮未额外运行“每类2个独立开发样本”门禁和事前GPU小时估算；部署门禁已完成。可据第一批正式耗时估算剩余时间，但不要承诺一晚跑完，不以测试得分重新选参。
6. 尚缺严格的连续NVML峰值监测、原生prefill与压缩拆分计时、所有方法的有效payload/分配容量全量分解、所有层缓存布局完整导出。当前已有指标不要冒称覆盖这些项。
7. 检查成功配对与失败分母，分开历史复测和未见样本，生成本轮新报告及图表。不要与旧Qwen/BF16/重实现的绝对数值拼排名。
8. README、配置和冻结manifest须随新报告保存。若改代码，记录新哈希并新开输出目录；不覆写已发表的历史报告。

## 本地复现与来源

正式服务器只读取已冻结输入；本地重建数据需要 Transformers tokenizer、numpy、PyYAML、nltk3.9.1、wonderwords2.2.0、tenacity9.0.0，以及下载脚本的html2text/BeautifulSoup。`prepare_benchmark.py` 接受 `--model`、`--history`、`--out`。完整LongBench zip留在本地 `bench_assets/longbench_data.zip`，不重复塞进增量包；本地生成日志在 `bench_assets/prepare.log`。

RULER仓库提交：`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`。文章清单218条全部获取，解析顺序与官方脚本一致；下载transport使用HTTPS/raw GitHub，源码没有改动。解析依赖和文章内容随时间可能变化，已下载原文/解析文本哈希和最终冻结输入为本轮依据。

- [原始方案](../docs/experiments/next_round_plan_20261028.txt)
- [KIVI 官方](https://github.com/jy-yuan/KIVI)
- [SnapKV 官方](https://github.com/FasterDecoding/SnapKV)
- [SCOPE 官方](https://github.com/Linking-ai/SCOPE)
- [RULER 官方](https://github.com/NVIDIA/RULER)
- [LongBench v1 数据](https://huggingface.co/datasets/THUDM/LongBench)

本 README 优先说明当前实际交付状态；旧方案中“尚未实现”的列表属于2026-10-28时点，不能用它覆盖今天已经完成的部署事实，也不能用今天部署成功推断正式质量测试已经完成。
