# PhaseKV：Linux / 单张 4090D 对比测试

入口是项目根目录的 `run_4090d.sh`，实现是 `scripts/benchmark_4090d.py`。默认测试 **Qwen2.5-7B-Instruct + Llama-2-7b-chat-hf**，顺序执行 full、KIVI、SnapKV、SCOPE、PyramidKV、PhaseKV。一次只加载一个模型、一个算法的运行进程。

## 上传与启动

你已经上传过项目时，补传这两个新文件，保持相同目录结构；这份说明可以一同上传。脚本还需要远程项目中已有的 `PhaseKV/`、`KIVI/`、`SnapKV/`、`SCOPE/`、`PyramidKV/` 和 `models/` 目录。启动时会检查它们。

本地检查发现 `eval/`、`baselines/` 已缺失。本入口使用独立生成循环和进程内的小型接口适配，直接调用已有算法，不依赖这些缺失目录，也不恢复或覆盖它们。

在 Linux 容器项目根目录执行，替换两个权重路径：

```bash
bash run_4090d.sh --setup \
  --qwen /实际路径/Qwen2.5-7B-Instruct \
  --llama /实际路径/Llama-2-7b-chat-hf
```

`--setup` 必须放在第一个参数。它使用 Python 3.10–3.12，在项目中创建独立 `.venv-4090d`，安装 PyTorch 2.5.1 CUDA 12.1 / Triton 3.1.0 及固定版本依赖，不替换容器原有环境。首次需要联网下载依赖。容器的 NVIDIA 驱动必须支持 CUDA 12.1。原环境已符合版本时，可省略 `--setup`；使用 `PYTHON=/path/to/python` 指定解释器。

如果没有本地权重路径，可以使用默认官方模型 ID：

```bash
bash run_4090d.sh --setup
```

这种方式会下载两个模型的原始 safetensors 权重。Llama 官方权重需要已获得授权的 Hugging Face 账号；登录该账号，或由容器环境提供 `HF_TOKEN`。不要把 token 写进脚本。脚本固定远程快照，并为本地权重和配置计算 SHA256；不接受 GPTQ/AWQ/GGUF 权重量化文件。

如果你用的是 **Qwen2-7B**，必须明确指定匹配的配置：

```bash
bash run_4090d.sh --setup \
  --qwen-profile Qwen2-7B-Instruct \
  --qwen /实际路径/Qwen2-7B-Instruct \
  --llama /实际路径/Llama-2-7b-chat-hf
```

## 默认测试内容

| 测试 | 默认设置 | 能回答的问题 |
|---|---|---|
| GPU 数值检查 | 随机小模型使用对应 7B 的 query/KV 头数和 head dimension；对照 eager 原模型、分块 prefill、真实打包解码、缓存驱逐，跨 80 步维护边界 | 新运行循环和模型适配是否通过基本数值验证 |
| 显存容量 | 每方法从 1024 输入开始，翻倍增长，遇 OOM 或达到显存目标后按 256 token 精度二分；固定生成 128 token | 当前实现和模型上下文范围内的可运行长度 |
| 长上下文检索 | 1K、2K、3K、4K、8K、16K、24K、32000 输入；关键信息放在 10%/50%/90% 位置，各 3 个随机口令 | 压缩是否丢失不同位置的信息 |
| 输入长度测速 | 上述输入长度，固定输出 256 token | 长输入下的 prefill、decode 速度和真实 KV 字节 |
| 输出长度测速 | 输入 1024，输出 256/1024/2048/3072/4096 token | decode 热层/冷层长期增长、淘汰和量化开销 |
| 长输入加长输出 | 8192+1024、16384+2048、24576+4096 | 同时存在长上下文和长生成时的稳定性 |
| 真实任务质量 | 8 个 LongBench v1 子任务，每任务固定随机抽样 32 条，在 1536/3072/8192/16384 输入预算下评测 | 相同输入下的问答和摘要质量损失 |

每种测速形状先完整预热一次，再测 3 次，报告中位数。真实任务正常遇 EOS 停止；测速强制生成规定长度，忽略 EOS，**这些强制输出不用于质量评分**。检索测试是脚本自建的 `custom_passkey`，不是官方 RULER。

8 个真实任务包括 narrativeqa、qasper、multifieldqa_en、hotpotqa、2wikimqa，以及 gov_report、qmsum、multi_news。问答用 QA F1，摘要用 ROUGE-L；摘要输出上限 512 token，能更充分覆盖 decode 压缩，但是否实际触发仍由日志记录。使用官方 [LongBench v1 数据和提示词](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)，数据无需训练或人工标注。

默认 32 条/任务属于覆盖较广的初步实验。需要完整子任务样本时使用 `--samples 0 --out results/4090d_all_samples`。默认套件也可能运行很多小时；实际时间取决于算法、输出速度与可运行长度，无法在本地预先保证。

## 24GB 显存与上下文边界

默认目标是 **PyTorch 实际 allocated 峰值占显存的 92%**，约为标称 24GiB 的 22GiB。可以在首次运行时传 `--target-vram 0.95`。脚本还为显卡上已有的外部占用留出空间。显存不足记录为 `oom`，释放当前案例的缓存，容量探测缩小长度后继续。

这不是任意长度都能成功的保证。容量探测用 128 输出；真实长输出或文本仍可能 OOM，会单独记录。正式质量对比只汇总所有选定方法均成功、输入哈希相同的样本，不把 OOM 当作零分，也不偷偷给某一方法缩短问题。

**不能保证每种方法都占满 24GB。** [Llama-2 7B 的原生上下文为 4096](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)，脚本约束输入与输出之和，输入 1024 时最长可测 3072 输出。当前 Qwen 配置是 32768；不会擅自扩展 RoPE。达到模型上下文上限后，即使显存尚有余量也会停止增长，不创建无用张量填显存。

所有方法 batch=1、BF16 原始权重；KIVI/PhaseKV 使用现有真实 uint8 打包缓存和 Triton 内核，其残留/热缓存按原实现为 FP16。其他方法的 KV 为 BF16。prefill 先保留完整 dense KV，再压缩，因此容量上限仍可能受压缩前的缓存和临时空间限制。

新循环采用 SDPA 分块 prefill，仅最后一个观察窗口请求注意力分数；原位置编码保持绝对 token 位置。这样避免存储所有层的 N×N 分数矩阵。它与旧的 eager 评测协议不同，**不能把两套结果混合**。启动时 GPU 数值检查必须通过；通过小模型检查仍不代表完整 7B 长上下文质量已经验证。

校验 v3 保留 v2 的 FP32 分块语义检查和 BF16 同路径适配器检查，并修正 decode 的参考路径。full/SnapKV/SCOPE/PyramidKV 使用相同 SDPA 模块、相同 `output_attentions` 设置和独立缓存副本，要求 logits、K/V 和 SCOPE 注意力分数逐元素相等。KIVI/PhaseKV 按每层相同输入分别检查 Q/RoPE、追加 K/V、实际 Triton 内核与 FP32 稠密参考、输出投影；反量化 KV 保持 FP16，不额外转 BF16。内核对照采用已有 `tests/test_kivi_packed.py` 的 `atol=.002, rtol=.01`，其余同路径检查要求完全相等。80 步仍覆盖冷却、残留刷新和淘汰边界。所有临时检查只存在于校验子进程，不改变正式测量的计算路径。尚需在租用 GPU 上执行确认。

如果此前出现第一个 prefill 对照失败或 `decode_vs_materialized_cache / full / step 4`，请上传新版 `scripts/benchmark_4090d.py`。旧运行已保存 manifest，更新代码后必须换输出目录（例如 `results/4090d_qwen_v3`），不要强制复用旧协议。此前模型下载已完成、缓存还在时，可用 `HF_HUB_OFFLINE=1 bash run_4090d.sh --models qwen --out results/4090d_qwen_v3` 从原缓存继续准备，不重新下载权重。v2 未对齐 decode 的 SDPA/eager 路径，不能将它的跨路径 BF16 logits 失败直接归因为压缩错误。

## 数据下载

安装旧版启动脚本时，如果出现 `flit_core` 找不到、日志只显示 PyTorch 专用源，说明普通构建依赖也被限制到了该源。新版启动脚本已改为 PyPI 加 PyTorch CUDA 源，并明确固定 `torch==2.5.1+cu121`。远程旧脚本无需重新上传也可恢复：

```bash
PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_PREFER_BINARY=1 \
  bash run_4090d.sh --setup --models qwen --out results/4090d_qwen
```

这是安装阶段失败，尚未进入正式测试，不需要 `--resume`。

如果随后出现 `files.pythonhosted.org ... Read timed out`，是软件下载超时。可在项目根目录使用清华 PyPI 镜像，升级 pip 后重新安装 CUDA PyTorch；接着用 `--install-deps` 继续，跳过旧启动脚本的 PyTorch 安装步骤：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=300
export PIP_RETRIES=10
export PIP_RESUME_RETRIES=10
export PIP_PREFER_BINARY=1

.venv-4090d/bin/python -m pip install --upgrade pip && \
.venv-4090d/bin/python -m pip install 'torch==2.5.1+cu121' \
  --extra-index-url https://download.pytorch.org/whl/cu121 && \
bash run_4090d.sh --install-deps --models qwen --out results/4090d_qwen
```

这些设置只涉及 pip 软件依赖下载，不会切换 Hugging Face 模型和数据下载地址。新版启动脚本默认增加超时和重试，并遵循 `PIP_INDEX_URL`；遇网络长期不可达仍可能失败。

正式启动会下载并校验 LongBench；也可以在尚未租卡或 GPU 空闲时先下载：

```bash
python3 scripts/benchmark_4090d.py --download-only
```

默认位置是 `.cache/longbench/data/<任务名>.jsonl`。已有数据可以通过 `--data-dir /path/to/longbench` 指定，该目录还需有官方 `dataset2prompt.json`，缺失时会自动下载。

如果容器无法直连 Hugging Face，可自行配置可用的 `HF_ENDPOINT`；数据脚本遵循该变量。提示词来自 GitHub raw；该地址不可达时，在能联网的电脑从[官方提示词文件](https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json)下载到数据目录后再运行。不要在首次下载前设置 `HF_HUB_OFFLINE=1`。

## 结果与断点续跑

默认输出：`results/4090d_suite/`。

- `summary.md`：可直接阅读的质量、速度对比。
- `summary.json`：结构化指标和相对 full 的逐题配对 bootstrap 95% 区间。
- `records/*.jsonl`：逐题预测、答案、输入哈希、输出 token、prefill/压缩时间、decode 时间、真实 KV 字节、allocated/reserved 峰值和实际 decode 压缩触发次数。
- `records/*.capacity.json`：达到显存目标前的长度和已成功完成的最大探测长度。
- `records/*.environment.*.json`：每次进程启动时的软件、模型配置、显卡和可用显存。
- `manifest.json`：代码、数据、权重哈希和完整实验设置。
- `gpu_validation_*.json`：GPU 数值检查通过记录。
- `logs/*.log`、`run_state.json`：运行日志及任务完成/错误状态。

按 Ctrl-C 可停止。再次运行**相同参数**并追加 `--resume`，会跳过已经记录的案例：

```bash
bash run_4090d.sh \
  --qwen /实际路径/Qwen2.5-7B-Instruct \
  --llama /实际路径/Llama-2-7b-chat-hf \
  --resume
```

如果第一次用过 `--target-vram`、`--samples`、`--out` 等选项，续跑也需带上。修改代码、模型、数据或参数要换一个 `--out`；不会混入旧结果。已记录的 OOM/上下文跳过也会保留。电源故障或强制杀进程可能留下 `RUNNING.lock`；先确认其中 PID 的进程已退出，再删除锁续跑。

不运行模型、只重新汇总已有结果：

```bash
python3 scripts/benchmark_4090d.py --report-only --out results/4090d_suite
```

长时间运行建议放在 `tmux` 中，避免 SSH 断开导致中断。每条完成的案例都会落盘。

## 参数与结果解释

仅测 Qwen：`--models qwen`。仅测部分方法：`--methods full,kivi,phasekv`。仅跑压力测试：`--phases capacity,performance`。查看计划而不下载或用 GPU：`--dry-run`。

算法参数来自各模型的 `compression.json`。默认是不同机制的显式配置对比，**不是等物理 KV 内存比较**；要证明相同内存下的优势，应继续扫描位宽/保留比例并比较质量—实际 KV 字节曲线。可用 `--compression-config /path/to/config.json` 指定含 `methods` 列表的配置；新输出目录会保存其内容。该入口固定已覆盖数值检查的维护窗口：KIVI 64、PhaseKV 热窗口32/懒批16、SCOPE 最近16/生成预算32；可调整位宽及保留比例。

SCOPE 是本仓库 ALLKV+Slide 适配；PyramidKV/KIVI 等也使用本仓库实现和内核，不能称为各论文官方实现的速度复现。计时包含同步、有限值检查、生成循环和缓存维护；质量运行的自然输出长度不同，不用于直接比较吞吐。`capacity_specific` 行的输入长度可能因方法而异，也不能直接比较速度。

本地验证命令：

```bash
python3 -m unittest discover -s tests -p test_benchmark_4090d_standalone.py -v
```

开发机完成的是无 GPU 的控制流程/数据/报告测试和语法检查。实际 CUDA 内核检查和完整模型测试由租卡启动流程执行；遇数值错误会停止，不生成虚假的通过结果。
