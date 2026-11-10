# 按模型组织的 KV 算法接入

## 目录与职责

```text
models/
  Qwen2-7B-Instruct/        # 明确的模型版本：profile、architecture、compression、run
  Qwen2.5-7B-Instruct/      # 独立版本，不能混作同一个实验
  Qwen2-1.5B-Instruct/      # 已有小模型的同一套新入口
  Llama-2-7b-chat-hf/       # 约6.74B，MHA，独立模型配置
  Llama-3.1-8B-Instruct/    # 约8.03B，GQA，独立模型配置
  Qwen2/                   # Qwen2/Qwen2.5 共用架构适配：PhaseKV、KIVI、运行调度
  Llama/                   # Llama六方法调度、原生投影/RoPE、模板适配
  packed_runtime.py        # 家族共用真实打包生成循环
  OPT/                     # 已有 OPT 的 PhaseKV 适配
  registry.py              # 显式模型/适配器注册
  requirements.txt         # 租卡环境依赖
```

根目录的 `PhaseKV/`、`KIVI/`、`SCOPE/`、`SnapKV/`、`PyramidKV/` 保存与模型无关的算法。模型投影、RoPE、注意力包装、模型运行入口统一在上述模型目录。旧 `qwen2/`、`PhaseKV/qwen2/`、`KIVI/qwen2_attention.py` 等只转发，不复制算法或模型实现。旧接口与新接口指向同一模块对象，保留 monkeypatch 与类身份兼容。

## 当前接入范围

| 模型 | full | KIVI | SnapKV | SCOPE | PyramidKV | PhaseKV |
|---|---|---|---|---|---|---|
| Qwen2-7B-Instruct | 已接入 | 已接入 | 已接入 | ALLKV+Slide适配 | 已接入 | 已接入 |
| Qwen2.5-7B-Instruct | 已接入 | 已接入 | 已接入 | ALLKV+Slide适配 | 已接入 | 已接入 |
| Qwen2-1.5B-Instruct | 已接入 | 已接入 | 已接入 | ALLKV+Slide适配 | 已接入 | 已接入 |
| Llama-2-7b-chat-hf | 已接入 | 已接入 | 已接入 | ALLKV+Slide适配 | 已接入 | 已接入 |
| Llama-3.1-8B-Instruct | 已接入 | 已接入 | 已接入 | ALLKV+Slide适配 | 已接入 | 已接入 |
| OPT | 旧入口保留 | 尚未接入统一比较 | 尚未接入统一比较 | 尚未接入统一比较 | 尚未接入统一比较 | 旧适配保留 |

“已接入”指代码路由和配置完成，不表示已经跑过完整7B权重。按用户安排，本次不下载7B权重、不执行7B推理或评分；模型结构/参数检查与CPU回归检查已执行。完整模型、28:4 GQA的GPU内核数值对照和长输入显存峰值，需在租用GPU上验证。

两种7B的官方结构都采用28层、hidden size 3584、28个query头、4个KV头、head dim 128：[Qwen2官方配置](https://huggingface.co/Qwen/Qwen2-7B-Instruct/raw/main/config.json)、[Qwen2.5官方配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json)。结构相同不代表权重或模型身份相同。提供本地权重路径时会核对结构，实际来源路径也会记录；不根据文件夹名推断权重的发布者身份。

## 租卡运行

以下命令均在项目根目录执行。选择与本环境兼容的CUDA镜像，验证版本为 PyTorch 2.5.1+cu121、Triton 3.1.0、Transformers 4.46.3；可用于3090/4090类GPU。其他架构或更新软件栈需要重新验证，不能直接宣称兼容。

```bash
python -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r models/requirements.txt
```

已有环境可直接使用，无须覆盖重装。准备同一份 LongBench 数据，目录为 `<数据目录>/data/narrativeqa.jsonl` 等五个任务。当前机器已有 `.cache/longbench`，迁移项目时要同时迁移该被Git忽略的数据目录，或使用 `scripts/download_data.py` 准备数据。

以 Qwen2-7B-Instruct 为例，已下载权重时：

```bash
.venv/bin/python models/Qwen2-7B-Instruct/run.py --mode check --model-path /path/to/Qwen2-7B-Instruct
.venv/bin/python models/Qwen2-7B-Instruct/run.py --mode smoke --model-path /path/to/Qwen2-7B-Instruct --fixed-output --max-new-tokens 80
.venv/bin/python models/Qwen2-7B-Instruct/run.py --mode longbench --model-path /path/to/Qwen2-7B-Instruct --longbench-dir .cache/longbench
```

尚未下载权重时，用 `--allow-download` 代替 `--model-path ...`，由官方模型ID下载至Hugging Face缓存。`check`只检查部署条件，不下载模型；不满足条件时退出码为2。`smoke`和`longbench`才加载权重。Qwen2.5-7B对应入口为 `models/Qwen2.5-7B-Instruct/run.py`。

默认每次运行输出到 `results/<完整模型名>/<模式_时间>` 文件前缀（`.json`、`.md`、`.predictions.jsonl`），也可用 `--out` 指定新前缀。已有结果不覆盖，不会把7B数据写进1.5B旧报告。暂未实现断点续跑；失败保留已完成样本，重新运行请使用新前缀。

## 测试口径

- `check`：配置、环境、权重存在性和最低显存检查。Qwen7B的BF16权重约14.18 GiB；显存检查只是最低门槛，未涵盖全部eager注意力矩阵，不保证长输入不OOM。
- `smoke --fixed-output`：六条路径均使用同一输入，忽略EOS生成指定长度，用于检查触发记录；不计算质量分数。80输出最多79次追加，默认SCOPE预算32可触发，PhaseKV可跨48步冷却阈值。
- `longbench`：默认5任务×32条，正文截断1536、batch=1、BF16、贪心，正常遇EOS停止；每种方法使用同一批token化输入。输出上限按原任务设置。
- `full`为完整BF16缓存。KIVI与PhaseKV走真实打包路径，禁止伪量化入口；SnapKV、SCOPE、PyramidKV通过淘汰token压缩，KV精度仍为BF16。
- 默认参数保存在各模型的 `compression.json`：prompt观察窗口统一32；KIVI残留长度64；SCOPE近期窗口16、生成预算32。各窗口含义不同，本组默认配置不构成相同内存预算对比，也与旧报告的SCOPE预算256有区别。
- SCOPE保留完整prompt，只实现ALLKV+Slide的GQA组内分数聚合，未实现Adaptive/Discontinuous和压缩prompt插件。
- 所有输出附带实际触发记录、物理KV字节、模型版本/路径、输入哈希、数据哈希和代码哈希。速度为包含prefill和日志的端到端速度，不是纯decode速度。

可通过 `--methods full,kivi` 筛选方法，通过 `--compression-config path.json` 使用另一组显式参数，通过 `--tasks`、`--n-samples`、`--max-length`、`--max-new-tokens` 调整实验。修改输入长度或输出上限会成为新的实验配置；先进行少量样本测试。

## Llama 7B/8B部署补充

- 精确7B档选 [Llama-2-7b-chat-hf](Llama-2-7b-chat-hf/README.md)；8B档选 [Llama-3.1-8B-Instruct](Llama-3.1-8B-Instruct/README.md)。两者均接入六方法，但不混作同一模型的实验。
- 对应配置使用Llama无QKV偏置的参数量计算与原生RoPE，加载前核对结构；完整基线/淘汰KV为BF16，打包方法的残留/热缓存为FP16，与当前Qwen打包实现一致。
- Llama输入只应用一次原生chat模板，避免重复BOS；全部EOS结束标记同时生效。Qwen历史输入编码和停止口径保留。结果记录原生输入哈希、实际长度、EOS和代码版本，因此跨模型不能假设token序列相同。
- Llama权重存在访问门槛，租卡时使用已获访问权限的Hugging Face账号或完整本地原始权重；GGUF、GPTQ/AWQ等量化权重不在本比较范围。
- Llama-2/3.1 BF16权重本身分别约12.55/14.96 GiB；eager注意力和压缩准备仍需额外显存，输入长度限制以实测为准。
- CPU验证使用随机小模型及meta结构，不是预训练质量评测。租卡后执行 `python -m unittest tests.test_llama_deployment.TestLlamaGPU -v`，再执行对应模型smoke和LongBench；GPU检查被跳过不代表通过。

## 后续增加模型

1. 建立 `models/<完整模型名>/`，增加 `profile.json`、`architecture.json`、`compression.json` 和 `run.py`。
2. 同属Qwen2架构的模型可复用 `models/Qwen2/integration.py`，但必须提供准确结构并验证GQA/位宽约束。
3. 新架构在 `models/<架构名>/` 实现适配和 `generate()`，通过 `models/registry.py` 的 `FAMILIES` 显式注册。不得直接把未完成适配的新架构标成六算法均支持。
4. 对新架构完成原前向透明性、绝对位置、真实打包数值、淘汰/迁移边界验证后，再运行正式比较。跨模型结果保存在不同模型名目录。
