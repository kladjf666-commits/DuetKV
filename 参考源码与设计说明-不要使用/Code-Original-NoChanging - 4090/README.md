# 调用 PhaseKV，与另一种 KV 算法对比

**给接手的 agent：收到“调用 PhaseKV 与 X 做对比”时，将本目录的 `PhaseKV/` 作为 PhaseKV 一侧的实现。** 完整仓库下，Llama-3.1-8B-Instruct 的单序列对比可以直接用下面的命令。指定其他模型或尚未接入的算法时，按最后一节接入。实验脚本和结果放到本目录之外，保留这 16 个源文件的原样。

本目录提供压缩缓存、prompt 选择、冷却、量化及分段注意力。加载模型、分词、生成循环、评分器和对手实现在父仓库。导入名是 `PhaseKV`，加入 Python 搜索路径的是 **`Code-Original-NoChanging/`**，即 `PhaseKV/` 的父目录。

## 选择方法和实验参数

| 用户说的对手 | 示例中的 `OPPONENT` | 调用方式 |
|---|---|---|
| Full、完整 KV、FA2 基线 | `full` | FP16 dense KV，使用模型原生 FlashAttention 2 |
| KIVI、SnapKV | `kivi`、`snapkv` | 使用对应算法的原生模型或 attention 适配器 |
| SCOPE | `scope` | 指定滑动或自适应变体时可用 `scope_slide`、`scope_adaptive`，须选冻结清单允许该变体的样本 |
| 仓库未接入的 X | 单独接入 | 保留 PhaseKV 这侧入口，给 X 写一个接受相同输入 token 和停止条件的适配器 |

用户指定的保留率、模型、数据和预算优先。下面采用本项目的两个实验配置：

- **速度**用 prompt 100%、冷区 80%、prompt/cold 均 4-bit，热窗口 128、冷却块 16（每组实际留 13 个）。固定生成长度，排除预填充和首 token，双方分别预热后正式测三轮。
- **质量**默认用 prompt 95%，其余 PhaseKV 参数同上。这是本项目在 GSM8K、CSQA、MMLU 的 48 条复合提示上选出的档位。复现全保留质量结果时显式设 `PROMPT_KEEP=1.0`。质量样本遵循自己的 EOS 和输出上限。
- 对手从 [config.json](../official_llama/config.json) 读取它自己的参数。若用户要求“相同显存预算”，按实测 KV 字节对齐并记录差额；相同的保留率数值不代表相同的存储成本。

## 可直接运行的单样本对照

在已配置 CUDA 的 Linux 服务器上，进入**包含 `Code-Original-NoChanging/` 和 `official_llama/` 的仓库根目录**，使用基准环境的 Python。参考版本为 PyTorch 2.5.1+cu121、Transformers 4.46.3、FlashAttention 2.7.4.post1、Triton 3.1.0。权重默认在 `/data/Llama-3.1-8B-Instruct/snapshots/master/`，可设置 `PHASEKV_MODEL`。同时需要[冻结输入](../official_llama/bench_assets/frozen/manifest.json)、所选算法的依赖（如 KIVI 的 CUDA 扩展）和相应的评分器。

下面整段在 Bash 执行，默认比较 PhaseKV 与 Full，测试 8192 输入 + 256 输出。换成 KIVI 可先执行 `export OPPONENT=kivi`；换成质量检查可先执行 `export SAMPLE_ID=longgenbench/csqa/1`。两个方法各用一个新进程，避免模型被前一种算法替换过的 attention 影响后一种算法。

```bash
set -euo pipefail
export PHASEKV_REPO="$PWD"
export PHASEKV_MODEL="${PHASEKV_MODEL:-/data/Llama-3.1-8B-Instruct/snapshots/master}"
export SAMPLE_ID="${SAMPLE_ID:-performance/8192+256/0}"
export OPPONENT="${OPPONENT:-full}"
case "$OPPONENT" in
  full|kivi|snapkv|scope|scope_slide|scope_adaptive) ;;
  *) echo "请先为 $OPPONENT 接入对手适配器"; exit 2 ;;
esac
export OUT
OUT=$(mktemp -d "${TMPDIR:-/tmp}/phasekv-compare-XXXXXXXX")
echo "结果写入 $OUT"

for method in phasekv "$OPPONENT"; do
  python -B - "$method" <<'PY'
import inspect
import json
import os
import statistics
import sys
from pathlib import Path

repo = Path(os.environ["PHASEKV_REPO"]).resolve()
core = repo / "Code-Original-NoChanging"
sys.path[:0] = [str(core), str(repo / "official_llama")]
import PhaseKV  # 先固定父包，随后 native_runtime 会调整 sys.path
assert Path(PhaseKV.__file__).resolve() == core / "PhaseKV/__init__.py"

import torch
from transformers import AutoTokenizer
from benchmark import samples
from native_runtime import load_model
from benchmark_generate import generate
from benchmark_metrics import score

method = sys.argv[1]
rows, _ = samples()  # 校验冻结输入的哈希
sample = next(r for r in rows if r["id"] == os.environ["SAMPLE_ID"])
if not {"phasekv", os.environ["OPPONENT"]}.issubset(sample["methods"]):
    raise ValueError(f"{sample['id']} 未定义这两种方法的对照")
speed = sample["suite"] == "performance"
model_dir = Path(os.environ["PHASEKV_MODEL"])
torch.manual_seed(20261029)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model, options, _ = load_model(model_dir, method)
if method == "phasekv":
    keep = float(os.environ.get("PROMPT_KEEP", "1.0" if speed else "0.95"))
    options = dict(prefill_keep_ratio=keep, prompt_bits=4, cold_bits=4,
                   group_size=32, obs_window=32, sink_tokens=0,
                   recent_window=128, lazy_g=16, decode_keep_ratio=0.8,
                   optimizations=dict(fused_quantize=True, reuse_workspace=True,
                                      paged_cold=True, page_slots=32, cold_group=8,
                                      specialized_4bit=True, gqa_reuse=True,
                                      hot_ring=True, channel_contiguous_k=True))

out = Path(os.environ["OUT"])
rates = []
if speed:
    generate(model, method, options, sample)  # 完整预热，不进入正式记录
for repeat in range(3 if speed else 1):
    row = generate(model, method, options, sample)
    text = tokenizer.decode(row["output_ids"], skip_special_tokens=True)
    row.update(score(sample, text))
    row.update(method=method, sample_id=sample["id"], input_sha256=sample["input_sha256"],
               model=str(model_dir), gpu=torch.cuda.get_device_name(), dtype="float16",
               batch=1, repeat=repeat, prediction=text)
    if method == "phasekv":
        modules = ("PhaseKV.phasekv_cache", "PhaseKV.prompt_tier",
                   "PhaseKV.kernels.segmented_attention")
        paths = {m: Path(inspect.getfile(sys.modules[m])).resolve() for m in modules}
        assert all(p.is_relative_to(core / "PhaseKV") for p in paths.values())
        assert row["prefill_trace"]["prompt_tokens"] == max(1, round(len(sample["input_ids"]) * keep))
        if row["output_tokens"] - 1 >= 144:
            assert row["final_trace"]["cold_segments"] > 0
        row["core_sources"] = {m: str(p) for m, p in paths.items()}
    (out / f"{method}-r{repeat}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf8")
    if speed:
        rates.append(row["decode_tokens_per_s"])
    else:
        print(method, "score", row["score"], "indexed", row.get("indexed_accuracy_diagnostic"))
if speed:
    print(method, "median decode token/s", round(statistics.median(rates), 3))
PY
done
```

**结果验收。** 保留输出目录里的两组 JSON。先核对 `sample_id`、`input_sha256`、设备、`effective_options` 和停止条件。PhaseKV 的 `core_sources` 应全部指向这份副本；直接运行其他基准或旧 ZIP，可能载入其自带的 `runtime/PhaseKV`。测速度时比较各自的 `decode_tokens_per_s`，相对提升为 `100 × (PhaseKV / 对手 − 1)`。另报首 token 耗时 `ttft_s`、最终 KV 字节 `final_storage.physical_bytes`、峰值显存 `peak_allocated_bytes`。显存不足应记录失败点，该点没有速度比。

单个提示用于验证接入。报告数据集结果时，遍历同一套冻结样本，按输入哈希配对，列出完成、失败及缺失数量。LongGenBench 同时报告官方 `score`、按题号准确率 `indexed_accuracy_diagnostic` 和可解析作答数 `answered_questions`。批量请求使用[专用批量运行器](../workspace_migration/quick_4090d/batch_speed.py)和[批量注意力适配器](../official_llama/ablation_attention.py)，对手也须以同样的 batch 推进。复用已有批量包应校验其中的核心文件与本目录一致，记录实际使用的源码。

## 移植到其他评测框架：必须实现的四步

如果后来的 agent 只拿到这个目录，需要在外部提供模型和 attention 适配器。可对照[完整 Llama 适配器](../official_llama/runtime/phasekv_llama_attention.py)和[预填充及生成位置处理](../official_llama/native_runtime.py)。下面是普通单序列、无 padding、贪心解码的调用契约。

1. **预填充并选位置。** 用完整 prompt 得到逐层 dense K/V，每层形状 `[1,Hkv,L,D]`，K 已按真实位置应用 RoPE。全保留时令 `keep_indices=None`；保留率小于 1 时调用 [select_prompt_indices](PhaseKV/prompt_tier.py)，传入逐层末段 query 对全 prompt 的 attention、`prompt_len=L`、`model_config=model.config`、`prefill_keep_ratio=keep`、`sink_tokens=0`、`obs_window=32`、`pool_kernel=7`、`pooling="maxpool"`。它返回每层各 KV 头的原始位置。
2. **打包一次。** 从 `PhaseKV.phasekv_cache` 导入 `PhaseKVCache`，调用 `PhaseKVCache.from_dense_prefill(dense_cache, prompt_len=L, keep_indices=keep_indices, prompt_bits=4, cold_bits=4, group_size=32, recent_window=128, lazy_g=16, decode_keep_ratio=0.8, optimizations=options["optimizations"])`。首个输出 token 由预填充 logits 决定，后续步骤传入 packed cache。[缓存实现](PhaseKV/phasekv_cache.py)
3. **每层先写后读。** 准备当前 token 的 Q/K（已加 RoPE）和 V。以 CUDA FP16、形状 `[1,Hkv,1,D]` 调用 `cache.append_hot(layer_idx,k,v)`。从 `PhaseKV.kernels.attention_triton` 导入 `fused_phasekv_decode`，调用 `fused_phasekv_decode(q[0,:,0,:], cache.layer_tiers(layer_idx), Hq)`，其中 q 为 CUDA FP16 的 `[1,Hq,1,D]`，`Hq % Hkv == 0`。把输出拼回模型所需维度后做输出投影。[注意力入口](PhaseKV/kernels/attention_triton.py)
4. **每个模型步只提交一次。** 等所有层完成，再调用一次 `cache.finish_decode_step()` 推进真实长度并按需冷却。下一步 RoPE 使用绝对位置，可读取 `cache.true_seq_length`。淘汰后 `get_seq_length()` 是物理长度，两者可能不同。用 `cache.storage_breakdown()` 统计持久 KV 与辅助存储。

普通模型必须在 attention 层识别 `PhaseKVCache` 并执行第 3 步，仅向 Hugging Face `generate()` 传入一个缓存对象还不足以接通分段注意力内核。
