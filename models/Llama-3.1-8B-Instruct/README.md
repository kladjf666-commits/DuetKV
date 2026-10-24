# Llama-3.1-8B-Instruct

约8.03B参数，32个query头/8个KV头（GQA），保留原生Llama-3.1 RoPE缩放。本目录包含模型配置、六方法参数与运行入口。权重ID为 `meta-llama/Llama-3.1-8B-Instruct`；本次仅完成代码接入，未下载或运行完整权重。

## 租卡运行

在项目根目录执行，环境安装步骤见[模型部署总说明](../README.md)。使用完整的原始Hugging Face权重目录，包含 `config.json`、`generation_config.json`、原生tokenizer/chat模板与safetensors分片。

```bash
.venv/bin/python models/Llama-3.1-8B-Instruct/run.py --mode check --model-path /path/to/Llama-3.1-8B-Instruct
.venv/bin/python models/Llama-3.1-8B-Instruct/run.py --mode smoke --model-path /path/to/Llama-3.1-8B-Instruct --fixed-output --max-new-tokens 80
.venv/bin/python models/Llama-3.1-8B-Instruct/run.py --mode longbench --model-path /path/to/Llama-3.1-8B-Instruct --longbench-dir .cache/longbench
```

也可以使用 `--allow-download` 代替本地权重路径。官方仓库需要具备访问权限并事先在租卡环境登录Hugging Face；代码不替你申请权限或接受许可。`check`不下载权重。

结果默认写入 `results/Llama-3.1-8B-Instruct/`，保留原有Qwen报告。默认采用与Qwen配置相同的五任务×32条数据、正文1536 token预算、BF16权重、batch=1、eager和贪心解码；不同模型的分词与模板不同，实际token长度会单独记录。

## 算法与验证范围

- 六方法：full、KIVI、SnapKV、SCOPE、PyramidKV、PhaseKV。
- KIVI和PhaseKV读取真实uint8打包载荷，无完整浮点KV副本；残留/热缓存为FP16，完整基线及淘汰方法为BF16。权重不做量化。
- SCOPE为ALLKV+Slide，支持MHA和GQA分数聚合；未实现Adaptive/Discontinuous。
- 各方法保留自身参数语义，默认配置不是等物理内存预算。压缩参数完整记录于 `compression.json` 和结果报告。
- 检查层数、头数、RoPE、偏置和 `pretraining_tp=1`；使用模型全部EOS结束标记，避免续写额外轮次。`--fixed-output`只用于功能检查。
- `architecture.json`是期望结构，加载时与真实checkpoint核对；本次浏览工具无法读取官方受限仓库的原始配置，不能将其视作下载出的配置副本。官方来源：[模型卡](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)、[配置入口](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json)。

CPU检查已覆盖随机小模型的原前向透明性、MHA/GQA缓存淘汰、真实打包、EOS、模板和参数量。租卡后先执行数值验证，再执行上述smoke和正式评测：

```bash
.venv/bin/python -m unittest tests.test_llama_deployment.TestLlamaGPU -v
```

GPU检查必须实际运行且通过；显示skip不代表通过。该检查为小模型内核数值对照，完整7B/8B模型、实际长输入与吞吐仍需另行验证。
