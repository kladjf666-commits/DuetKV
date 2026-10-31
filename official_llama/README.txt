Llama-3.1-8B-Instruct / FP16 / 原生压缩算法部署
日期：2026-10-28

1. 本次交付及边界

已从 GitHub 克隆 KIVI、SnapKV、SCOPE，放在 vendor/，并固定到既定方案的提交。
runtime/ 是运行副本；兼容修改在 patches/ 中逐行保存；sources.json 保存源码 SHA256。
PhaseKV 使用当前工程 P5 实现的独立快照，避免以后主工程修改改变本轮实验。
Full 使用 Transformers 原生 Llama + FlashAttention2。
本地已做 CPU 接口/数学语义验证；本机没有 CUDA，尚未宣称 CUDA 编译、4090D 数值验证、8B 模型运行成功。
这一包完成原生代码接入和部署入口，不包含方案中的 5456 条正式质量评测。
所有正式结果必须等服务器验证通过后产生，不能把随机小模型/合成吞吐当作准确率。

2. 固定来源

KIVI   https://github.com/jy-yuan/KIVI
       876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
SnapKV https://github.com/FasterDecoding/SnapKV
       e216ddc84c5bd210378cbdbbba12ba02102aa640
SCOPE  https://github.com/Linking-ai/SCOPE
       8cbae48042e34da630c7a044f535de7a991c2999

vendor/ 为 GitHub 原始源码，不是此前工程里同名的本地改写版本。
本地目录包含各自 .git；上传包为固定提交的文件快照，省略 .git，无需服务器再访问 GitHub。
上游自带的 data/results 等也保留在源码快照中，仅作来源资料，runner 不会把它们当作本轮数据。
算法许可证保留在各仓库内。下载、使用 Llama 权重仍按权重发布方授权条款执行。

3. 主配置（详见 config.json）

full          原生 Full KV，FlashAttention2
kivi          官方 K/V 4-bit，group32，residual128，use_flash=True
snapkv        官方容量2048（含window32），maxpool，kernel7；不额外压缩生成KV
scope         官方 H2O prefill，容量2048，近期8；discontinuous，decode预算512，近期256
scope_slide   同上，官方 slide 调度
scope_adaptive 同上，官方 adaptive 调度
phasekv       prompt保留70%，4-bit；hot128，cold4-bit，decode保留100%，lazy_g16，group32
              P5：fused_quantize/reuse_workspace/paged_cold/specialized_4bit/hot_ring=true
              page_slots32，cold_group8，gqa_reuse=false

全部权重、未压缩KV及量化比例/偏移存储为 FP16；量化 codes 是整数。
FP16 不意味着禁用原算法中的 FP32 softmax/归约累加，这些计算按作者实现保留。
不加载4/8-bit模型权重、不启用BF16、CPU offload或伪量化替代路径；batch=1，无padding。

4. Linux 服务器使用

将 delivery/PhaseKV_native_llama_20261028.zip 上传到 /root/KV/PhaseKV。
在服务器运行：

    cd /root/KV/PhaseKV
    unzip PhaseKV_native_llama_20261028.zip
    cd official_llama
    bash setup_linux.sh

安装使用独立 official_llama/.venv，不修改原有 .venv-4090d。
要求 Python3.10-3.12（推荐3.10）、可用NVIDIA驱动和CUDA开发工具nvcc；只有nvidia-smi不等于具备编译工具。
使用 PyTorch2.5.1+cu121、Transformers4.46.3、FlashAttention2.7.4.post1。
建议 CUDA12.1 development 镜像；驱动可更新，编译器/toolkit必须兼容PyTorch CUDA12.x。
默认编译目标为4090D的sm89，MAX_JOBS默认4；编译内存不足可以 MAX_JOBS=2 bash setup_linux.sh。
不运行上游旧requirements.txt，避免它们覆盖统一环境。
所有依赖从发布源安装；这一步需要网络。模型运行完全离线，不会再次下载模型。

权重目录必须含全部4个safetensors分片、索引、config和tokenizer文件。original/不需要。
先检查，例如（替换成服务器上实际路径）：

    .venv/bin/python check_model.py /root/models/Meta-Llama-3.1-8B-Instruct

然后依次验收原生CUDA算子、小型Llama3-RoPE、真实8B五算法及SCOPE附加调度：

    bash validate_linux.sh /root/models/Meta-Llama-3.1-8B-Instruct

默认检查2304输入 + 1024输出，强制输出长度，以实际经过量化、冷热区、SCOPE淘汰。
每个方法独立子进程；真实8B先预热1次，再测1次。此处是部署验收，不能直接作为论文速度表。
目录为 results/validation_时间/，包含 kernel_checks.json、tiny/、llama31_8b/。
校验失败就停止后续阶段；不会自动提高误差阈值、切换本地替代实现或偷偷减少模型/上下文长度。
单个worker失败会记录状态并继续尝试其他方法，check_run最终将整轮判为未通过。
KIVI/PhaseKV内核校验基于相同压缩KV的解包参考，不要求有损量化结果等于Full。
CUDA数值检查固定atol=0.005、rtol=0.01；失败需要定位原因，不应直接放宽。

终端可能关闭时，可用：

    nohup bash validate_linux.sh /root/models/Meta-Llama-3.1-8B-Instruct > validation.log 2>&1 &
    tail -f validation.log

5. 验收通过后的合成速度测试

短输入长输出（统一输入ID，预热1次，正式3次）：

    .venv/bin/python run.py --model /root/models/Meta-Llama-3.1-8B-Instruct --input-tokens 1024 --new-tokens 4096 --repeats 3 --out results/shortin_longout

此命令默认主表五组full/kivi/snapkv/scope/phasekv。
SCOPE原生在短于prompt预算时不压缩prefill，也仍使用2048作为保护边界；本包保留这个上游行为。
因此1024输入属于系统行为测试，不能声称SCOPE在此严格执行“真实prompt + 512生成KV”的固定边界。
正式SCOPE论文配置的主要长输出组应同时包含输入>=2048的情况。

长输入短输出：

    .venv/bin/python run.py --model /root/models/Meta-Llama-3.1-8B-Instruct --input-tokens 8192 --new-tokens 256 --repeats 3 --out results/longin_shortout

请注意：SCOPE官方H2O会构造全量QK矩阵，8192/16384输入可能在24GB上OOM。
SnapKV/SCOPE原生按查询头重复KV后存储，实际容量高于保持8个KV头的改写版本。
这两项原生行为被保留。OOM是结果，不应换成别的算法后沿用同一名字。

6. 兼容层究竟改了什么

SnapKV：将旧RoPE seq_len接口换为真实position_ids/外部position_embeddings，支持Llama3.1 RoPE。
        恢复无padding路径的FlashAttention调用包装，核心仍调用flash_attn_func。
SCOPE：Transformers4.46将mask工具函数移入LlamaModel，调整导入；H2O打分/选择/调度未修改。
KIVI：仅修复尚无量化V的短输入分支GQA头数不匹配，采用相邻残留分支同样的repeat_kv。
       量化/解包/CUDA矩阵计算源码完全不变；长输入已量化分支不变。
PhaseKV：包初始化不再加载历史fake-quant注册器；真实packed cache/kernel/selector原样快照。
         原生Llama packed wrapper接在HF FA2上，prefill最后32个query用HF eager输出选择分数。
         保留因果mask/真实RoPE位置；CPU验证其分块预填充与完整预填充结果一致。

生成层统一用显式绝对位置和greedy循环，每步仅投影最后一个hidden state。
这是模型接口与生成驱动适配；不调用上游旧generate准备函数，也不改算法的attention/cache逻辑。
Full及各方法的prefix处理成本如实计入prefill_and_compress_s；不把PhaseKV获取分数说成零成本。
SnapKV/SCOPE每个新样本重置kv_seq_len/cluster，SCOPE三个类级计数器全部重置。

SCOPE delta=(输出上限-256)//(512-256)。4096输出时delta=15。
对于adaptive/discontinuous，输出上限<512导致delta<=0，本包显式使用None解码策略，记录
scope_mode=prefill_only_short_output_boundary；保持prefill压缩，生成KV不裁剪。
这是短输出边界处理，不冒充论文discontinuous结果；512以上沿用官方公式。
官方delta取整使实际动态预算不一定严格等于512；delta=1时adaptive/discontinuous甚至不会淘汰。
因此验收输出设为1024以触发真实淘汰；保留上游调度，并报告实际容量，不自行改公式凑预算。

7. 数据和结果口径

每轮先冻结输入ID；所有方法使用同一输入、输出上限、种子和权重。
run_manifest.json 保存提交、源码清单摘要、驱动摘要、模型JSON摘要及分片大小。
每个method.jsonl记录：prefill+压缩时间、decode时间/速度、实际KV存储、辅助工作区、峰值显存、输出ID。
实际KV按唯一storage计数，包含完整预留页/热区、scale/min等元数据；辅助工作区单列。
cached_generated_tokens=output_tokens-1：最后一个采样token尚未进入下一次模型前向，此口径所有算法相同。
decode token/s的分子也为output_tokens-1，首token成本计入prefill。
每128步有进度日志；计时区间排除该进度输出。不得把GPU功率变化单独当成卡住证据。

--inputs可接冻结JSONL，每行：
{"id":"sample_001","input_ids":[1,2,3],"max_new_tokens":512,"stop_ids":[128001,128009],"natural_eos":true}
示例ID只是格式说明，真实输入必须来自该模型官方tokenizer/chat模板。
--prompt-file 支持UTF-8文本，自动使用官方chat模板；配合 --natural-eos 用于人工输出检查。
输入文件不自动评分；LongBench/RULER/LongGenBench数据准备与官方评分器接入属于下一阶段。
既有Qwen/BF16/改写基线历史结果不能拼入本轮原生Llama/FP16速度排名。

8. 目录

vendor/          三个GitHub原仓库
runtime/         兼容运行副本及PhaseKV源码快照
patches/         相对于原始文件的最小补丁
sources.json     每个文件的来源/哈希
config.json      论文配置与PhaseKV候选参数
setup_linux.sh   隔离环境和原生CUDA扩展安装
validate_linux.sh CUDA内核、小模型、真实8B验证
run.py           独立子进程运行器
native_runtime.py 接口适配层
check_model.py   离线分片完整性检查
check_run.py     部署结果和压缩触发检查
tests/           CPU语义验证及CUDA数值验证
validation/      本地验证记录，不含伪造GPU成功记录
delivery/        可上传ZIP及SHA256

复核源码：.venv/bin/python prepare.py --verify
本地从Git检出重新生成快照：python prepare.py（会更新PhaseKV快照，应作为新实验版本）
上传包不含.git，只执行--verify，不要在服务器重新运行无参数prepare.py。
