# DeepSeek V4 Flash 部署

本目录提供 **vLLM 原生部署入口**。默认使用
`deepseek-ai/DeepSeek-V4-Flash`（预览版）；`--variant 0731` 使用
`deepseek-ai/DeepSeek-V4-Flash-0731`。两个版本是不同权重，租卡和对比实验时
应记录具体版本。建议在独立的 vLLM 环境或官方镜像中运行，避免覆盖本仓库为
Qwen/Llama/OPT 固定的 Transformers 4.46.3 依赖。

## 显存与磁盘

| 版本 | vLLM 配方的最低显存参考值 | 推荐租卡起点 |
| --- | ---: | --- |
| 预览版 | 约 170GB | 同机 4×80GB，先做短上下文启动验证 |
| 0731 | 约 200GB | 同机 4×80GB，先做短上下文启动验证 |

**单张 80GB 不够**：13B 是每 token 激活参数量，完整 MoE 权重约 284B 参数。
最低显存数值尚未包含你要留出的全部服务余量；能否启动还取决于 GPU 架构、
vLLM 版本、上下文长度及内核。建议数据盘预留至少 250GB 存权重和缓存。
默认上下文上限设为 16K，避免第一次启动就预留百万 token 的 KV 空间。

## 运行

在项目根目录、已安装兼容 vLLM 的环境执行：

```bash
python -m deepseek_v4_flash --dry-run
python -m deepseek_v4_flash --check --tensor-parallel-size 4
python -m deepseek_v4_flash --model /path/to/DeepSeek-V4-Flash \
  --tensor-parallel-size 4 --max-model-len 16384

# 0731 权重：需要与预览版分别下载和记录
python -m deepseek_v4_flash --variant 0731 \
  --model /path/to/DeepSeek-V4-Flash-0731 --tensor-parallel-size 4
```

也可直接运行 `python deepseek_v4_flash/deepseek_v4_flash.py`。
入口会先核对当前可见 GPU 的总显存预算和 vLLM 版本；不满足要求时直接退出，
不会开始下载或加载权重。默认监听 `127.0.0.1:8000`。

## PhaseKV 接入状态

`deepseek_v4_flash_attention.py` 是显式兼容性检查，**尚不是 PhaseKV 适配器**。
DeepSeek V4 的 CSA/HCA 压缩池、稀疏索引、共享 K=V 和部分 RoPE 与当前
`PhaseKVCache` 的逐 token K/V 布局不同。直接包装 Qwen/Llama 注意力会改变模型
计算，因此目前只提供模型原生部署；要比较 PhaseKV，需先实现并验证独立的
DeepSeek V4 缓存及注意力内核。

资料：[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)、
[vLLM 部署配方](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)。
