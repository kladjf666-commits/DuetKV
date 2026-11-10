# SCOPE 实现核对（2026-10-24）

## 核对依据

- [论文](https://aclanthology.org/2025.acl-long.529/)，ACL 2025 主会长文，非 Findings。
- [作者仓库](https://github.com/Linking-ai/SCOPE)，提交 `8cbae48042e34da630c7a044f535de7a991c2999`。
- 本地官方源码：`official/model/kv_utils.py` 的 `ALLKVCluster.update_kv_in_decoding`、`official/model/llama_model.py`。

## 结论与修正

旧版 `baselines/scope.py` 使用整个生成过程累加的注意力，不能称作官方 Slide 的忠实复现。官方 Slide 使用当前 query 的注意力选择生成历史。本地现已取消累计评分，并明确标成 ALLKV + Slide 的 GQA 适配；代码迁至 `compressor.py`。

SCOPE 将 prefill 与 decode 缓存分开管理，保护的是 prefill 阶段最终保留的缓存。它可以与 prefill 压缩方法组合，不能概括为“一定保留原始完整 prompt”。当前本地只支持 ALLKV（完整 prompt）组合，以及固定 decode 预算的 Slide；Adaptive、Discontinuous、压缩 prompt 插件未实现，传入相关参数会明确拒绝。

本地为每个 KV head 聚合其 query group 的分数，官方适配使用重复后的 query-head KV；两者不能宣称完全相同。保留下标排序也与官方 top-k 顺序不同。单步 attention 使用删除前的 KV；新缓存供下一步使用，与核对的官方 Llama 路径效果一致。在恰好达到预算时，本地跳过不会删除 token 的重排。

## 已验证与待验证

- 合成 MHA 输入上，连续18步与官方 ALLKV + Slide 的保留集合逐步一致。
- 反例测试确认：旧历史高分 token 不会因累积优势盖过当前 query 的选择。
- 已检查未达到预算、刚好达到预算、首次超预算、最近窗口及 prompt 保留。
- 目录整理和兼容检查完成后，已在本机3070 Ti上运行 Qwen2-1.5B：同一1555-token输入，强制输出80/258 tokens。256生成段预算分别未触发/在第257次追加首次触发；日志见 `../results/compression_triggers_qwen2_20261024.json`。
- 上述验证覆盖本地GQA适配的运行与触发边界，不等于与官方扩展到query-head缓存的完整模型逐logit一致性证明。

## 旧结果的解释

2026-10-23 的160条 SCOPE 样本使用旧累计评分实现，且256预算从未触发淘汰。其与 full 相同的结果只能说明该配置未执行压缩，不能用于衡量官方 SCOPE 的有效压缩表现。旧预测文件保留原样，不能为它们补造当时未采集的触发日志。
