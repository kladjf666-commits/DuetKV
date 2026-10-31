# PhaseKV 单卡测试结果

记录状态：`{"ok": 1179}`。
排除未完成全部方法的质量/检索案例：0。

分数仅在所有选定方法成功且输入哈希一致的样本交集上比较；OOM 不计作零分。
参数是各方法显式配置，不是等 KV 字节预算；SCOPE 为本仓库 ALLKV+Slide 版本。
截断 LongBench、custom_passkey 与官方完整排行榜/RULER 有不同口径。
长输出测速忽略 EOS，不参与质量评分。速度含生成循环、缓存维护和有限值检查；不是纯 kernel 速度。
容量测试追踪真实 allocated 峰值；reserved 显存不代表有效 KV；不做虚假显存填充。

| 模型 | 任务 | 输入预算 | 方法 | 配对样本 | 分数 | 相对 full | KV MiB | decode触发样本 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | custom_passkey | 1024 | phasekv | 9 | 100.00 | — | 5.8 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 2048 | phasekv | 9 | 100.00 | — | 9.0 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 3072 | phasekv | 9 | 88.89 | — | 12.1 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 4096 | phasekv | 9 | 100.00 | — | 15.3 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 8192 | phasekv | 9 | 100.00 | — | 27.8 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 16384 | phasekv | 9 | 100.00 | — | 53.0 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 24576 | phasekv | 9 | 100.00 | — | 78.3 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 32000 | phasekv | 9 | 77.78 | — | 101.1 | 0 |
| Qwen2.5-7B-Instruct | custom_passkey | 32640 | phasekv | 9 | 100.00 | — | 103.0 | 0 |
| Qwen2.5-7B-Instruct | 2wikimqa | 1536 | phasekv | 32 | 19.58 | — | 7.4 | 0 |
| Qwen2.5-7B-Instruct | 2wikimqa | 3072 | phasekv | 32 | 30.53 | — | 12.1 | 0 |
| Qwen2.5-7B-Instruct | 2wikimqa | 8192 | phasekv | 32 | 39.50 | — | 22.4 | 0 |
| Qwen2.5-7B-Instruct | 2wikimqa | 16384 | phasekv | 32 | 41.59 | — | 25.0 | 0 |
| Qwen2.5-7B-Instruct | gov_report | 1536 | phasekv | 32 | 25.74 | — | 14.8 | 32 |
| Qwen2.5-7B-Instruct | gov_report | 3072 | phasekv | 32 | 27.13 | — | 20.1 | 32 |
| Qwen2.5-7B-Instruct | gov_report | 8192 | phasekv | 32 | 29.16 | — | 33.0 | 32 |
| Qwen2.5-7B-Instruct | gov_report | 16384 | phasekv | 32 | 28.97 | — | 43.8 | 32 |
| Qwen2.5-7B-Instruct | hotpotqa | 1536 | phasekv | 32 | 30.89 | — | 7.4 | 0 |
| Qwen2.5-7B-Instruct | hotpotqa | 3072 | phasekv | 32 | 32.79 | — | 12.1 | 0 |
| Qwen2.5-7B-Instruct | hotpotqa | 8192 | phasekv | 32 | 51.48 | — | 27.2 | 0 |
| Qwen2.5-7B-Instruct | hotpotqa | 16384 | phasekv | 32 | 51.06 | — | 45.4 | 0 |
| Qwen2.5-7B-Instruct | multi_news | 1536 | phasekv | 32 | 22.08 | — | 15.2 | 32 |
| Qwen2.5-7B-Instruct | multi_news | 3072 | phasekv | 32 | 22.54 | — | 17.4 | 32 |
| Qwen2.5-7B-Instruct | multi_news | 8192 | phasekv | 32 | 22.52 | — | 19.4 | 32 |
| Qwen2.5-7B-Instruct | multi_news | 16384 | phasekv | 32 | 22.50 | — | 19.7 | 32 |
| Qwen2.5-7B-Instruct | multifieldqa_en | 1536 | phasekv | 32 | 38.58 | — | 7.4 | 7 |
| Qwen2.5-7B-Instruct | multifieldqa_en | 3072 | phasekv | 32 | 41.92 | — | 11.8 | 6 |
| Qwen2.5-7B-Instruct | multifieldqa_en | 8192 | phasekv | 32 | 46.12 | — | 21.2 | 6 |
| Qwen2.5-7B-Instruct | multifieldqa_en | 16384 | phasekv | 32 | 49.49 | — | 24.6 | 8 |
| Qwen2.5-7B-Instruct | narrativeqa | 1536 | phasekv | 32 | 20.95 | — | 7.4 | 0 |
| Qwen2.5-7B-Instruct | narrativeqa | 3072 | phasekv | 32 | 23.37 | — | 12.1 | 0 |
| Qwen2.5-7B-Instruct | narrativeqa | 8192 | phasekv | 32 | 19.29 | — | 27.8 | 0 |
| Qwen2.5-7B-Instruct | narrativeqa | 16384 | phasekv | 32 | 22.85 | — | 48.8 | 0 |
| Qwen2.5-7B-Instruct | qasper | 1536 | phasekv | 32 | 30.69 | — | 7.4 | 0 |
| Qwen2.5-7B-Instruct | qasper | 3072 | phasekv | 32 | 31.89 | — | 12.0 | 0 |
| Qwen2.5-7B-Instruct | qasper | 8192 | phasekv | 32 | 35.31 | — | 18.2 | 0 |
| Qwen2.5-7B-Instruct | qasper | 16384 | phasekv | 32 | 35.31 | — | 18.6 | 0 |
| Qwen2.5-7B-Instruct | qmsum | 1536 | phasekv | 32 | 19.75 | — | 7.9 | 29 |
| Qwen2.5-7B-Instruct | qmsum | 3072 | phasekv | 32 | 20.98 | — | 12.6 | 28 |
| Qwen2.5-7B-Instruct | qmsum | 8192 | phasekv | 32 | 22.91 | — | 27.3 | 27 |
| Qwen2.5-7B-Instruct | qmsum | 16384 | phasekv | 32 | 23.74 | — | 42.1 | 28 |

## 固定长度测速（已排除各形状第0次预热）

capacity_specific 各方法长度可能不同，不能直接比较其速度。

| 模型 | 方法 | 类型 | 输入 | 输出 | 次数 | decode token/s | prefill+压缩 s | 峰值 GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | phasekv | capacity_specific | 32640 | 128 | 3 | 41.82 | 7.658 | 18.16 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 1024 | 256 | 3 | 49.95 | 0.161 | 14.37 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 2048 | 256 | 3 | 49.83 | 0.285 | 14.50 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 3072 | 256 | 3 | 49.67 | 0.418 | 14.62 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 4096 | 256 | 3 | 49.30 | 0.559 | 14.73 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 8192 | 256 | 3 | 48.34 | 1.196 | 15.22 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 16384 | 256 | 3 | 45.35 | 2.842 | 16.21 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 24576 | 256 | 3 | 43.88 | 4.965 | 17.19 |
| Qwen2.5-7B-Instruct | phasekv | context_sweep | 32000 | 256 | 3 | 41.54 | 7.428 | 18.09 |
| Qwen2.5-7B-Instruct | phasekv | joint_sweep | 8192 | 1024 | 3 | 46.18 | 1.196 | 15.22 |
| Qwen2.5-7B-Instruct | phasekv | joint_sweep | 16384 | 2048 | 3 | 41.52 | 2.840 | 16.21 |
| Qwen2.5-7B-Instruct | phasekv | joint_sweep | 24576 | 4096 | 3 | 36.48 | 4.961 | 17.19 |
| Qwen2.5-7B-Instruct | phasekv | output_sweep | 1024 | 256 | 3 | 50.09 | 0.161 | 14.37 |
| Qwen2.5-7B-Instruct | phasekv | output_sweep | 1024 | 1024 | 3 | 47.66 | 0.161 | 14.37 |
| Qwen2.5-7B-Instruct | phasekv | output_sweep | 1024 | 2048 | 3 | 45.27 | 0.161 | 14.38 |
| Qwen2.5-7B-Instruct | phasekv | output_sweep | 1024 | 3072 | 3 | 42.72 | 0.161 | 14.45 |
| Qwen2.5-7B-Instruct | phasekv | output_sweep | 1024 | 4096 | 3 | 40.66 | 0.161 | 14.52 |
