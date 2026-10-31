# PhaseKV 单卡测试结果

记录状态：`{"ok": 5895}`。
排除未完成全部方法的质量/检索案例：1105。

分数仅在所有选定方法成功且输入哈希一致的样本交集上比较；OOM 不计作零分。
参数是各方法显式配置，不是等 KV 字节预算；SCOPE 为本仓库 ALLKV+Slide 版本。
截断 LongBench、custom_passkey 与官方完整排行榜/RULER 有不同口径。
长输出测速忽略 EOS，不参与质量评分。速度含生成循环、缓存维护和有限值检查；不是纯 kernel 速度。
容量测试追踪真实 allocated 峰值；reserved 显存不代表有效 KV；不做虚假显存填充。

| 模型 | 任务 | 输入预算 | 方法 | 配对样本 | 分数 | 相对 full | KV MiB | decode触发样本 |
|---|---|---:|---|---:|---:|---:|---:|---:|

## 固定长度测速（已排除各形状第0次预热）

capacity_specific 各方法长度可能不同，不能直接比较其速度。

| 模型 | 方法 | 类型 | 输入 | 输出 | 次数 | decode token/s | prefill+压缩 s | 峰值 GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | full | capacity_specific | 32640 | 128 | 3 | 18.99 | 7.622 | 16.47 |
| Qwen2.5-7B-Instruct | full | context_sweep | 1024 | 256 | 3 | 52.36 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | full | context_sweep | 2048 | 256 | 3 | 50.72 | 0.266 | 14.42 |
| Qwen2.5-7B-Instruct | full | context_sweep | 3072 | 256 | 3 | 48.74 | 0.399 | 14.48 |
| Qwen2.5-7B-Instruct | full | context_sweep | 4096 | 256 | 3 | 46.28 | 0.540 | 14.53 |
| Qwen2.5-7B-Instruct | full | context_sweep | 8192 | 256 | 3 | 38.51 | 1.178 | 14.82 |
| Qwen2.5-7B-Instruct | full | context_sweep | 16384 | 256 | 3 | 28.20 | 2.816 | 15.38 |
| Qwen2.5-7B-Instruct | full | context_sweep | 24576 | 256 | 3 | 22.62 | 4.935 | 15.94 |
| Qwen2.5-7B-Instruct | full | context_sweep | 32000 | 256 | 3 | 19.21 | 7.393 | 16.44 |
| Qwen2.5-7B-Instruct | full | joint_sweep | 8192 | 1024 | 3 | 37.75 | 1.178 | 14.85 |
| Qwen2.5-7B-Instruct | full | joint_sweep | 16384 | 2048 | 3 | 27.45 | 2.816 | 15.47 |
| Qwen2.5-7B-Instruct | full | joint_sweep | 24576 | 4096 | 3 | 21.61 | 4.935 | 16.15 |
| Qwen2.5-7B-Instruct | full | output_sweep | 1024 | 256 | 3 | 52.38 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | full | output_sweep | 1024 | 1024 | 3 | 51.96 | 0.139 | 14.38 |
| Qwen2.5-7B-Instruct | full | output_sweep | 1024 | 2048 | 3 | 51.03 | 0.139 | 14.44 |
| Qwen2.5-7B-Instruct | full | output_sweep | 1024 | 3072 | 3 | 49.97 | 0.139 | 14.51 |
| Qwen2.5-7B-Instruct | full | output_sweep | 1024 | 4096 | 3 | 48.93 | 0.140 | 14.58 |
| Qwen2.5-7B-Instruct | kivi | capacity_specific | 32640 | 128 | 3 | 25.33 | 7.725 | 16.73 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 1024 | 256 | 3 | 33.46 | 0.154 | 14.35 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 2048 | 256 | 3 | 33.77 | 0.280 | 14.42 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 3072 | 256 | 3 | 33.65 | 0.413 | 14.48 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 4096 | 256 | 3 | 33.73 | 0.553 | 14.55 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 8192 | 256 | 3 | 33.45 | 1.195 | 14.86 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 16384 | 256 | 3 | 32.93 | 2.853 | 15.49 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 24576 | 256 | 3 | 28.63 | 4.994 | 16.11 |
| Qwen2.5-7B-Instruct | kivi | context_sweep | 32000 | 256 | 3 | 25.48 | 7.489 | 16.68 |
| Qwen2.5-7B-Instruct | kivi | joint_sweep | 8192 | 1024 | 3 | 33.51 | 1.191 | 14.86 |
| Qwen2.5-7B-Instruct | kivi | joint_sweep | 16384 | 2048 | 3 | 32.42 | 2.850 | 15.49 |
| Qwen2.5-7B-Instruct | kivi | joint_sweep | 24576 | 4096 | 3 | 27.75 | 4.995 | 16.11 |
| Qwen2.5-7B-Instruct | kivi | output_sweep | 1024 | 256 | 3 | 33.13 | 0.154 | 14.35 |
| Qwen2.5-7B-Instruct | kivi | output_sweep | 1024 | 1024 | 3 | 33.70 | 0.154 | 14.35 |
| Qwen2.5-7B-Instruct | kivi | output_sweep | 1024 | 2048 | 3 | 33.72 | 0.154 | 14.35 |
| Qwen2.5-7B-Instruct | kivi | output_sweep | 1024 | 3072 | 3 | 33.71 | 0.154 | 14.35 |
| Qwen2.5-7B-Instruct | kivi | output_sweep | 1024 | 4096 | 3 | 33.66 | 0.154 | 14.35 |
| Qwen2.5-7B-Instruct | pyramidkv | capacity_specific | 32640 | 128 | 3 | 35.74 | 7.633 | 18.16 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 1024 | 256 | 3 | 54.35 | 0.144 | 14.37 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 2048 | 256 | 3 | 53.56 | 0.269 | 14.50 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 3072 | 256 | 3 | 52.90 | 0.402 | 14.62 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 4096 | 256 | 3 | 52.34 | 0.542 | 14.74 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 8192 | 256 | 3 | 49.96 | 1.180 | 15.22 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 16384 | 256 | 3 | 44.55 | 2.821 | 16.21 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 24576 | 256 | 3 | 39.44 | 4.945 | 17.19 |
| Qwen2.5-7B-Instruct | pyramidkv | context_sweep | 32000 | 256 | 3 | 35.92 | 7.404 | 18.09 |
| Qwen2.5-7B-Instruct | pyramidkv | joint_sweep | 8192 | 1024 | 3 | 49.28 | 1.180 | 15.22 |
| Qwen2.5-7B-Instruct | pyramidkv | joint_sweep | 16384 | 2048 | 3 | 42.75 | 2.821 | 16.21 |
| Qwen2.5-7B-Instruct | pyramidkv | joint_sweep | 24576 | 4096 | 3 | 36.46 | 4.945 | 17.19 |
| Qwen2.5-7B-Instruct | pyramidkv | output_sweep | 1024 | 256 | 3 | 54.37 | 0.144 | 14.37 |
| Qwen2.5-7B-Instruct | pyramidkv | output_sweep | 1024 | 1024 | 3 | 53.38 | 0.144 | 14.37 |
| Qwen2.5-7B-Instruct | pyramidkv | output_sweep | 1024 | 2048 | 3 | 52.34 | 0.144 | 14.41 |
| Qwen2.5-7B-Instruct | pyramidkv | output_sweep | 1024 | 3072 | 3 | 51.24 | 0.144 | 14.47 |
| Qwen2.5-7B-Instruct | pyramidkv | output_sweep | 1024 | 4096 | 3 | 50.34 | 0.144 | 14.54 |
| Qwen2.5-7B-Instruct | scope | capacity_specific | 32640 | 128 | 3 | 15.19 | 7.622 | 16.48 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 1024 | 256 | 3 | 39.48 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 2048 | 256 | 3 | 39.70 | 0.265 | 14.42 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 3072 | 256 | 3 | 39.50 | 0.399 | 14.48 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 4096 | 256 | 3 | 36.90 | 0.540 | 14.53 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 8192 | 256 | 3 | 30.84 | 1.178 | 14.82 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 16384 | 256 | 3 | 22.86 | 2.815 | 15.38 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 24576 | 256 | 3 | 18.03 | 4.934 | 15.94 |
| Qwen2.5-7B-Instruct | scope | context_sweep | 32000 | 256 | 3 | 15.02 | 7.393 | 16.44 |
| Qwen2.5-7B-Instruct | scope | joint_sweep | 8192 | 1024 | 3 | 30.46 | 1.177 | 14.82 |
| Qwen2.5-7B-Instruct | scope | joint_sweep | 16384 | 2048 | 3 | 22.50 | 2.816 | 15.38 |
| Qwen2.5-7B-Instruct | scope | joint_sweep | 24576 | 4096 | 3 | 17.72 | 4.931 | 15.94 |
| Qwen2.5-7B-Instruct | scope | output_sweep | 1024 | 256 | 3 | 39.78 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | scope | output_sweep | 1024 | 1024 | 3 | 39.01 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | scope | output_sweep | 1024 | 2048 | 3 | 39.14 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | scope | output_sweep | 1024 | 3072 | 3 | 39.00 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | scope | output_sweep | 1024 | 4096 | 3 | 39.06 | 0.139 | 14.35 |
| Qwen2.5-7B-Instruct | snapkv | capacity_specific | 32640 | 128 | 3 | 35.62 | 7.634 | 18.16 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 1024 | 256 | 3 | 54.13 | 0.144 | 14.37 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 2048 | 256 | 3 | 53.48 | 0.270 | 14.50 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 3072 | 256 | 3 | 52.57 | 0.402 | 14.62 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 4096 | 256 | 3 | 52.09 | 0.542 | 14.74 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 8192 | 256 | 3 | 49.85 | 1.180 | 15.22 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 16384 | 256 | 3 | 45.20 | 2.823 | 16.21 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 24576 | 256 | 3 | 39.98 | 4.942 | 17.19 |
| Qwen2.5-7B-Instruct | snapkv | context_sweep | 32000 | 256 | 3 | 35.88 | 7.405 | 18.09 |
| Qwen2.5-7B-Instruct | snapkv | joint_sweep | 8192 | 1024 | 3 | 49.25 | 1.179 | 15.22 |
| Qwen2.5-7B-Instruct | snapkv | joint_sweep | 16384 | 2048 | 3 | 43.10 | 2.821 | 16.21 |
| Qwen2.5-7B-Instruct | snapkv | joint_sweep | 24576 | 4096 | 3 | 36.40 | 4.946 | 17.19 |
| Qwen2.5-7B-Instruct | snapkv | output_sweep | 1024 | 256 | 3 | 54.01 | 0.144 | 14.37 |
| Qwen2.5-7B-Instruct | snapkv | output_sweep | 1024 | 1024 | 3 | 53.26 | 0.144 | 14.37 |
| Qwen2.5-7B-Instruct | snapkv | output_sweep | 1024 | 2048 | 3 | 52.29 | 0.144 | 14.40 |
| Qwen2.5-7B-Instruct | snapkv | output_sweep | 1024 | 3072 | 3 | 51.23 | 0.144 | 14.46 |
| Qwen2.5-7B-Instruct | snapkv | output_sweep | 1024 | 4096 | 3 | 50.24 | 0.144 | 14.53 |
