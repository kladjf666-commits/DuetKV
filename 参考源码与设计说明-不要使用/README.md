# PhaseKV：按 token 的生命周期分层管理 KV Cache

> 这是新建的说明文档（`README.md` 保持原样，仓库里的其他文件也没有改动）。目标是回答四个问题：
> **prompt 的 KV 怎么压、生成的 KV 怎么管、token 怎么被丢掉、这两类 token 为什么不能用同一套做法。**
> 然后是实测结果，以及这些数字能说明什么、不能说明什么。

## 目录

- [0. 一句话版本](#0-一句话版本)
- [1. 设计出发点：三条硬件事实](#1-设计出发点三条硬件事实)
- [2. 缓存在显存里长什么样](#2-缓存在显存里长什么样)
- [3. Prompt 层：一次选择、一次量化、永久冻结](#3-prompt-层一次选择一次量化永久冻结)
- [4. 生成 token：热区、懒批冷却、逐 head 驱逐](#4-生成-token热区懒批冷却逐-head-驱逐)
- [5. Prompt 和生成 token 的做法差异（重点）](#5-prompt-和生成-token-的做法差异重点)
- [6. 注意力怎么在压缩域里算](#6-注意力怎么在压缩域里算)
- [7. 显存管理：账怎么记、峰值为什么反着走](#7-显存管理账怎么记峰值为什么反着走)
- [8. 参数：代码默认、实验定版、推荐值](#8-参数代码默认实验定版推荐值)
- [9. 实测结果](#9-实测结果)
- [10. 这些数字的边界](#10-这些数字的边界)
- [11. 代码地图](#11-代码地图)
- [12. 怎么跑](#12-怎么跑)
- [13. 术语表](#13-术语表)

---

## 0. 一句话版本

PhaseKV 把每条序列的 KV Cache 拆成三层，各用各的表示：

1. **Prompt 层**：prefill 时一次性决定留哪些 token（可以全留）、压到 4-bit，之后只读，永不改写；
2. **热层（hot）**：最近 128 个生成 token，保持 FP16，逐步追加；
3. **冷层（cold）**：滑出热层的生成 token，每攒够 16 个处理一次，每个 KV head 按 V 向量长度留 13 个，压到 4-bit。

注意力**不把 KV 解压回显存**，而是一个 Triton kernel 直接读 4-bit 码，在片上反量化并做 online softmax。

在 RTX 4090、Llama-3.1-8B、batch 6、8192 输入 + 2048 输出下：整批吞吐 132.01 → 164.99 token/s（+25.0%），最终 KV 7.50 → 2.43 GiB，解码阶段峰值显存 22.59 → 17.65 GiB。

代价同样明确：**预填充峰值反而更高**（22.99 GiB）；**batch 1、短 prompt 时可能和 Full 持平或更慢**；**prompt 保留率低于 90% 时，长输出任务质量会明显下降**。

---

## 1. 设计出发点：三条硬件事实

### 1.1 prefill 算得多，decode 读得多

- prefill：P 个 query 对 P 个 key，计算量约 O(P²)，读 KV 只有 O(P)。**算力受限，带宽有余。**
- decode：每步 1 个 query 对 S 个 key，计算量 O(S)，读 KV 也是 O(S)。每读一个字节只做很少的乘加，**带宽受限**。

对 8B 模型，单步 decode 还要读约 15 GiB 权重。所以 batch=1 时，权重读取占了大头，KV 压不压影响不大；batch 变大后同一份权重只读一次、服务多条序列，KV 读取才变成瓶颈。这直接解释了后面的现象：**B1 两条曲线几乎重合，B6 拉开 25%**。

### 1.2 两类 token 被读的次数不一样

- 一个 **prompt token** 的 KV：写一次，之后每一步 decode 都读一次 → 读放大 **T**（T 是解码步数）。
- 第 i 个 **生成 token** 的 KV：只被它之后的 (T−i) 步读到 → 平均读放大约 **T/2**。

推论：省 prompt 的 1 个字节，比省生成 token 的 1 个字节更值钱（约 2 倍）。所以同样的压缩比例，应该优先考虑 prompt。

### 1.3 重要度信号的价格不一样

- prefill 本来就要算完整的注意力矩阵 → 顺手拿分数做选择，几乎零额外成本。
- decode 每步只算 1 个 query；为了“选 token”去 materialize 注意力分数，要额外读写，正好和最缺的带宽打架 → 换一个便宜信号：**V 向量的 L2 范数**。

---

## 2. 缓存在显存里长什么样

真实的缓存类在 [PhaseKV/phasekv_cache.py](PhaseKV/phasekv_cache.py)。这套结构在代码里叫“相位异构三层缓存”：按 token 处在 prompt 阶段还是生成阶段、以及生成阶段的冷热，给不同的表示。每个模型层对应一个 `LayerTiers`：

```text
PhaseKVCache
└─ layers[0 .. 31]                     每层一个 LayerTiers
   ├─ prompt                           首轮 prompt：保留的 token，4-bit 打包，只读
   ├─ prompt_extensions[]              多轮新增输入：整块 4-bit 打包，只读
   ├─ cold[]                           已冷却的生成 token：每段 13 个，4-bit，只读
   └─ hot_k / hot_v                    最近的生成 token：FP16 环形缓冲，容量 144
```

几个容易误解的点：

- **三层不是一块连续内存**。它们是各自独立的打包段/张量，注意力通过“描述表”分别去读（见第 6 节）。所以不存在“把 prompt 和 cold 拼起来”的步骤。
- **prompt 每个 head 保留的 token 位置可以不同**，但数量相同（都是 K0 个），张量仍是矩形。kernel 按 head 索引，所以一张描述表能覆盖所有 head。
- **物理长度可以小于真实长度**。丢掉的 token 是物理删除；但模型仍然需要知道“我现在在第几个位置”。

三个长度要分清：

| 名称 | 含义 | 用在哪 |
|---|---|---|
| `true_seq_length` | 已处理过的真实 token 数（含被丢掉的） | 下一次前向的 `position_ids` / RoPE |
| `get_seq_length()` | 当前物理保留的 token 数（prompt + cold + hot） | `cache_position`、容量检查 |
| `hot_len` | 热区里还有多少个 token | 冷却判断 |

一句话：**丢掉的是“记忆”，不是“时间”**。每个 K 在打包前就按它的真实位置加了 RoPE，保留下来的 token 按原始位置升序存放；之后的 query 用真实绝对位置去点乘，算出来的仍是正确的相对距离。

三层对照：

| | Prompt 层 | 热层 | 冷层 |
|---|---|---|---|
| 存什么 | prefill 保留下来的 token | 最近生成的 token | 滑出热区的生成 token |
| 位宽 | `prompt_bits`（实验用 4-bit） | FP16 | `cold_bits`（实验用 4-bit） |
| 谁写 | prefill 时一次 | 每步追加一个 | 每 16 步批量写入 |
| 能否改写 | 否，冻结 | 是，环形覆盖 | 否，写入即冻结 |
| 读放大 | T | 小（越新读得越少） | 约 T/2 |
| 位置是否连续 | 否，per-head 选择 | 是（环形） | 否，per-head 选择 |

---

## 3. Prompt 层：一次选择、一次量化、永久冻结

对应代码：[prompt_tier.py:24](PhaseKV/prompt_tier.py#L24) 的 `select_prompt_indices`，以及 [phasekv_cache.py:233](PhaseKV/phasekv_cache.py#L233) 的 `from_dense_prefill`。

### 3.1 先选 token

设 prompt 长度为 P，保留比例为 r：

```text
K0 = clamp(round(P × r), 1, P)                  # 这一层最终留多少
强制保留：最前面 s 个（sink，实验里 s = 0）
        + 最后 w = 32 个（最近上下文，防止刚说过的话被丢）
剩余预算：从中间候选区 [s, P−w) 里按分数选 K0 − s − w 个
```

分数怎么来：

```text
score_h[j] = Σ_{q ∈ 最后 32 个 query} A_h[q, j]    # 每个 KV head 单独一组，GQA 组内求和
score_h    = maxpool_1d(score_h, kernel=7)          # 让高分向邻居扩散
top_h      = topk(score_h[s : P−w], K0 − s − w) + s # 每个 head 独立选
keep_h     = sort(concat(sink, top_h, recent))      # 按原始位置升序
```

四个细节值得注意：

1. **用最后 32 个 query 行投票**。prefill 末尾的 query 最接近“真正要开始生成”的状态，它们关注过的 prompt token，大概率也是生成时需要的。这就是观察窗口（`obs_window=32`）。
2. **max-pool 平滑**。不平滑的话，top-k 容易选出一堆孤立的尖峰 token，把它们周围的上下文全丢掉；核长 7 的 1D max-pool 会把高分“染”到邻居上，选出来的是一簇一簇的连续片段。
3. **每个 KV head 独立选**。同一个 token 对不同 head 的重要性不一样；强行所有 head 用同一组位置，会浪费预算。
4. **下标按原始位置升序保存**。RoPE 已经烘进 K，纯 decode 的 q·k 其实不依赖存储顺序；但升序能让“缓存里的 token 顺序 = 原序列顺序”，分块 prefill 的逐 token 因果检查、调试和结果对齐都靠这个约定。

举个具体例子：8192 个 token、r = 0.95 → K0 = 7782；末尾 32 个直接保留；剩下 7750 个从前面 8160 个候选里按分数选。如果 r = 1.0，就跳过选择，全部保留、只做量化。

### 3.2 再量化

选完之后，把保留下来的 K/V 打包成低比特。量化策略沿用 KIVI 的结论：

- **K 按 channel 量化**：每个（层, head, channel）沿 token 轴每 32 个一组，组内取 min/max。因为 K 的某些 channel 有系统性离群值，按 channel 分组可以把离群值关在自己的组里。
- **V 按 token 量化**：每个（层, head, token）沿 head_dim 每 32 个一组。V 没有明显的 channel 离群问题，按 token 分组更自然。

每组做非对称 min-max 量化：

```text
scale = (max − min) / 15           # 4-bit，qmax = 15
q     = round((x − min) / scale)   # 0..15，存成 uint8
x_deq = q × scale + min          # 只在 kernel 里现算，不写回显存
```

- 每个量化组额外存 2 个 FP16：`scale` 和 `min`，共 4 字节。
- 4-bit 码两个挤进一个字节。K 的码沿 token 轴打包，所以读一个字节就拿到相邻两个 token 的同一个 channel；V 的码沿 head_dim 打包。
- 打包后的形式是 `uint8 码 + FP16 scale + FP16 min`，**没有 FP16 的 KV 副本**。这是“真实量化”和“伪量化（量化完再反量化回 FP16）”的区别。

### 3.3 4-bit 不等于省 4 倍

这是很容易算错的地方。以一层、一个 token 为单位（Llama-3.1-8B：32 层、8 个 KV head、head_dim 128）：

| 形式 | K + V 每 token 每层 | 相对 FP16 |
|---|---:|---:|
| FP16 | 4096 B | 1× |
| 4-bit，组大小 32 | 约 1280 B | **约 3.2×** |
| 4-bit，组大小 32，只留 70% token | 约 896 B | 约 4.6× |

为什么不是 4×：每组 32 个值只有 32 × 0.5 = 16 字节的码，却要配 4 字节的 scale/min，元数据吃掉了 20%。所以标称 4-bit，实际压缩比约 3.2×。整个 8B 模型一个 token 的 KV 是 128 KiB（FP16），4-bit 后约 40 KiB，70% 保留再降到约 28 KiB。

### 3.4 为什么 prompt 可以这么狠

- 读放大是 T：压它一次，收益乘 T。
- 只写一次，而且写在 prefill —— 这段时间算力受限、带宽有余，多花一点打包时间不心疼。
- 压完就冻结：没有“每步重新量化”的误差累积，也没有每步搬数据的开销。

所以 prompt 允许用更小的位宽、也可以丢 token（实验最终选 4-bit + 100% 保留；2-bit 在 Qwen 复测里质量不过关，见第 8 节）。

### 3.5 多轮新增输入怎么办

新增一轮用户输入时，不能重跑整个 prompt。做法（[phasekv_cache.py:455](PhaseKV/phasekv_cache.py#L455)）：

1. `begin_prompt_chunk(length)`：临时借用热区，不够就扩容；
2. 每层逐 token 写入，写一个就算这个 query 的注意力（保证因果，看不到未来 token）；
3. `finish_prompt_chunk()`：把这一整块按 `prompt_bits` 打包成 `prompt_extensions` 的一个新段，然后恢复热区容量。

注意两点：**新增 prompt 不做选择，全部保留**；**块开始前已经在热区里的生成 token 原样保留**，之后照常按 128/16 的规则冷却。旧的 prompt 段和冷段一个字节都不动。

---

## 4. 生成 token：热区、懒批冷却、逐 head 驱逐

对应代码：[phasekv_cache.py:355](PhaseKV/phasekv_cache.py#L355) 的 `finish_decode_step`、[:376](PhaseKV/phasekv_cache.py#L376) 的 `_cool_batch`、[:416](PhaseKV/phasekv_cache.py#L416) 的 `_cool_aged_tokens`。

### 4.1 热区：最近 128 个，保持 FP16

- 每层每步追加一个 token，写入是 O(1) 的拷贝。
- 为什么不压：新 token 的重要度还不知道；热区只有 144 个位置，2048 输出时只占总 KV 字节的 5% 左右——把它也压到 4-bit 只能再省约 13 MB（约 3%），却要在每步的关键路径上付压缩/解压成本，不划算。
- 用环形缓冲（`hot_ring`）：写指针转圈，冷却时**不搬动**保留下来的热 token，只把 `hot_start` 往前挪。

### 4.2 什么时候冷却

```text
热区容量 = recent_window + lazy_g = 128 + 16 = 144
冷却条件：hot_len − 128 ≥ 16
```

也就是：热区填满 144 个 token 后第一次冷却，之后**每 16 步一次**。这就是“懒批”（lazy batching）：把“选择 + 量化”的成本从每步一次，摊薄成每 16 步一次。

### 4.3 驱逐：16 个里留 13 个

冷却时，对刚滑出热区的那 16 个 token：

```text
对每个 KV head：
    signal[j] = ||V[j]||₂              # 这个 head 上第 j 个 token 的 V 向量长度
    keep = round(decode_keep_ratio × 16) = round(0.8 × 16) = 13
    按 signal 取 top-13，丢掉 3 个
```

- 每个 head 丢的 token 可以不同，但数量相同（保持矩形）。
- 0.8 是名义比例，实际是 **13/16 = 81.25%**，所以配置里写 80%，真实保留率是 81.25%。
- 如果 `decode_keep_ratio=1.0`，整批 16 个都保留，只做 4-bit 量化、不丢 token（Qwen 的交互示例就是这种模式）。
- 为什么用 V 范数：不用 materialize 注意力分数（那会额外读写，破坏带宽）；V 的模长在实践里和“这个 token 值不值得留”有相关性，而且算起来基本免费。
- **一旦丢掉就没了**：冷层不会重新选择，也不会重新量化。老 token 不会被“捞回来”。

### 4.4 量化与冻结

保留下来的 13 个 token 压到 `cold_bits`（实验用 4-bit），K 按 channel、V 按 token，和 prompt 用的是同一套量化。打包结果直接写进页池的固定槽位（见第 7 节），写入即冻结。

### 4.5 生成段整体保留多少

设输出 T 个 token、热窗口 W=128、冷区保留率 D=13/16：

```text
整体保留率 ≈ (W + D × (T − W)) / T
T = 2047 时： (128 + 0.8125 × 1919) / 2047 ≈ 82.4%
```

其中 128 个热 token 是 100% 保留的 FP16，其余按 81.25% 留。冷层再用 4-bit，所以生成段相对 FP16 的体积约为 `0.824 / 3.2 ≈ 26%`。

### 4.6 懒批 + 环形的收益

- **懒批**：每 16 步才做一次 top-k 和打包，而不是每步。选择/量化的开销被摊薄，且都在“非关键路径”的间隙里发生。
- **环形**：冷却时只挪指针，不搬 128 个热 token。否则每 16 步就要拷贝一次 128 个 token 的 FP16 KV。
- **只处理冷却批**：已经冻结的冷段和 prompt 段不参与任何重排，描述表只追加新行（见 6.2）。
- **跨层合并**：当配置正好是“16 个里留 13 个、4-bit、head_dim 128”时，冷却会把 32 层的同一批 token 当成一个 batch，一次算完 norm/topk/打包，而不是启动 32 次小 kernel。这是 4090 定版路径里的一个特例优化（`_cool_batch`），条件不满足时退回逐层处理。

---

## 5. Prompt 和生成 token 的做法差异（重点）

这是 PhaseKV 和“统一量化”“只压 prompt”“只驱逐”的方法最不一样的地方。

| 维度 | Prompt token | 生成的 token |
|---|---|---|
| **压缩时机** | prefill 结束，一次性 | decode 过程中，每 16 步一批 |
| **为什么是这个时机** | prefill 算力受限、带宽有余；且写一次读 T 次 | decode 带宽受限，只能摊到间隙里做；新 token 的重要度还没稳定 |
| **重要度信号** | prefill 注意力分数（最后 32 个 query 行投票 + max-pool） | V 向量的 L2 范数（不用注意力分数） |
| **选择粒度** | 每层、每个 KV head 独立 | 每层、每个 KV head 独立，但只在那 16 个候选里选 |
| **保留比例** | `prefill_keep_ratio`（实验 70%–100%） | `decode_keep_ratio`（实验 0.8 → 13/16） |
| **量化位宽** | `prompt_bits`（实验 4-bit；支持 2/4/8） | `cold_bits`（实验 4-bit） |
| **量化轴** | K 按 channel、V 按 token，组大小 32 | 同左 |
| **是否留在 FP16** | 否，全部打包 | 最近 128 个留在 FP16 热区 |
| **生命周期** | 成型即冻结，永不改写 | 先 FP16 用着，滑出热区后量化冻结，不再改写 |
| **读放大** | T | 约 T/2 |
| **压缩强度** | 机制上可以最狠（允许丢 token、支持 2/4-bit）；定版配置是 100% 保留 + 4-bit | 只丢 3/16，4-bit |
| **错了的后果** | 影响之后**所有** decode 步 | 只影响**剩余**的 decode 步，越老影响越小 |
| **多轮新增** | 新增块全部保留、单独打包 | 不受影响，继续按 128/16 冷却 |

把差异归纳成三句话：

1. **时机不同，成本不同。** prompt 的压缩发生在“带宽有余、算力受限”的 prefill，可以慢慢做；生成 token 的压缩发生在“带宽就是瓶颈”的 decode，必须攒批、必须用便宜信号。
2. **信号不同。** prompt 用注意力分数，因为 prefill 反正要算；生成 token 用 V 范数，因为 decode 算注意力分数太贵。这不是“哪个更准”的问题，是“哪个付得起”的问题。
3. **可逆性不同。** prompt 一次定终身；生成 token 先在 FP16 里“观察” 128 步，再决定去留。所以 prompt 的选择要保守（95%–100% 才稳），生成 token 的驱逐可以激进一点。

---

## 6. 注意力怎么在压缩域里算

入口是 [attention_triton.py:672](PhaseKV/kernels/attention_triton.py#L672) 的 `fused_phasekv_decode`，实际执行在 [segmented_attention.py:158](PhaseKV/kernels/segmented_attention.py#L158)。

### 6.1 两次 launch，不拼、不解压

- **第一次 launch**：每个 program 负责“一个 query head（或一组共享同一 KV head 的 query）× 一个 partial”。partial 覆盖一段 prompt、一段 prompt extension、一组冷段，或者热区。program 内部按 token 块循环，读 4-bit 码 → 片上反量化 → 算 score → 更新 online softmax 的 max/分母/累加器，最后把这三个量写到 partial 缓冲区。
- **第二次 launch**：每个 query head 一个 program，把所有 partial 用稳定的 online softmax 合并，除以分母，写出 FP16 输出。

关键点：**没有 `torch.cat`，没有把整份 KV 反量化到一个 dense 缓冲区，也没有 dense 回退**。HBM 里读到的就是 4-bit 码 + FP16 scale/min。

decode 时 query 在所有已保留 token 之后，所以 kernel 不需要因果 mask；分块 prefill 的因果性由“写一个 token、算一次注意力”保证（见 3.5）。

### 6.2 描述表：地址不复制，只记地址

[segmented_plan.py:110](PhaseKV/kernels/segmented_plan.py#L110) 维护一张 GPU 上的指针表，每行 24 个 int64：

```text
K 码指针, K scale 指针, K min 指针,
V 码指针, V scale 指针, V min 指针,
start, end,                       # 这段覆盖的 token 区间
K/V 位宽、K/V 组大小,              # 4 个
K/V 码和 scale 的 stride           # 12 个
```

- 表只在“冻结段发生变化”（新增冷段、新增 prompt 扩展、重排、换 stream/配置）时重建；正常 decode 每步只读。
- 新增冷段时**只追加新行**，旧行的地址不变（页池的槽位地址稳定，见 7.2）。
- FP32 的 partial 工作区按层复用；开启 `reuse_workspace` 时容量按 2 的幂增长，避免每步重新分配。

### 6.3 片上反量化

对每个 4-bit 码：

```text
code = (byte >> shift) & 0xF
x    = code × scale + min        # 转成 FP32 参与点乘
```

- 打开 `channel_contiguous_k` 时，K 的 4-bit 码按 channel 连续存放，kernel 可以用固定的位移/掩码快路径（`specialized_4bit`），不用每次查位宽和组大小。
- 整个过程在寄存器/共享内存里完成，反量化结果不会写回 HBM。

### 6.4 合并与数值稳定性

- 第一次 launch 的每个 partial 都带自己的 max、分母、累加器；第二次 launch 用标准的 online softmax 合并（先比 max，再缩放累加）。
- 空热区会贡献 `(max=−inf, 分母=0, 累加=0)`，合并时特殊处理，避免 `NaN`。
- 累加全程 FP32，输出才转 FP16。

### 6.5 三个调度优化

| 开关 | 做什么 | 为什么 |
|---|---|---|
| `cold_group=8` | 8 个冷段塞进一个 program，顺序处理 | 冷段很短（13 个 token），一段一个 program 会让 launch 和 partial 数量爆炸 |
| `gqa_reuse` | 同一个 KV head 的多个 query 共享一次 KV 读取/反量化 | GQA 下 4 个 query head 共用一个 KV head，读一次能算四次 |
| `hot_ring` | 热区按 `hot_start` / `hot_len` 取模读 | 冷却不搬数据，热区可以环绕 |

### 6.6 数值上到底差多少

- 在 RTX 3070 Ti 上，新旧两版分段内核在 8 组形状（12/32 个 query head、0/1/32/128 个冷段）上 **max_abs_error = 0.0**；3 个模型用例的 final logit 最大绝对差也是 0.0。记录见 [分段主核本地验证.json](PhaseKV与Full消融/分段主核本地验证.json)。
- 4-bit 量化本身是有损的（误差有多大，看第 9.5 节的质量实验）。
- 另一个容易被混淆的数字：批量测试里用“整批 vs 逐条单独跑”的 logits 相对 L2 做检查，实测最大 2.09%，预注册阈值 2.5%（见 [batch_check.py](workspace_migration/quick_4090d/batch_check.py)）。这个数字衡量的是“把 B 条序列折进 KV head 轴有没有破坏请求独立性”，不是量化误差。

---

## 7. 显存管理：账怎么记、峰值为什么反着走

### 7.1 字节都花在哪

一个打包段（prompt 或 cold）占：

```text
码：tokens × layers × kv_heads × head_dim × bits / 8     （K、V 各一份）
元数据：量化组数 × 4 字节                                  （scale + min，各 FP16）
```

热区占：

```text
容量 144 × layers × kv_heads × head_dim × 2 × 2 字节
```

热区是**按容量**算的，不是按实际用了多少：哪怕只生成了 10 个 token，144 个槽位也已经分配。[phasekv_cache.py:553](PhaseKV/phasekv_cache.py#L553) 的 `physical_nbytes()` 统计的是“实际持有的张量字节”，含页内空闲和热区预留；真正有效的 payload 由 `storage_breakdown()` 单独给出。

### 7.2 页池：冷段不单独分配

冷段每 16 步产生一个、每段只有 13 个 token。如果每次都 `torch.empty`，分配器会被小张量碎片拖死。

[packed_storage.py:34](PhaseKV/packed_storage.py#L34) 的 `PackedPagePool` 按 `(形状, 位宽, 组大小, 设备, 通道布局)` 分池，一页 32 个槽位，每个槽位是一份固定布局的打包张量。一个冷段占一个槽位，槽位地址稳定，所以描述表里的指针不用改。

代价是：**页的容量按 32 个槽位算，未使用的槽位也算物理字节**，但只算一次，而且 `storage_breakdown()` 会把 `page_free_bytes` 单独列出来，不混进“有效 payload”。

### 7.3 工作区：只建一次

- 描述表 + FP32 partial 缓冲区（`heads × capacity × (2 + head_dim) × 4B`）按层缓存；
- 正常 decode 步只读不建；
- 新增冷段时只追加描述行；
- 只有在 prompt/扩展段变化、重排、换 stream 或执行配置变化时才重建；
- `physical_nbytes()` 只统计 KV 张量，不含描述表和 partial；这些算进 `auxiliary_bytes`，但会体现在 `torch.cuda.max_memory_allocated()` 的峰值里。

### 7.4 为什么预填充峰值反而更高

这是 PhaseKV 最容易被误读的地方。

首次 prefill 的流程是：**先用普通的 dense cache 跑完 prefill（需要选 token 时还要 `output_attentions=True` 拿注意力分数），再把 dense KV gather + 打包**。在打包完成之前，dense KV 和打包后的 KV 同时存在；打包结束后调用方释放 dense cache，进入只有打包 KV 的解码稳态。所以：

```text
预填充峰值 = 权重 + dense KV + 打包后的 KV + 临时量 + 工作区
解码峰值   = 权重 + 打包 KV + 工作区
```

在 8192+2048、batch 6 下：

| 指标 | Full + FA2 | PhaseKV |
|---|---:|---:|
| 预填充峰值 | 21.31 GiB | **22.99 GiB** |
| 解码峰值 | 22.59 GiB | **17.65 GiB** |
| 最终 KV | 7.50 GiB | **2.43 GiB** |

所以正确的说法是：**PhaseKV 降低的是“稳态 KV”和“解码阶段峰值”，不是全过程峰值**。要降低预填充峰值，得改成流式 prefill + 边算边压，那是另一件事。

### 7.5 为什么最终 KV 省 3 倍，峰值只省 5 GiB

因为 8B 模型的 FP16 权重约 15 GiB，是个常数项：

```text
Full：     15 GiB 权重 + 7.5 GiB KV              ≈ 22.5 GiB  → 实测 22.59
PhaseKV：  15 GiB 权重 + 2.4 GiB KV + 工作区      ≈ 17.4 GiB  → 实测 17.65
```

省下的 4.9 GiB 基本就是 KV 的差值。看显存收益时，应该看“权重之上那一层”，而不是看总峰值。

---

## 8. 参数：代码默认、实验定版、推荐值

下表“代码默认”列取自共享运行入口（`models/packed_runtime.py` 的 `options`）和 `PhaseKVCache.from_dense_prefill` 的默认参数。

| 参数 | 代码默认 | 4090 定版实验 | 说明 |
|---|---:|---:|---|
| `prefill_keep_ratio` | 0.3 | **1.0** | prompt 保留比例；1.0 表示不丢 token，只量化 |
| `prompt_bits` | 2 | **4** | 2-bit 在 Qwen 复测里质量不过关，不要用于正式对比 |
| `cold_bits` | 4 | **4** | 冷层位宽 |
| `group_size` | 32 | **32** | 量化组大小 |
| `recent_window` | 32 | **128** | 热区长度 |
| `lazy_g` | 16 | **16** | 每攒多少 token 冷却一次 |
| `decode_keep_ratio` | 1.0 | **0.8** | 冷区保留比例（16 留 13） |
| `sink_tokens` | 0 | **0** | 强制保留的开头 token 数；Qwen 交互示例用 4 |
| `obs_window` | 32 | **32** | 投票窗口 / 强制保留的末尾 token 数 |
| `optimizations` | 全关 | **P5 + 通道连续 K + GQA 复用** | 见下 |

**默认值是为了跑通和兼容，不是推荐配置。** 默认 `prefill_keep_ratio=0.3`、`prompt_bits=2` 是历史默认，直接拿来跑长输出会很难看。

优化开关在 [optimizations.py:5](PhaseKV/optimizations.py#L5)，`stage_configs()` 给出累积档位：

| 档位 | 内容 |
|---|---|
| `p1` | 融合量化（4-bit Triton 量化直接写目标槽位） |
| `p2a` | 描述表/工作区复用 |
| `p2b` | 冷区分页 |
| `p3` | 冷段分组（`cold_group=8`） |
| `p4` | 4-bit 专用注意力快路径 |
| `p5` | 环形热区（在 p4 之上） |

定版实验还额外打开 `channel_contiguous_k`（K 的 4-bit 码按通道连续存放）和 `gqa_reuse`（同一 KV head 的多个 query 共享一次读取），这两个不在 `p5` 档里。这些开关默认全关；打开方法是：

```python
optimizations = {
    "fused_quantize": True, "reuse_workspace": True, "paged_cold": True,
    "page_slots": 32, "cold_group": 8, "specialized_4bit": True,
    "hot_ring": True, "channel_contiguous_k": True,
}
```

### 8.1 为什么是 4-bit，不是 2-bit

Qwen2.5-7B 的强制复测（[analysis.txt](reports/phasekv_prompt_retest_20261027/analysis.txt)）在同一 prefill 路径下比较了 70%/100% × 2/4/8-bit：

- 2-bit：QA 相对同路径 Full 下降 1.37 / 1.48 分，未过“每组均值下降不超过 1 分”的门槛；
- 4-bit：QA 下降 0.04 / 0.30 分，通过；六种形状的 KV 比值几何平均降到 26.13% / 33.36%；
- 8-bit：QA 只多 0.15 / 0.19 分，配对区间跨零，KV 比却升到 39.42% / 51.55%。

结论：**4-bit 是这批实验里的甜点，2-bit 不要用，8-bit 不划算。**

### 8.2 可选的预算分配器

[planner.py:217](PhaseKV/planner.py#L217) 是一个离线的 roofline 预算分配器：给定模型尺寸、prompt 长度、解码长度和目标显存比例，在网格上搜索 `(保留比例, 位宽, 热窗口, 冷区比例, 懒批 G)`。

它的目标函数是字典序：

1. 在“稳态逻辑字节 ≤ 目标”的可行域里，**用满预算**（逻辑字节最大，作为精度的单调代理）；
2. 同等预算下，**读流量最小**（把预算优先给读放大低的 decode 层）。

这只是“选超参”的工具，不产出任何精度/速度结论。**正式 4090 实验没有用它**，参数是手选的。

---

## 9. 实测结果

### 9.1 环境与口径

| 项 | 值 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct，FP16 |
| GPU | 单张 RTX 4090（24 GB） |
| 软件 | PyTorch 2.5.1+cu121、FlashAttention 2.7.4.post1、Transformers 4.46.3、Triton 3.1.0 |
| 计时 | 首 token 之后开始，固定输出长度，贪心解码 |
| 重复 | batch 1 取三轮中位数；batch 2–6 每个配置预热一次、正式测一次 |
| 配置 | prompt 100% 保留 + 4-bit；热区 128 个 FP16；冷区 80%（16 留 13）+ 4-bit |

batch 1 的完整数据在 [summary.json](PhaseKV与Full消融/4090_prompt100_speed_2cases_VjjdG5Fk/summary.json)，batch 2–6 在 [summary.json](PhaseKV与Full消融/4090_batch2_6_fAryLQ4D/summary.json)。

**batch 2–6 是怎么跑的**：不是 Hugging Face 的批量 API，而是把 B 条等长序列折进 KV head 轴（`[B, Hkv, S, D] → [1, B×Hkv, S, D]`），用一次前向算完 B 条。每条序列的 KV 头只属于自己，GQA 比例不变。跑之前做过“整批 vs 逐条”的 logits 一致性检查（≤2.1%，阈值 2.5%）。这能测吞吐趋势，但不能当成通用 batch serving 实现。

### 9.2 单序列（batch = 1）

| 输入 + 输出 | Full + FA2 | PhaseKV | 差异 |
|---|---:|---:|---:|
| 9192 + 256 | 44.50 | 44.19 | −0.7% |
| 8192 + 2048 | 44.56 | 44.72 | +0.4% |

单序列基本是平的。原因在第 1.1 节：batch=1 时每步要读约 15 GiB 权重，KV 只占一小部分，压缩省下的带宽被 kernel 的固定开销抵掉了。在同一次运行里（8192+2048），PhaseKV 的预填充慢约 7%（1.03 s vs 0.96 s），因为要多做一次 eager 注意力 + 打包。

### 9.3 批量（整批 token/s）

| Batch | 9192+256 Full | PhaseKV | 提升 | 8192+2048 Full | PhaseKV | 提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.50 | 44.19 | −0.7% | 44.56 | 44.72 | +0.4% |
| 2 | 74.46 | 78.26 | +5.1% | 74.32 | 78.88 | +6.1% |
| 3 | 93.32 | 105.00 | +12.5% | 94.34 | 106.49 | +12.9% |
| 4 | 109.26 | 126.73 | +16.0% | 109.69 | 129.86 | +18.4% |
| 5 | 120.98 | 144.73 | +19.6% | 122.45 | 148.67 | +21.4% |
| 6 | 130.59 | 显存不足 | — | 132.01 | **164.99** | **+25.0%** |

怎么读：

- batch 越大，权重读取被摊得越薄，KV 读取占比越高，4-bit 的收益才显出来。
- 到 batch 6，8192+2048 的**每序列**速度是 27.50 vs 22.00 token/s（+25%），不是“整批除以 6”的错觉。
- 9192+256 在 batch 6 失败：6 × 9192 个 token 的 dense prefill 加上打包临时量，在 24 GB 卡上放不下。这是预填充峰值问题，不是解码问题。

### 9.4 显存（8192+2048，batch 6）

| 指标 | Full + FA2 | PhaseKV | 变化 |
|---|---:|---:|---:|
| 预填充峰值 | 21.31 GiB | 22.99 GiB | +7.9% |
| 解码峰值 | 22.59 GiB | 17.65 GiB | −4.94 GiB |
| 最终 KV | 7.50 GiB | 2.43 GiB | 3.09× 更小 |

batch 1 是同样的模式：8192+2048 的最终 KV 从 1.25 GiB 降到 0.41 GiB，但整轮峰值从 16.98 GiB 升到 18.12 GiB（预填充两份 KV 同时存在）。

### 9.5 质量：prompt 保留比例是主开关

实验设置：LongGenBench 的 GSM8K、CSQA、MMLU 各 16 条复合提示，共 1600 道子题；batch 1、贪心、最多 4096 token；只改 `prefill_keep_ratio`，其余固定（prompt 4-bit、热区 128、冷区 80%）。评分按**题号**核对，漏答/无法解析记错。

| Prompt 保留率 | GSM8K | CSQA | MMLU | 三项平均 |
|---:|---:|---:|---:|---:|
| 70% | 31.88 | 19.69 | 37.50 | 29.69 |
| 75% | 37.29 | 45.78 | 44.58 | 42.55 |
| 80% | 44.17 | 54.22 | 45.42 | 47.93 |
| 85% | 53.54 | 71.72 | 46.04 | 57.10 |
| 90% | 56.88 | 71.72 | 45.63 | 58.07 |
| 95% | **64.17** | **75.31** | 47.08 | **62.19** |
| 100% | 60.00 | 74.84 | **48.75** | 61.20 |

数据：[七档质量结果](PhaseKV与Full消融/4090_prompt_quality_scan/summary.json)。其中 70% 和 100% 两个端点是复用的历史记录，75%–95% 是本轮新跑；同一张卡、同一份代码指纹。

- 70% → 95%，三项平均从 29.69 涨到 62.19，翻了一倍多。**丢 prompt token 是质量风险的主要来源，比量化位宽的影响大得多。**
- 95% 比 100% 净多答对 **15 题**（1016 vs 1001）：GSM8K +20、CSQA +3、MMLU −8。95% 是“三个任务同时满足官方分不低于 100% 档 −3 分、且无严重循环”的最低已测档位。
- 但每个任务只有 16 条提示，15 题的差距**不足以证明 95% 普遍优于 100%**。

把 100% 档和 4090D 上 Full 的历史按题号成绩放在一起看：Full 是 GSM8K 64.79 / MMLU 47.08 / CSQA 75.94，平均 62.60；95% 档 62.19、100% 档 61.20，差距在 1.4 分以内。**注意这不是配对比较**：设备不同、轮次不同，Full 用 FA2 预填充而 PhaseKV 用 eager，样本也只有 16 条。

### 9.6 更全面的同轮对照（4090D，prompt 70%）

这一轮把 Full、KIVI、SnapKV、SCOPE、PhaseKV 放在同一台 4090D 上跑，共 5861 条计划记录（5021 成功、840 OOM），PhaseKV 完成全部 1072 条质量生成和 72/81 条性能测量。配置是 **prompt 70% + 冷区 16 个全留**，和 9.3/9.5 的 4090 实验不是同一套参数，所以绝对值不能横向拼（[分析结论](不同算法的不同数据集测试/analysis_20261031/分析结论.txt)）：

| 维度 | 结果 |
|---|---|
| 解码速度 | 在 6 个与 Full 共同成功的形状上，是 Full 的 **81.3%–89.8%**；在 8 个与 KIVI 共同成功的形状上快 1.59–1.73× |
| KV 容量 | 最终 KV 本体减少 **64.8%–75.0%**；含辅助空间减少 58.1%–69.0% |
| LongBench | 多数任务与 Full 接近；gov_report 低 1.70 分（95% 区间 [−2.78, −0.61]） |
| RULER | 任务级平均与 Full 接近（multikey/vt 均为 100，multivalue 均为 98.70，cwe 88.65 vs 88.54） |
| LongGenBench | 明显退化：GSM8K/MMLU/CSQA 官方分 20.63/35.63/24.53，Full 为 53.54/48.13/75.94 |

形状之间差异也很大：prompt 越短、输出越长，越吃亏——压缩省下的主要是 prompt 的读取，短 prompt 没多少可省，但 kernel 和冷却的固定开销要付满整段解码；反过来，长 prompt、大 batch 才轮到压缩发挥。

另一轮 4090 的 80% 复测（prompt 70% + 冷区 80%，1126 条正式记录）给出同样的画面：LongBench 44.37 对 Full 44.45、RULER 96.77 对 96.81，几乎重合；LongGenBench 官方分只有 23.70，Full 是 59.20。短输出任务看不出差别，长输出任务一眼就能看出问题（Full 基线来自历史 4090D 实验，见 [80% 复测](PhaseKV与Full消融/80%复测结果对比.md)）。

这一轮用的是 **prompt 70%**。它和第 9.5 节的结论并不矛盾：LongBench 的回答通常只有几百个 token，生成段短，冷层反复冷却/驱逐的影响很小（在 80% 复测里，640 条 LongBench 只有 144 条真正触发了冷却），所以 70% 的 prompt + 4-bit 基本够用。LongGenBench 要连续生成 4096 个 token，还要跟踪 30–40 道小题：prompt 丢 30% 是主因，冷层在 4096 步里每 16 步触发一次，量化误差和格式漂移都会被长输出放大，最后表现为“漏题、重复输出 Answer_6”（CSQA 有 5/16 条触发重复，Full 是 0/16）。

**所以：LongBench/RULER 的“接近”不能外推到长输出任务；长输出必须把 prompt 保留率拉到 95%–100%。**

### 9.7 另一个模型：Qwen2.5-7B（4090D）

[summary.md](results/4090d_qwen_phasekv_segmented/summary.md) 记录了 1179 条全部成功的单卡测试（无未配对样本）：

| 输入 | 输出 | 解码 token/s | 预填充 + 压缩 (s) | 峰值 (GiB) |
|---:|---:|---:|---:|---:|
| 1024 | 256 | 49.95 | 0.161 | 14.37 |
| 8192 | 256 | 48.34 | 1.196 | 15.22 |
| 16384 | 256 | 45.35 | 2.842 | 16.21 |
| 32000 | 256 | 41.54 | 7.428 | 18.09 |

质量上，custom_passkey 在多数长度是 100%，3072 输入为 88.89%、32000 输入为 77.78%；LongBench 的绝对分随任务和长度波动很大（例如 2wikimqa 从 19.58 到 41.59），而且这张表只列了 PhaseKV 自己的分数，没有同轮 Full 对照。它只能说明“这套实现在 7B 模型、32K 上下文下能跑通，速度和显存是合理的”，不能说明所有任务都不掉点。

### 9.8 测试本身

- **CPU 回归**：[release_test_log.txt](results/phasekv_optimization_20261027/release_test_log.txt) 跑了 47 项，**34 通过、13 跳过（无 CUDA）、0 失败**。跳过的是 Triton 编译/数值对照类测试，和“CPU 上通过了”是两回事。
- **GPU 数值**：3070 Ti 上 8 组形状新旧内核 `max_abs_error=0.0`，3 个模型用例 final logit 差 0.0；4090D 上 Qwen2.5-7B 1179 条记录全部成功。
- 测试覆盖的关键不变量：prefill 后 K/V 分别按不同轴打包、prompt 分块不破坏热窗口、冷却阈值与热区尾部、旧段不重打包、描述表增量追加不丢行、页池旧字节不被覆盖、热区环绕/重排、`update()` 拒绝 dense 回退、优化开关可以单独关闭。

---

## 10. 这些数字的边界

1. **跨设备/跨轮次不能直接相减。** 9.5 的 95–100% 档是 4090 的 PhaseKV 内部对比，Full 的 62.60 是历史 4090D 数字；9.6 的 81.3%–89.8% 是另一轮、另一个配置（prompt 70%）。不要把这些数字拼成一张“全面领先”的表。
2. **batch 2–6 的缓存布局是折叠的**，不是通用 batch serving。它验证了吞吐趋势和请求独立性，但不等于“HF generate 支持 batch”。
3. **“省显存”指稳态 KV 和解码峰值**，不是全过程峰值。预填充峰值反而更高，batch 6 的 9192 形状就是因此 OOM。
4. **质量差里有一部分不是压缩造成的。** PhaseKV 为了拿注意力分数用 eager prefill，Full 用 FA2/SDPA；Qwen2.5-7B 的参数扫描里，1024 条质量样本中有 198 条（19.3%）在压缩前首 token 就不同。比较 Full 时要记得这一层路径差异。
5. **每个任务 16 条提示。** 95% vs 100% 的 15 题差距、LongBench 的 ±1 分，都不足以做统计结论。
6. **RULER 有天花板效应**（multikey/vt 接近 100），不能拿“接近满分”证明复杂推理不退化。
7. **冷区 80% 的独立代价没有干净对照。** 80% 与 100% 的对比同时掺了实现版本和设备差异，不能单独归因给“丢了 3/16”。
8. **所有结果都是单卡、单模型、贪心解码**，没有覆盖采样、beam search、多卡、padding batch。
9. **4-bit 是有损的**，没有 dense 回退；接受不了误差就不要开。

---

## 11. 代码地图

| 文件 | 作用 | 是否正式路径 |
|---|---|---|
| [phasekv_cache.py](PhaseKV/phasekv_cache.py) | 三层存储、打包、冷却、多轮输入、字节统计 | ✅ |
| [kernels/packing.py](PhaseKV/kernels/packing.py) | 分组 min-max 量化 + sub-byte 打包 | ✅ |
| [kernels/quantize_triton.py](PhaseKV/kernels/quantize_triton.py) | 4-bit 融合量化，直接写目标槽位（P1） | ✅ 可选 |
| [kernels/segmented_plan.py](PhaseKV/kernels/segmented_plan.py) | 描述表 + partial 工作区 | ✅ |
| [kernels/segmented_attention.py](PhaseKV/kernels/segmented_attention.py) | 两次 launch 的主 kernel（P2–P5） | ✅ |
| [kernels/segmented_baseline.py](PhaseKV/kernels/segmented_baseline.py) | 原始两次 launch 内核的字节副本，所有开关关闭时使用 | 对照 |
| [kernels/attention_triton.py](PhaseKV/kernels/attention_triton.py) | `fused_phasekv_decode` 入口 + 旧的逐段参考实现 | 入口 ✅ / 旧实现对照 |
| [kernels/launch_config.py](PhaseKV/kernels/launch_config.py) | split / block / warps 规划 | ✅ |
| [prompt_tier.py](PhaseKV/prompt_tier.py) | prompt 选择（`select_prompt_indices`） | ✅ 选择函数；`select_and_pack` 是旧路径 |
| [hetero_cache.py](PhaseKV/hetero_cache.py) | GQA 聚合、投票、池化、top-k 工具；以及旧的 fake-quant 状态 | 工具 ✅ / 状态类旧路径 |
| [packed_storage.py](PhaseKV/packed_storage.py) | 冷区页池 | ✅（P2b） |
| [optimizations.py](PhaseKV/optimizations.py) | P1–P5 开关 | ✅ |
| [quant.py](PhaseKV/quant.py) | fake-quant + 逻辑字节折算 | ❌ 旧路径 |
| [compressor.py](PhaseKV/compressor.py) | 旧的 `KVCompressor` 适配 | ❌ 旧路径 |
| [decode_tier.py](PhaseKV/decode_tier.py) | 旧的热/冷维护 | ❌ 旧路径 |
| [planner.py](PhaseKV/planner.py) | roofline 预算分配器 | 离线可选 |
| [models/packed_runtime.py](models/packed_runtime.py) | 生成循环：prefill → 选择/打包 → decode | ✅ |
| [models/Llama/llama_attention.py](models/Llama/llama_attention.py) | Llama 适配 | ✅ |
| [models/Llama/kivi_attention.py](models/Llama/kivi_attention.py) | KIVI 基线适配 | 对照 |

**看到 `compressor.py` / `decode_tier.py` / `quant.py` 里的 fake-quant，不要以为正式测试用的是它。** 那是早期为了接 `eval` 接口做的“量化-反量化”路径：物理张量仍是 FP16，逻辑显存靠位宽折算，不减少真实 HBM 读取。正式路径是 `phasekv_cache.py` + `kernels/`，仓库里的对比脚本也会阻断 fake-quant 入口。

---

## 12. 怎么跑

### 12.1 最小调用

模型必须以 `attn_implementation="eager"`、FP16、CUDA 加载（适配器要求 eager；需要选 token 时 prefill 还要拿注意力分数）：

```python
from models.Llama.packed_runtime import generate_packed

out = generate_packed(
    model, tokenizer, input_ids,          # input_ids: [1, L]
    method="phasekv",
    params={
        "prefill_keep_ratio": 1.0,        # 长输出建议 0.95–1.0
        "prompt_bits": 4, "cold_bits": 4, "group_size": 32,
        "recent_window": 128, "lazy_g": 16, "decode_keep_ratio": 0.8,
        "sink_tokens": 0, "obs_window": 32,
        "optimizations": {
            "fused_quantize": True, "reuse_workspace": True, "paged_cold": True,
            "page_slots": 32, "cold_group": 8, "specialized_4bit": True,
            "hot_ring": True, "channel_contiguous_k": True, "gqa_reuse": True,
        },
    },
    max_new_tokens=2048,
)
print(out["text"], out["storage_evidence"]["physical_bytes"])
```

限制：batch=1、无 padding、CUDA、FP16 解码（BF16 权重会在写入前转 FP16）、`pretraining_tp=1`。

### 12.2 复现实验

实验包在 `PhaseKV与Full消融/`。在 Linux 4090 环境：

```bash
cd /root/data

# 单序列解码速度
unzip -o PhaseKV_4090_prompt100_speed_2cases.zip
bash phasekv_4090_prompt100_speed_2cases/start.sh

# 批量吞吐与显存
unzip -o PhaseKV_4090_batch2_6.zip
bash phasekv_4090_batch2_6/start.sh

# 七档 prompt 质量扫描
unzip -o PhaseKV_4090_prompt_quality_scan.zip
bash phasekv_4090_prompt_quality_scan/start.sh

# B 组：固定 128-token 分块，只改变预算分布
unzip -o PhaseKV_4090_block70.zip
nohup bash phasekv_4090_block70/start.sh >> /root/data/block70.log 2>&1 &
tail -f /root/data/block70.log

# B-local：复用 B 的分块预算，由块内 query 投票
unzip -o PhaseKV_4090_block_local70.zip
nohup bash phasekv_4090_block_local70/start.sh >> /root/data/block_local70.log 2>&1 &
tail -f /root/data/block_local70.log
```

[B-local 离线包](PhaseKV与Full消融/PhaseKV_4090_block_local70.zip)仅运行 B-local 的 48 次自由生成，按输入配对复用 B 以及普通 70%、85%、100% 对照。末尾 32 个位置仍受保护。其余位置的分数取自本块内能看见它的 query（含自身），按可见次数取平均，同组查询头求和。softmax 使用每个 query 的完整可见前缀，随后沿用 B 的宽度 7 池化和各块名额。结果目录为 `/root/data/phasekv-results/4090_block_local70_v1/`。完成后在 `summary.json` 查看各任务的 `indexed_gain_vs_block70_pp`（B-local 减 B 的百分点差）。每条记录还包含选位重合率 `mask_overlap_with_B`、计算局部票数的 GPU 时间 `local_vote_gpu_ms` 以及整次首 token 延迟 `ttft_s`。重建命令为 `python workspace_migration/quality_coverage/build_block_local.py`。

[B 组离线包](PhaseKV与Full消融/PhaseKV_4090_block70.zip)仅运行 48 次新的自由生成，历史 70%、85%、100% 结果按输入配对复用。先保留末尾 32 个位置，余下候选区从头按 128 个位置分块，各块按候选长度分配剩余名额，累计配额四舍五入。池化仍在整个候选区进行，再按原分数在块内选择。代码中设置 `model.config.phasekv_prompt_block_size = 128` 可启用同一规则。

结果目录为 `/root/data/phasekv-results/4090_block70_v1/`。同设备、同协议可续跑。完成后看 `summary.json` 中各任务的 `indexed_gain_pp` 和 `fraction_of_70_to_85_gap_recovered`。每份记录的 `position_coverage` 保存 A、B 各层各 KV 头在十个输入区段的实际保留数。重建命令为 `python workspace_migration/quality_coverage/build_future_oracle.py --block-budget`。

重新打包用 [build_prompt100_speed.py](workspace_migration/quick_4090d/build_prompt100_speed.py)、[build_batch_speed.py](workspace_migration/quick_4090d/build_batch_speed.py)、[build_quality_scan.py](workspace_migration/quality_coverage/build_quality_scan.py)。运行结束会打印 `/root/data/phasekv-results/` 下的结果目录，里面有 `summary.json` 和 `run.log`。

测试：完整回归的入口、依赖和分阶段复跑方式见 [optimization_delivery.txt](docs/experiments/optimization_delivery.txt)。注意测试要跑在包含 `eval/` 包的完整工程布局里——本工作区是去掉 `eval/` 的独立快照，直接 `unittest discover` 会在 `PhaseKV/__init__.py` 的 `from eval.compressors import register` 处失败。仓库保存的那次回归记录是 47 项（34 通过、13 跳过、0 失败），见 [release_test_log.txt](results/phasekv_optimization_20261027/release_test_log.txt)。

---

## 13. 术语表

| 术语 | 意思 |
|---|---|
| prefill | 一次性处理整段 prompt 的前向 |
| decode | 逐 token 生成的前向 |
| 读放大 | 同一份 KV 在整个解码过程中被读了多少次 |
| 热区 / 热层 | 最近生成的 token，FP16，环形缓冲 |
| 冷区 / 冷层 | 滑出热区后保留下来的生成 token，4-bit 打包 |
| 驱逐 | 把选不中的 token 物理丢掉 |
| 懒批 | 攒够一批再统一做选择+量化，而不是每步都做 |
| 打包 | 把 4-bit 码真实塞进 uint8，而不是回写成 FP16 |
| 描述表 | GPU 上的指针表，告诉 kernel 每段 KV 在哪、怎么解码 |
| partial | 一次 launch 内对某一段算出的 (max, 分母, 累加)，留给第二次 launch 合并 |
| GQA | 多个 query head 共享一个 KV head |
| sink | 强制保留的开头几个 token |
| 观察窗口 | 用最后若干个 query 的注意力给 prompt token 打分 |
| Fake-quant | 量化后立刻反量化回 FP16，只用于早期精度评估，不省显存 |

---

**三句话总结**：

1. 省内存的主力是 **4-bit 真实打包（约 3.2×）**；丢 prompt token 是在此之上继续加码，也是质量风险的主要来源；冷区 3/16 驱逐的独立代价没有干净对照（见第 10 节第 7 条）。
2. 省时间的前提是 **KV 读取占比够高**：长 prompt、batch ≥ 2；batch 1 短 prompt 可能持平甚至更慢。
3. 长输出任务对 **prompt 保留率**极其敏感：70% 会崩，95%–100% 才回到 Full 的水平线；LongBench/RULER 的“接近”不能外推。
