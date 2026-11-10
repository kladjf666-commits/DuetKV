# A800 TTFT 同卡复测

只复测两个点：`8192+256 / B1`、`32768+256 / B8`。比较同一张A800、同一个已加载模型中的优化前Prompt90路径（baseline）和BetterMemory路径（optimized）。

核心算法、选择规则、量化规则和解码内核保持与已保留源码一致。Baseline沿用原Prompt90运行适配，使用同一份缓存类的默认非消费路径；optimized使用已有的memory_prefill和消费式打包。没有新增算法优化。

## 测量设计

- 参数：Prompt90%、冷块目标80%、Prompt/冷KV均INT4、recent128、obs32、group32、lazy16、sink0。
- 两个点每条路径预热1轮、正式3轮：首次完整运行共4轮预热、12轮正式生成。
- 每轮A/B交替先后顺序，两个场景合计各有3次由baseline先运行、3次由optimized先运行。
- 统一强制输出256 token，固定工程输入；本次不评估任务质量。
- 两条路径的正式计时调用原有 `engine.run`，传入普通内存字典。状态保存、检查点写盘、硬件快照、输出哈希和报告均在测量区间外。
- 两条路径都维持原有的轮间释放及 `empty_cache` 策略。这一策略没有被只针对某一条路径修改。
- 每组A/B都成功且输出哈希一致后，才一次性保存该组；中断后重跑未保存的完整配对，不将不同进程中的两个半组拼起来。
- 每次进程启动执行B1/B8正确性检查，包括首步logits、Prompt打包内容、164-token生成一致性及冷却/热环检查。

## 独立分段诊断

全部正式生成完成后，每个场景、每条路径各补1次只运行预填充和首token的诊断，共4次。诊断数据不计入正式TTFT、速度和峰值统计。

诊断临时包装运行入口，记录CPU调用耗时和CUDA流时间区间，正常结束和异常退出都会恢复方法。没有在每层插入GPU同步。

| 阶段 | 含义 |
|---|---|
| `prefix_forward` | 512-token分块前缀前向，逐调用记录并汇总。 |
| `observation_forward` | 最后32-token观察窗口前向；optimized内部包含位置选择。 |
| `selection` | 位置选择本身。baseline整批调用一次、内部遍历32层；optimized逐层调用32次。 |
| `observation_and_selection` | 可比的观察窗口加选择总区间；baseline相加，optimized使用已包含选择的观察前向区间。 |
| `flatten` | 将Batch映射到KV头布局。 |
| `packing` | 选中KV收集、校验、量化打包及相关释放。 |
| `first_token` | 根据logits选取首token。 |

`selection`已包含在`observation_and_selection`中，不要重复相加。CUDA流区间可能包含CPU提交间隙，不等于纯GPU核执行时间；CPU调用耗时也不等于同步后的GPU耗时。单次诊断用于定位阶段，不替代三轮正式统计。

GPU频率、功耗、温度、利用率及CPU负载、CPU亲和性采集于每次调用前后；它们是前后快照，不是计时区间内的连续遥测。检查点写盘耗时另外保存在`checkpoint_io/`，不计入TTFT。

## 运行与输出

复用服务器原配置、模型和环境，不安装新依赖：

- 默认配置：`/data/phasekv-results/a800_quality_fulltext_v1/config.json`
- 默认新输出：`/data/phasekv-results/a800_ttft_ab_v1`
- 默认GPU：0；`--gpu`可以指定其它空闲A800 80GB。
- 只允许输出目录名以`a800_ttft_ab_`开头，避免误写旧结果目录。
- 检查到选中GPU已被其它计算进程占用时停止，不终止其它进程。
- `run.sh`支持`prepare`、`run`、`status`、`export`。已经完整完成时再次运行不会加载模型或重测。
- 恢复运行必须使用同一物理GPU及同一输入/代码/权重身份。每个已提交配对始终来自同一进程；中断恢复可能使整次实验涉及多个进程会话，报告会列出会话数量。

```text
输出目录/
├── summary.txt / summary.json    正式A/B结果与单独的诊断汇总
├── pairs/<场景>/r1..r3.json      6个配对文件，内含12条正式测量
├── diagnostics/<场景>.json       2个诊断配对文件，内含4次预填充分段测量
├── warmups/<会话>/<场景>/        预热记录，不计成绩
├── gates/<会话>.json             本次进程的正确性检查
├── checkpoint_io/<会话>.json     计时区间外的写盘耗时
├── protocol.json / plan.json     模型、参数、环境及源码身份
├── gpu.json / state.json         GPU身份及当前状态
└── run.log                      运行日志
```

正式TTFT/吞吐取三轮中位数，峰值取三轮最大值；同时保存每轮配对TTFT变化百分比及其中位数。负变化表示optimized更快。相比历史结果时，应先使用这里的同卡A/B结论。

本地可验证控制逻辑、输入身份和小模型数学一致性；A800硬件计时、实际显存峰值及CUDA事件采集仍需服务器运行后确认。`validation.json`记录本地验证范围。
