PhaseKV 实验图集
================
8张图，每张PNG（300 dpi）、PDF和SVG矢量版本。
PDF嵌入字体，SVG文字转路径；绘图数值与来源哈希见plot_data.json。

01_prompt_retention_and_bits：Prompt保留率与位宽消融
来源：2026-10-27 prompt复测；同一批256篇文档 × 4种输入预算，按任务等权。
折线纵轴局部放大；均值差异不等于统计显著。Full-matched采用一致prefill路径。

02_decode_parameter_screening：热区与冷区参数筛选
来源：2026-10-26 decode筛选；每配置4任务 × 8篇 × 2种预算，其中摘要2任务。
小样本探索，不与正式复核分数混算；QA输出普遍较短，不能据其不变推断冷区参数没有影响。

03_quality_deltas_with_uncertainty：六组配置的配对质量区间
来源：2026-10-27 prompt复测；同一文档的4种预算一起重采样，未校正多重比较。
区间跨0时不能断言提升/下降；均值损失小于1分，不等于95%置信度下损失也小于1分。

04_phasekv_speed_and_kv_by_workload：70%/4-bit的分场景速度与KV
来源：2026-10-27 prompt复测；每形状3次正式测量，速度与KV取中位数，预热不计入。
左侧工作负载标签为“输入 + 输出 token”；decode不含prefill，但包含缓存维护。

05_algorithm_quality_comparison：五算法质量对比
历史五算法批次：同模型、同数据/输入，PhaseKV单独复跑；各方法prefill路径不完全相同。
PhaseKV=P30/B2/W32/C4/D1；KIVI=2-bit/R64；SnapKV=P30%；SCOPE=ALLKV+Slide/生成预算32。
来源：2026-10-25对比报告；每方法1024条质量记录，QA与摘要分别按任务等权；不含P5质量数据。

06_algorithm_long_context：06  长输入：五算法速度与 KV 占用
历史五算法批次：同模型、同数据/输入，PhaseKV单独复跑；各方法prefill路径不完全相同。
PhaseKV=P30/B2/W32/C4/D1；KIVI=2-bit/R64；SnapKV=P30%；SCOPE=ALLKV+Slide/生成预算32。
每点为3次正式测量的中位数；速度误差棒为实测最小—最大值，非置信区间。

07_algorithm_long_output：07  长输出：五算法速度与 KV 占用
历史五算法批次：同模型、同数据/输入，PhaseKV单独复跑；各方法prefill路径不完全相同。
PhaseKV=P30/B2/W32/C4/D1；KIVI=2-bit/R64；SnapKV=P30%；SCOPE=ALLKV+Slide/生成预算32。
每点为3次正式测量的中位数；速度误差棒为实测最小—最大值，非置信区间。

08_optimized_phasekv_and_other_algorithms：优化前后PhaseKV及其他算法
本图33.49→37.05为同配置A/B；图07的40.66来自不同配置，不能连成优化前后变化。P5未开启独立GQA复用。P5实际KV增加2.71 MiB，来自空闲页容量。
历史参考：KIVI 2-bit/R64、SnapKV保留30%、SCOPE ALLKV+Slide/预算32；未做等KV预算或等质量配平。
右两图只能作跨批次参考，不能据此确认P5优于其他算法。P5未测准确率；两版输出从第92个token开始分叉。

没有纳入PyramidKV。历史PhaseKV参数与70%/4-bit复测不同，已在各图标注。
第08图只有优化前/后是最新同轮实验；其他算法为历史参考，不是同轮排名。
没有P5质量数据，未补画准确率；没有逐项P1—P5 GPU结果，未虚构加速消融。
复现：项目根目录运行 .venv/Scripts/python.exe Figure/generate_figures.py。
依赖Matplotlib、NumPy；Windows中文字体使用Microsoft YaHei。
