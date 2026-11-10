PhaseKV Prompt90 / Decode80 完整输入质量 + 速度测试包 v1

本轮只测用户选定的这一组参数，不继续选参，不运行其它算法。
Prompt KV 保留 90%；生成冷 KV 目标保留 80%；Prompt/冷 KV 均 INT4；
最近 FP16 窗口 128；观察窗口 32；量化 group_size=32；lazy_g=16；sink=0。
沿用已验证的 A800 MMA、GQA 复用、热环形缓存等开关。
生成冷块按 16 个选 13 个取整，最近 128 个及未满块尾部仍保留 FP16；
因此“生成保留80%”不是所有生成 token 最终恰好剩 80%。
内部来源名 J_p90_w128_o32 只用于核对上一轮结果，界面和汇总显示实际参数。

测试范围
1. 新推理：320 条 LongBench，10 个任务各 32 条，使用完整原始 prompt，绝不裁剪。
   narrativeqa/qasper: cap128；multifieldqa_en:64；hotpotqa/2wikimqa:32；
   gov_report/qmsum/multi_news:512；lcc/repobench-p:64。
   这是上一轮固定评估子集，不等同于官方 LongBench 全量数据。
2. 新推理：64 条 RULER32K，niah_multikey_1/niah_multivalue/vt/cwe 各16条，
   原输入和停止条件不变，生成 cap 依次128/128/30/120。
3. 直接复用：上一轮细化测试中完全相同参数的48条 LongGenBench，
   gsm8k/mmlu/csqa各16条，cap4096。原文件逐字节核验后复制进新结果目录。
   它们参与过选参，汇总明确标记为复用的选参集，不作为新的独立验证集。
4. 速度：输入8192/32768 × 输出256/2048 × batch1/2/4/8，共16点。
   每点1次预热、3次正式计时，报告中位数；固定输出长度，不因EOS提前结束。
   使用8个不同输入前缀；真实多batch前向，不用逐请求循环伪装吞吐。
质量全部B1、贪心生成、自然EOS、原生成上限；本轮合计整合432条质量记录。

速度实现与口径
旧多batch测速曾固定Prompt保留100%。本包单独适配按请求/头选择90%，
保留原始位置，沿用既有多batch解码路径。FA2前缀以512-token块处理，
最后32-token观察窗口使用eager注意力完成选择；选择和量化时间计入TTFT。
质量仍调用上一轮相同的B1完整输入实现，不改变已复用LongGenBench的语义。
正式测试前会核对实际保留数/位宽/绝对位置，并验证B1/2/4/8请求隔离，
B8长解码经过冷却和环形窗口，以及分段注意力对反量化KV参考的误差。
本地已做CPU检查；实际CUDA门禁要在A800执行，失败会停止，不自动降级。

运行顺序
加载一次模型 → 质量门禁 → 10条LongBench检查样本 → 多batch门禁 →
16个速度点 → 其余310条LongBench → 64条RULER。
10条检查样本本身计入320条，不重复运行。OOM如实保存，不缩短输入补成绩。
运行错误会停下并保存错误现场；不要删结果后盲目重跑。

服务器前提
单张空闲A800 80GB，保留原/data。无需上传模型、环境、数据或重新编译安装依赖。
默认读取：
  /data/phasekv-results/a800_quality_fulltext_v1/config.json
  /data/a800_quality_fulltext_v1
  /data/phasekv_a800_4gpu_v1
  /data/a800_four_gpu_v1
  /data/phasekv-results/a800_four_gpu_v1
  /data/phasekv-results/a800_phasekv_refine_v2
模型/离线main环境路径由旧config读取。模型元数据、文件大小/时间戳、
环境版本、历史结果指纹均会核对。准备失败时尚未加载模型，不会自动安装任何东西。
不要解压覆盖旧测试包；本包独立使用新目录。

上传与启动
把 a800_phasekv_p90_final_v1.tar.gz 及同名 .sha256 上传到 /data，然后执行：

cd /data
sha256sum -c a800_phasekv_p90_final_v1.tar.gz.sha256 && tar -xzf a800_phasekv_p90_final_v1.tar.gz
python3 /data/a800_phasekv_p90_final_v1/runner.py prepare

确认 prepare 成功，再后台启动：

nohup bash /data/a800_phasekv_p90_final_v1/run.sh run --gpu 0 > /data/a800_phasekv_p90_final_v1/launcher.log 2>&1 &

查看状态和运行日志：

python3 /data/a800_phasekv_p90_final_v1/runner.py status
tail -n 40 /data/phasekv-results/a800_phasekv_p90_final_v1/run.log
nvidia-smi

状态含已完成数量、当前样本/阶段、粗略剩余时间和北京时间ETA。
ETA由旧同样本耗时估计，完成至少3条后按当前速度校准；不含加载/门禁/JIT，
早期仅供参考。state更新时间很久未变时，还需结合进程和日志判断是否在运行。

断点续跑
关机后保留原/data，重新执行上面的 nohup 启动命令即可。
已完成质量样本、已复用48条、已完成速度点和每点已保存的正式轮次均跳过。
中断中的一条/一轮没有提交结果则重做；恢复未完成速度点会重新预热一次。
OOM为终态，不自动重跑。错误结果需查明原因后处理，脚本不会用新结果覆盖它。
每次启动检查GPU是否空闲并加进程锁，重复启动不会抢占当前任务。
续跑绑定首次记录的GPU UUID；如果更换物理卡需先审查测速可比性，
不要手工改结果文件。欠费暂停后原实例/原卡恢复通常可直接续跑。

结果位置与下载
所有新结果独立保存在：
  /data/phasekv-results/a800_phasekv_p90_final_v1/
summary.txt / summary.json：按数据集质量、配对差值、16点速度和进度。
results/phasekv/quality：384条新质量原始记录，含输入hash、生成文本、得分、KV。
reused/longgenbench：48条来源原样副本；plan.json记录来源hash和选参集属性。
results/phasekv/speed：16点正式汇总；speed_runs：逐轮断点。
warmups：预热记录；gates：验证证据；protocol.json/profile：实际配置证据。
KV为最终真实缓存存储字节，包含量化元数据/预留空间，不是理论估算。
peak_allocated_bytes为PyTorch分配峰值，另有prefill和decode峰值；不等同nvidia-smi整进程占用。
aggregate_tokens_per_s是解码总吞吐，per_sequence_tokens_per_s是每请求解码速度；
TTFT包含预填充/选择/量化及首token，解码速度分母不包含TTFT。
历史Full和旧PhaseKV仅供参考：跨时段/可能跨GPU，并非同卡严格加速比，
旧PhaseKV速度使用100% Prompt，质量使用95% Prompt，不能当成本轮90%的结果。
缺失/OOM不记零分；分数显示有效样本覆盖数，质量按任务宏平均。

完成后打包下载（CPU操作，不跑模型）：

python3 /data/a800_phasekv_p90_final_v1/runner.py export

输出包在 /data/phasekv-results/a800_phasekv_p90_final_results_时间戳.tar.gz，
下载该文件和同名.sha256即可，包含新结果、48条复用副本和测试包。
确认状态 COMPLETE 后可以停止租用GPU；打包不需要GPU。
