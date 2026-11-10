A800 四卡接续：完整原文质量 + 四形状测速
2026-11-05

用途
继续使用原服务器四张 A800 80GB、现有模型和三个离线环境。
无需安装依赖、上传权重或重新编译 FlashAttention；复用原冻结算法实现。
本包不修改原 benchmark 的源码/manifest，不删除任何已完成结果。
本地已进行 CPU 数据校验和模拟调度测试；真实 Linux 进程接续和 GPU 推理仍须现场运行。

上传与启动
将 A800_quality_fulltext_v1.tar.gz 及同名 .sha256 文件上传至 /data。

cd /data
sha256sum -c A800_quality_fulltext_v1.tar.gz.sha256
tar -xzf A800_quality_fulltext_v1.tar.gz
python3 /data/a800_quality_fulltext_v1/recover.py prepare
nohup bash /data/a800_quality_fulltext_v1/run.sh > /data/a800_quality_fulltext_v1/launcher.log 2>&1 &

prepare 只校验本地文件、重建配置、复制已有完整输入的结果，不动进程、不占 GPU。
run.sh 再做幂等准备，然后接管原队列。看到配置/数据不匹配会拒绝，不能绕过检查。
复跑同一个 run.sh 即可续跑，不用换输出目录；完成的 ok/OOM/unsupported 不重跑。
如果只想检查十题先导，末尾加 --mode pilot；之后去掉此参数可继续正式队列。
先导检查运行是否正常，不是统计质量保证，也不自动根据分数调参数。

查看
python3 /data/a800_quality_fulltext_v1/recover.py status
tail -n 40 /data/a800_quality_fulltext_v1/launcher.log

新状态目录：/data/phasekv-results/a800_quality_fulltext_v1
summary.txt/json 包含完整原文质量、其他质量及新测速矩阵的完成/OOM/错误/待跑数量。
logs/<stage>-<method>.log 是逐算法输出。
不要再启动原 campaign.py、append.py；接管期间会持有其队列锁。
原 ETA 和 summary 对应旧 6016/850/950 点矩阵，不能作为新队列进度。

切换运行中任务
1. 准备和验证通过后，按脚本绝对路径、输出目录和 PID 启动时间识别旧调度器。
2. 冻结旧调度器，禁止继续派发旧任务；中止包含旧 LongBench 的质量进程。
3. 只终止已经识别并冻结的调度器父进程；保留独立运行的有效 worker。
4. 旧测速进程完成并保存当前配置点后结束，最多可能开始下一点但不继续旧矩阵。
   已保存点不会丢失；新矩阵中匹配条件的点跳过。原 RULER/LongGen 进程允许完成。
5. 释放该 GPU 的旧锁后，新队列接手；遇到不认识的占卡进程拒绝，不杀陌生进程。
Ctrl+C/终止新调度器会停止它自己启动的进程，逐条已保存结果保留。
接续日志与旧结果完整性快照保存在 adopted_workers.json、preserved/、preservation_check.json。
没有远程自动执行权限，本包上传并由你启动后才会改变服务器状态。

测速：共 104 个配置点，不再遍历 17 个质量任务组
8192+256、8192+2048、32768+256、32768+2048。
Full/PhaseKV/KIVI/SnapKV/PyramidKV/SCOPE：B=1,2,4,8，各 16 点。
ClusterKV/MILLION：原生适配器 B1，各 4 点；B2/4/8 不伪装成批量支持。
每点预热 1 次 + 正式 3 次；固定生成，无 EOS 早停。
报告 TTFT、decode 总吞吐、每序列吞吐和峰值显存。
256 个输出含 prefill 给出的第一个 token，decode 为 255 步；2048 对应 2047 步。
按原始 frozen speed_rows 的同一前缀构造输入，8 个 batch 行互不相同。
旧 9192+256、真实任务组、B3/5/6/7 等记录均保留，但不混入新 104 点汇总。
新测速仍写原 results/<算法>/speed，已有同输入/参数/GPU/三轮记录会复用。
各算法测速继续固定原物理 GPU；尚未开始的算法在空闲卡领取并固定。
四卡并发会共享 CPU/内存/PCIe；与 Full 的跨卡速度比明确标记，不称同卡加速比。

质量：每算法 432 条，八算法 3456 条（已有合格结果计入并跳过）
LongBench：原先选中的 320 篇独立原文，每篇一次；不是全量 LongBench 官方测试集。
10 个任务各 32 篇，不再重复 8K/16K 两种输入预算。
完整 prompt/chat 模板，输入 498–65401 token；不删除原文头、中、尾任何内容。
每条保存原始 ID/索引、原文和 prompt 哈希、最终输入 token 哈希。
274 篇有未截断的旧输入别名；仅在输入、权重、代码、环境和 gate 匹配时复用旧成功/OOM。
旧结果原样保存，复制到新目录的记录带来源文件/协议哈希，不把旧截断结果改名当新结果。
LongGenBench 48 条、RULER 32K 64 条保持已有完整输入和评分，原目录断点续跑。
保留原始生成上限和 EOS；“完整输入”不等于无限生成。
SCOPE 不修改，实际 OOM 单列；ClusterKV 超过已验证 512 簇限制记 unsupported。
OOM/unsupported/error 不计零分、不缩短输入、不降低 batch 后混入同一配置。
仅成功均分可能有覆盖率偏差，应同时看成功数和与 Full 同题成功配对差，不能只看均分排名。
前置十题选自 320 篇，已保存结果复用；八算法先导运行无错误后，优先新测速，再完整质量。

质量多 batch 评估：本次放弃，四卡各 B1
现有 Engine.run 明确拒绝 quality B>1，停止逻辑对单个 token 使用 item()。
质量题目长度不同，需要 padding mask、逐行绝对位置、独立 EOS/生成上限和缓存收缩。
当前等长固定步数测速的 batch 验证不能替代上述质量验证。
PhaseKV 质量分段核要求 hot KV batch=1；ClusterKV/MILLION 适配器也只支持 B1。
320 条按相同输入长度/输出上限/停止集合分组，只能得到 4 个 B2 对，0 个 B4 组。
这不是 GPU 显存不够的判定，而是现有实现和实际收益不足，不为它新改算法或租卡。
