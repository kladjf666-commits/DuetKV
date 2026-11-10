RTX 4090 D | PhaseKV 快速解码测速

同一张卡，同一个冻结提示 performance/8192+256/0。Llama-3.1-8B-Instruct 的 32 层全部参与，FP16，batch=1。Full 使用 FA2，PhaseKV 使用通道连续 K、4-bit prompt/cold、70% prompt、80% 生成冷区（每 16 个保留 13 个）。两个 PhaseKV 配置只切换 GQA 复用。

把压缩包上传到已配置好的 Linux 4090D 工作区数据盘，在包所在目录执行：
  unzip -o PhaseKV_4090D_speed_quick.zip
  bash phasekv_4090d_speed/run.sh

程序自动查找当前数据盘、/root/data 或 /data 的模型及既有离线环境。自定义安装可在启动命令前指定：
  PHASEKV_PYTHON=/path/to/venv/bin/python PHASEKV_MODEL=/path/to/checkpoint bash phasekv_4090d_speed/run.sh

需要单张可见的 RTX 4090 D。多卡机器可通过 CUDA_VISIBLE_DEVICES 选择一张空闲卡。使用已准备好的 PyTorch 2.5.1+cu121、Transformers 4.46.3、Triton 3.1.0、flash-attn 2.7.4.post1 与本地权重。本包自带本次测速代码和输入，只需解压至任意可写目录。

模型只加载一次。每种方法完整预热一次，然后默认交错测三轮，每轮三种方法各一次。计时覆盖首 token 之后的 255 个解码步，包含模型投影、注意力、缓存更新和贪心选词。每次以固定输出长度结束，预填充不计入 token/s，计时期间不启用 profiler。每轮结束后立即打印 PAIRED 行。只需一次初筛时可加 --rounds 1。

输出位于包旁的 phasekv-results/4090d_speed_时间_进程号/。run.log 保存终端输出；summary.json 保存每轮原始速度、输出哈希、GPU、配置、配对吞吐变化和胜出轮数。通常等到 FINISHED 再读取最终汇总。两种 PhaseKV 的输出哈希需要一致，Full 可不同。单提示快速测量只代表本形状。
