PhaseKV 新工作空间：四组 Llama 注意力消融的离线迁移包

先在 CPU 实例准备，再给同一工作空间挂两张 A800 80 GB。切换前确认平台可保留
或重新挂载数据卷，并确认实际挂载位置。以下示例使用 /root/data。GPU 阶段仍需 Linux
x86_64、glibc >= 2.35、CPython 3.10、PyTorch 2.5.1 CUDA 12.1 ABI=False、
gcc 和 Python.h。CPU 阶段需要同架构、同 glibc 要求的 Python 3.10 与 pip。
数据盘建议至少留 50 GB 可用空间。若平台只能新建实例，挂回同一数据卷和路径即可。

推荐在截图中选择 pytorch / 2.5.1-cuda12.2-python3.10-ubuntu22.04。
打开该镜像自带的 Python 环境，先检查卷挂载点、解释器、torch 版本和 ABI：
   findmnt -T /root/data
   df -h /root/data
   python -c "import sys,torch; print(sys.executable, sys.version_info[:2], torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"
CPU 上无需显卡即可运行这个检查。预期 (3, 10)、2.5.1（可带 +cu121）、12.1、False。
CPU/GPU 两个实例都应挂同一个 128 GB 数据卷，GPU 阶段再核对挂载点。

CPU 阶段
1. 将原离线包、校验文件、CPU 分阶段工具补丁上传到持久盘 /root/data。先解压原包，再解压
   补丁。补丁只更新安装器、并行启动器和本文。
   cd /root/data
   sha256sum -c PhaseKV_new_workspace_cp310_cu121.zip.sha256
   unzip PhaseKV_new_workspace_cp310_cu121.zip
   unzip -o PhaseKV_volume_tools.zip
2. 上传或挂载本机 weights/Llama-3.1-8B-Instruct/snapshots/master 到
   /root/data/Llama-3.1-8B-Instruct/snapshots/master。需要四个 safetensors 分片和该目录的
   config/tokenizer/index 等 JSON，无需上传 original/ 权重副本。
3. 用 CPU 镜像的 Python 3.10 在本地数据盘准备项目和离线依赖，并检查模型：
   python /root/data/migration/install.py --prepare-cpu --workspace /root/data/phasekv-ws --model /root/data/Llama-3.1-8B-Instruct/snapshots/master
   如果镜像的 torch 检查不符合上述值，或目标 GPU 镜像不能提供相同构建，请在 CPU 阶段
   给准备命令加 --with-cu121。它需要联网，从官方索引下载数 GB 的现成 wheel 到
   卷上，并在 CPU 上检查版本和 ABI。GPU 阶段自动优先使用该运行时。
   末尾显示 CPU preparation complete 即可切换。这个阶段不探测显卡。若权重仍在上传，
   可先省略 --model 执行环境准备，上传完成后追加 --model 重跑，只补做检查。
   模型检查覆盖分片索引、文件大小和 tokenizer/config 哈希。

GPU 阶段
4. 挂回 /root/data，选带 GPU 版 PyTorch 的匹配镜像，执行：
   python /root/data/migration/install.py --check-gpu --expected-gpus 2 --workspace /root/data/phasekv-ws
   此步把已准备的依赖接入 GPU 镜像，并逐卡运行 FA2 测试。CPU/GPU 实例可以是
   不同机器，无需搬运绑定 CPU 系统路径的虚拟环境。
5. 两张 A800 并行开始正式测试：
   cd /root/data/phasekv-ws/official_llama
   nohup ../venv/bin/python -u workspace_parallel.py --gpus 2 --parallel-performance --out /root/data/phasekv-results/a800_2_parallel > /root/data/a800_2_parallel.log 2>&1 &
   tail -f /root/data/a800_2_parallel.log
   每卡先做完整正确性检查。性能与质量阶段按输入分片，每张卡一次只跑一个任务，
   同一输入四种方法和三次速度测量固定在同一设备上。速度阶段结束后即可看
   performance_first/ablation_pairs.json。多卡同时计时可能受主机负载和功耗影响，
   运行清单会记录该模式，应与独占整机的计时分开解释。

结果：总目录 summary.csv、ablation_pairs.json、records.jsonl；设备日志在 lanes/<编号>/。
重复相同正式命令可续跑；已有成功记录跳过，OOM 保留，--retry-failed 可重试失败。
硬件、驱动、代码或分片变化时拒绝混批。OOM 会反映到最终状态/退出码中。
本机仅完成离线包与调度逻辑检查；目标硬件上的可运行性、32K 容量和速度以实测为准。
