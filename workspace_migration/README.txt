PhaseKV 新工作空间：四组 Llama 注意力消融的离线迁移包

选择同一台机器上的独占同型号 GPU。建议先用一个 48 GB 或以上设备做正确性检查，
正式跑全套优先 2 个相同设备；4 个是时间更紧时的选择。计费以平台实际报价为准。
标准 A16 一块物理板含 4 个独立的 16 GB GPU；租 8 个可见设备并不会让一个
模型获得 128 GB 显存。本入口不拆模型，16 GB A16 不推荐用于本轮完整比较。

前提：Linux x86_64、Ubuntu 22.04 或兼容 glibc >= 2.35、CPython 3.10、gcc、
Python.h、可工作的 NVIDIA 驱动、预装 PyTorch 2.5.1 CUDA 12.1 且 CXX11 ABI=False。
可先用 nvidia-smi -L 和 nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv
核对平台的“一张”实际对应一个 16 GB 设备还是含四个设备的物理板。

包中有三份校验过的项目压缩包（原生代码、冻结数据/评分器、四组消融）、已实测过的
FlashAttention 2.7.4.post1 Linux wheel、Triton 3.1.0、Transformers 4.46.3 等二进制依赖。
不含模型权重、系统驱动、torch/CUDA 基础镜像，不支持旧实验的 KIVI 扩展。
不从旧机器拷贝 Triton 缓存；运行时按实际 GPU 首次 JIT。解包后不需要联网或 nvcc
编译 FlashAttention。请使用新目录，安装器不会覆盖旧工作区或安装到镜像原环境。

1. 选择上面的基础镜像。上传本 zip 和同名 .zip.sha256 到 /data，运行：
   cd /data
   sha256sum -c PhaseKV_new_workspace_cp310_cu121.zip.sha256
   unzip PhaseKV_new_workspace_cp310_cu121.zip
   python /data/migration/install.py --workspace /data/phasekv-ws
   （python 必须是镜像中能 import CUDA torch 的 Python 3.10。）

2. 挂载/上传与旧实验完全相同的权重目录到：
   /data/Llama-3.1-8B-Instruct/snapshots/master
   所需文件：四个 model-*-of-00004.safetensors、index JSON、config/tokenizer 相关
   JSON。老数据盘可复用；另需上传时用本机 weights/Llama-3.1-8B-Instruct/snapshots/master。
   不需要 nested original/ 权重副本。路径不同时给运行命令追加 --model 实际路径。
   建议数据盘预留至少 50 GB 可用空间（含权重与运行缓存），两 GPU 配置宜有 64 GB
   以上主机内存；这两项是部署建议，不是实测硬下限。

3. 建议先短租单个设备：先跑正确性门禁，再测四组各一个冻结的 8K/16K/32K + 256
   样本，查看 capacity.json 的 12 项 OK/OOM 与峰值。此试验不产生正式速度数据，
   也不保证更长输出一定装得下。若只需要数值门禁可换成 --gate-only。
   cd /data/phasekv-ws/official_llama
   /data/phasekv-ws/venv/bin/python workspace_parallel.py --gpus 1 --capacity-only --out /data/phasekv-results/hw_single_pilot

4. 两个同型号 CUDA 设备的正式测试（新输出目录，不混旧结果）：
   cd /data/phasekv-ws/official_llama
   nohup /data/phasekv-ws/venv/bin/python -u workspace_parallel.py --gpus 2 --out /data/phasekv-results/hw_pair_2 > /data/hw_pair_2.log 2>&1 &
   tail -f /data/hw_pair_2.log
   已分配的 GPU 由调度器设定时，保留 CUDA_VISIBLE_DEVICES；不擅自使用其他租户的设备。
   每个设备先跑门禁。性能全套独占第一个设备，其他设备在此期间空闲；避免计时受到
   并发负载干扰。之后质量样本在各卡并行，同一输入的四种方法固定在同一 GPU。
   这是数据并行（每卡一份模型），不是显存池化或张量并行。
   如果由 2 改 4，需使用不同 --out；本协议将 GPU UUID、型号、显存、驱动、分片方案
   和代码哈希纳入清单。每个设备有独立门禁、日志、JIT 缓存、诊断与结果目录。

结果：总目录 summary.csv、ablation_pairs.json、records.jsonl；设备日志在 lanes/<编号>/。
重复相同正式命令可续跑；已有成功记录跳过，OOM 保留，--retry-failed 可重试失败。
硬件、驱动、代码或分片变化时拒绝混批。OOM 会反映到最终状态/退出码中。
本机仅完成离线包与调度逻辑检查；目标硬件上的可运行性、32K 容量和速度以实测为准。
