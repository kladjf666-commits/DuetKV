# Qwen2 模型集成

本目录保存 Qwen2/Qwen2.5 架构共用的模型入口和注意力适配器。六方法比较使用各具体模型目录中的 `run.py`，配置和结果按完整模型名隔离，参见[统一部署说明](../README.md)。

本目录的历史交互入口 `qwen2.py` 专用于 PhaseKV，仍默认加载 `Qwen/Qwen2-1.5B-Instruct`；它不负责六方法比较。

| 文件 | 用途 |
| --- | --- |
| `qwen2.py` | 模型加载、默认参数、采样、多轮对话和生成循环 |
| `qwen2_attention.py` | Qwen2 注意力适配、RoPE 和 PhaseKV 缓存写入 |
| `kivi_attention.py` | Qwen2 的真实打包 KIVI 注意力适配 |
| `packed_runtime.py` | KIVI/PhaseKV 打包缓存的生成、存储核验和触发记录 |
| `integration.py` | full、KIVI、SnapKV、SCOPE、PyramidKV、PhaseKV 统一调度 |
| `bench_phasekv.py`、`ablate_prompt.py`、`ablate_cold.py` | 从旧 scripts 迁入的 Qwen2 专用实验实现 |
| `__main__.py` | `python -m qwen2` 命令入口 |
| `__init__.py` | 包初始化，兼容旧脚本读取 `qwen2.MODEL_ID` 等入口属性 |

## 运行

在项目根目录执行：

```bash
.venv/bin/python -m qwen2 --check-cuda
.venv/bin/python -m qwen2 --local-files-only --max-new-tokens 512
```

也支持直接执行迁移后的文件：

```bash
.venv/bin/python qwen2/qwen2.py --local-files-only --max-new-tokens 512
```

从其他工作目录启动时，可以使用解释器和入口文件的绝对路径：

```bash
/home/kladjf/桌面/kvcomp-astra/.venv/bin/python /home/kladjf/桌面/kvcomp-astra/qwen2/qwen2.py --local-files-only
```

模型权重通过原有 Hugging Face 缓存或 `--model /path/to/model` 读取，不放入本目录。原根目录的 `python qwen2.py` 命令已改为上面的新入口。

## Python 导入

新代码使用明确的模型模块路径：

```python
from models.Qwen2 import qwen2 as entry
from models.Qwen2.qwen2_attention import install_qwen2_phasekv_attention
from PhaseKV import PhaseKVCache
```

旧的 `import qwen2` 属性读取和 `from PhaseKV import install_qwen2_phasekv_attention` 保留兼容；配置修改、函数替换等操作应针对实际入口模块 `models.Qwen2.qwen2`。旧的 `PhaseKV.qwen2` 与 `KIVI.qwen2_attention` 模块转发到本目录，不再保存重复实现。

## 后续模型目录约定

后续新增架构放在 `models/<架构名>/`，具体权重版本放在 `models/<完整模型名>/`。架构目录包含：

```text
models/<family>/
├── __init__.py
├── __main__.py             # 历史交互入口（如有）
├── <model>.py              # 模型入口、配置、加载与生成
├── <model>_attention.py    # 模型专用注意力适配
└── README.md              # 支持范围、依赖和运行说明
```

通用的缓存管理、量化、参数规划和 Triton 内核保存在各算法目录（例如 `PhaseKV/`、`KIVI/`），由模型适配器调用。`src/phasekv/` 保留历史转发接口。新架构通过 `models/registry.py` 显式注册，具体模型的配置格式见上级部署说明。

模型专用实验实现归入模型目录，通用评测和测试保留在 `eval/`、`scripts/`、`tests/`。每次迁移或新增模型时，同步维护命令、导入、源码路径记录和文档；命令行与CPU检查不能替代后续GPU数值验证。
