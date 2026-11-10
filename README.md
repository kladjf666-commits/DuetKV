# PhaseKV / A800 最终项目

当前保留优化前Prompt90的完整最终结果，以及BetterMemory本轮的速度与峰值显存结果。

- `Code-Original-NoChanging - A800/`：优化前A800核心源码。
- `Code-Original-NoChanging - A800 - BetterMemory/`：峰值显存优化后的源码。
- `A800最终测试/优化前_Prompt90/`：432条质量记录、16点速度与显存、报告和图表。
- `A800最终测试/BetterMemory_速度与峰值显存/`：16点、48正式轮速度与峰值、16预热及实验身份。
- `A800最终测试/多算法最终对照/`：Full等算法最终质量、速度、显存原始记录和覆盖状态。
- `A800复现资料/`：必要输入、评分器、实际适配源码；不是上传包。
- `参考源码与设计说明/`：收拢的早期手写源码和设计说明。
- `weights/`、`.venv/`、`.git/`、`config.txt`：模型、开发环境、Git历史和连接配置，默认保留。

BetterMemory原16点保留速度和峰值显存，不保留最终KV及生成文本。旧Prompt90速度不得改标成BetterMemory新成绩。峰值统计为PyTorch allocated，含权重和中间张量。

按用户要求，旧PhaseKV、Prompt90及BetterMemory的TTFT字段与相关比较已清理；其它算法TTFT及所有质量、解码/端到端速度、显存指标保留。PhaseKV的TTFT仅采用TTFT同卡复测_20261107中的两个测点，不补齐或拼接成16点成绩。
最新记录：[TTFT同卡复测分析](A800最终测试/TTFT同卡复测_20261107/结果查验与分析.txt)。
