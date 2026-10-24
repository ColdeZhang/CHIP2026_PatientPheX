# Qwen3 患者逐字quote

当前入口：[工作台账](WORKLOG.md)、[预输出固定方案](evidence/pre-output-plan.json)、[研究与复用边界](NOTES.md)。本线尚无模型训练或质量。

CPU部署入口 `cpu/runtime_transport.py --deployment <固定JSON>`，仅隐藏CUDA的tokenizer对照；保唯一意图，UNKNOWN时恢复原结果，不重跑。全量ids/log/预测/weights放仓库外 `patientphex-runs/round2-20261023/research/qwen3-grounded-quotes-v1`。
