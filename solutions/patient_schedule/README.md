# 患者优先四轮统一时程

这是一个尚未启动 GPU 的独立固定控制。恢复入口为 [evidence/ready.json](evidence/ready.json)，状态见 [WORKLOG.md](WORKLOG.md)，固定方法见 [实验记录](experiments/PS4-20261024-v1.md)。

目标是检验原 patient-first 标准 CE 在 fresh 四轮统一学习率时程下，能否提高完整 F1 总 Score。训练数据、目标、候选、tokenizer、基座和 LoRA 保持；只改变总曝光与配套 LR horizon/warmup。不会从旧 LR=0 checkpoint 接两轮，也不看中途 held 分数挑 epoch。

[training-plan.json](training-plan.json) 固定 124 行×4、64 次 optimizer call、4 次 warmup；[quality-plan.json](quality-plan.json) 固定主臂强全文 N∪P4、诊断 P4-alone。主机制需同时超过 N-alone 和匹配 N∪P2；任一预先固定臂超过强 N 都可作为候选，最高 Score 胜出、平分主臂。macro 全报告，不额外否决。

准备仅访问本地 CPU/文件，未部署、未加载模型、未占卡。源码及小摘要入 Git；训练行、完整预测、日志和 deployment 均在外部 run 目录。授权的唯一匹配旧模型 baseline 已完成，无旧臂重评。
