# 患者条件跨度 / max-MIL

当前固定8轮F1控制为负，见 [台账](WORKLOG.md)、[实验](experiments/E-20261024-PCS-001.md) 和 [终审/恢复入口](evidence/terminal.json)。源码独立分支，不改共享最佳。

核心链：`prepare.py` 完整文档/患者/概念袋 → `contracts.py` 真正autograd合同 → `driver.py --mode probe` 两整文真实更新 → fresh `driver.py --mode formal` 8轮、完整raw和固定规范化 → `assemble.py` 共冻双端点 → `quality.py` 唯一双v2。所有路径取配置/CLI；原运行命令与准确资产见plan.json及外部manifest。已完成实验直接读取现有结果，勿把这些入口当重跑当前评分的指令。

`model.py` 的所有窗口共享encoder、重叠token可微平均、全局患者锚点无detach；patient×concept max袋覆盖所有已表示真实跨度。global NER辅助复用GP multilabel loss，不制造nearest/all-span患者证据。患者头未见任意非表型负跨度，输出固定同时满足globalNER>0和patient>0，再用原strict/publicSap政策规范化。

完整编码、模型与raw在仓库外；Git只保存源码、固定方案、摘要和准确证据指针。部署checkpoint只含末轮model等必要字段，不能完整恢复optimizer/RNG。
