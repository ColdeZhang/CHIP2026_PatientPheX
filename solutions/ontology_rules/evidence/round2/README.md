# 第二轮证据索引

- [CPU第一阶段](cpu-stage1.json)：14组内验证关键结果，包含无收益与失败配置。
- [语义阈值内选](semantic-inner-sweep.json)、[冻结四折及dev](semantic-frozen-results.json)：生成与装配来源分别记录；独立逐字节复现通过。
- [文内缩写](abbreviation-results.json)、[关系候选负结果](relation-pool-sweep.json)、[宽候选微收益](broad-candidate-result.json)。
- [OOF候选校准初版](oof-calibration-v1.json)、[固定160对照与初版复现](oof-calibration-fixed160.json)：目前最好dev .646244，二阶段训练层级清晰区分。
- [嵌套训练内校准](calibration-nested-fold0.json)、[关系覆盖诊断](fold0-relation-diagnostic.json)：独立内验证与oracle诊断明确分开。
- [Qwen全文关系负结果](llm-fold0-negative.json)：原始、单轮格式修复、规则交集均保留，未达标。
- [CE1epoch、训练先验修正与固定融合](context-ce-epoch1.json)：训练48嵌套候选，固定内fold0对照，无有意义增益，不跑dev。
- [CE3epoch负结果](context-ce-epoch3.json)、[CRF多来源候选树](multi-candidate-fold0.json)：完整对照与来源隔离，未达标。
- [A800运行时实测](a800-runtime-verification.json)：CUDA矩阵、Qwen全模型加载与最小生成通过。
- [首个GPU作业](gpu-job.json)：启动命令、固定GPU与managed进程；后续作业详见实验与外部manifest。

本目录只存小型证据。模型、全量预测、候选与完整日志保存在方案台账所列外部目录；当前未达0.8，不上传官方。
