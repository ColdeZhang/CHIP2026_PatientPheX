# 第二轮证据索引

- [CPU第一阶段](cpu-stage1.json)：14组内验证关键结果，包含无收益与失败配置。
- [语义阈值内选](semantic-inner-sweep.json)、[冻结四折及dev](semantic-frozen-results.json)：生成与装配来源分别记录；独立逐字节复现通过。
- [文内缩写](abbreviation-results.json)、[关系候选负结果](relation-pool-sweep.json)、[宽候选微收益](broad-candidate-result.json)。
- [OOF候选校准初版](oof-calibration-v1.json)、[固定160对照与初版复现](oof-calibration-fixed160.json)：目前最好dev .646244，二阶段训练层级清晰区分。
- [首个GPU作业](gpu-job.json)：启动命令、固定GPU与managed进程；后续作业详见实验与外部manifest。

本目录只存小型证据。模型、全量预测、候选与完整日志保存在方案台账所列外部目录；当前未达0.8，不上传官方。
