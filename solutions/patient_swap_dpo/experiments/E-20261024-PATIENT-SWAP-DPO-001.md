# E-20261024-PATIENT-SWAP-DPO-001

假设、固定配方、父模型及主要质量端点集中在 [REPORT](../REPORT.md) 和 [candidate-plan](../candidate-plan.json)，不在此复制。

准备阶段的供给和分词CPU、NumPy梯度、实际tiny Qwen3/PEFT CPU均通过，源码/输出可由 [evidence/readiness](../evidence/readiness.json) 定位。准备阶段没有运行预训练模型或质量；后续已获root授权进入实际GPU执行，实时状态以 [WORKLOG](../WORKLOG.md) 为准。

计划变量是基于同CID患者交换的reference-relative DPO，并明确同时禁用dropout以支持确定性串行梯度。固定完整主终点替换四源中的Q，允许新增与删除而保护B471；不能把后续仅对弱Q单臂的提升当完成。

实际目标HF148编码通过后，唯一148 reference已完整成功。原probe首backward失败，保留原数值、完整日志和109成员；经真实CPU cache标志诊断，只关闭autocast权重缓存，在原probe绝对截止内独立恢复。完整13对/26FB/真实Adam/显存与成本门通过，fresh原543c＋newAdam正式训练已启动。详见 [原失败及恢复](../evidence/probe-failure-recovery.json)、[GPU工程门](../evidence/probe-pass-formal-start.json)。reference原字节复用，没有重新计算；数学配方、完整74对、数值门均未更改。

后续必须完成持久终态→全80fresh→冻结完整主终点，才执行唯一完整质量。任何失败保留原attempt；未知远端状态恢复原意图，不盲目重启。当前没有新质量分数。
