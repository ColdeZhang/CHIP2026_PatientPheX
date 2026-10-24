# E-20261024-PATIENT-SWAP-DPO-001

假设、固定配方、父模型及主要质量端点集中在 [REPORT](../REPORT.md) 和 [candidate-plan](../candidate-plan.json)，不在此复制。

目前阶段为readiness：供给和分词CPU、NumPy梯度、实际tiny Qwen3/PEFT CPU均通过，源码/输出可由 [evidence/readiness](../evidence/readiness.json) 定位。未运行预训练模型、未做大模型更新、未读heldgold或产生新的质量分数。

计划变量是基于同CID患者交换的reference-relative DPO，并明确同时禁用dropout以支持确定性串行梯度。固定完整主终点替换四源中的Q，允许新增与删除而保护B471；不能把后续仅对弱Q单臂的提升当完成。

后继按root统一资源安排：实际新增reject的目标HF编码检查；fresh父adapter/runtime/GPU门；唯一有界H51/replay/cost；仅门通过后固定reference→一轮74pair→持久终态→全80→冻结完整主终点。任何失败保留原attempt，未知远端状态恢复原意图，不重启重复训练。
