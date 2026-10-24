# GLiNER2.5实体监督适配

先看[最终实验决定](decision.json)。原[CPU准备报告](REPORT.md)和[配方](recipe.json)保留当时原字节，其中“paired base/adapted、冲突过滤、排NIL”的初始提案已被最终决定替代，不能当作当前执行计划。

真实CPU完整320窗目标与loader覆盖已过；GPU原生loss/梯度/显存探针仍独立验证。新路线为explore/gliner25-task1，旧schema-extraction关系零样本负结论不改写。全部训练/数据大产物在仓库外，main仅存准备和决定摘要。
