# 关系共识融合台账

- 当前配方已关闭：内部F1 **.686338981272→.690618521538**，F0 **.711507580027→.710063635556**；F1正/F0负，固定F0门失败，暂不迁原dev、不改cut或重训。原dev最高901保持。
- 固定 `base ∪ (DocRE30 ∩ schema)`；F1新22=10TP/12FP，F0新9=3TP/6FP，均0移除/实体与其他对象保持。两个单模型原负不改。
- F1完整16/39/1189、0fit/NN、1新v2/旧0；独审ae9c4631通过，25成员小包已从.100真实恢复。F0完整16/47/947、同model/规则、1新v2/旧0，集中增量审3cb75715通过。两份25成员小包均已实际从服务器恢复，参见[evidence/backups.json](evidence/backups.json)。
- 结果：[F1](evidence/result.json)、[F0](evidence/F0/result.json)；[实验](experiments/E-20261024-FUSION-001.md)。两次均先全预测freeze，再唯一v2，未触原dev/官网。
- F1首次pair-reader错误在pred/gold前，actual-v1保留；机械修复后完成actual-v2。F0只将计数门参数化，原F1源码/预测保持。
- 辅助DocRE F0结果已b81d988推送，schema供给7eb4039；所有GPU已释放。当前无新模型或质量任务，本次审查与结果包均已闭环。
