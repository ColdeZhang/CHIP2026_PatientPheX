# GLiNER2.5 Task1 工作台账

2026-10-24 UTC，分支 explore/gliner25-task1；仅维护 solutions/gliner25_task1。

- **当前：原生压力合同完成，正式训练未启动。** 恢复运行6次真实FB/Adam/两轮，72 raw loss分项与梯度有限、encoder/boundary每步真实改变；已有300 Adam状态后max49 gold/最长699通过。峰allocated4.841GB、reserved5.994GB；native训练2.604秒，完整wrapper18.289秒。
- **GPU已实释**：.100 GPU0 UUID126293b9，2026-10-24 11:46:25.813506 UTC 0MiB/noapps；恢复proc_e373f602/PID2735167。终态见 [probe-final](evidence/probe-final.json)，不得重跑已完成6步。
- **两个工程故障已保留**：首probe把不同随机schema旧IDs当固定值，15.528秒/0FB退出；恢复保native .2 query增强、先存actual schema/IDs/RNG后独立比对，6步全完后因无eval的best_metric=inf哨兵不能JSON化而rc1。仅CPU恢复summary，无GPU重跑，原rc1/failure/partial保持。
- **目标与固定端点**：主strong8cb61ba9 .728697536469 ∪ 全部规范化接受mapped/NIL实体，精确原子去重；另new-only完整诊断保原关系；2共冻v2、旧0，主严格胜才候选。尚无新质量结果，原dev最高不变。
- **预算建议已报root**：fresh公开权重、320窗全10epoch/1600更新、final epoch10唯一；以实测稳态.151–.421s/步及native平均.434s估计约675–695秒，建议训练1200+30秒，正式固定链已获root分配1200+30，fresh gate后唯一启动。无holdout/earlystop/epoch选择。
- **证据与源**：首失败11件ZIP2169be41、恢复30件ZIP8e386dec全SHA回收，CPU准备de08d1f继承320目标/loader，当前GPU实际源818a4a5/probe c18ecd5e。最近已核远端818a4a5，本次收尾待push。
- **外存恢复**：R=/workspace/patientphex-runs/round2-20261023；本线R/research/gliner25-task1-v1。原始probe-v1、probe-recovery-v1；唯一推荐派生终态metadata-recovery-v2（v1的native耗时为已读值直接填，v2改从同一partial原件严格读取，无神经或质量重算）。服务器原完整证据仍在同相对目录。

下一步：源/规范push后，fresh .100GPU0门启动唯一正式1200+30；成功后600+30全部F1，再同policy Sap≤180+30和2完整共冻v2。当前正式尚无start intent，不重跑probe，不访问原dev/官网。
