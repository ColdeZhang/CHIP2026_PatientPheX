# 当前状态

已完成固定训练侧539个BOUNDARY选择，3470总训练行；511新增词面中2个复用已有公共Top8，509需同公共银行补query。509必要Sap已完整成功（11.57秒），原进程已释放GPU；3470完整训练/1416held准备就绪，完整原生CPU已rc0：4886输入/mask、514状态/12300前缀、20实际HF约束全部通过；小终态与56原件SHA manifest已回收，原9.4MB大包断点恢复中。唯一GPU链15:21:37.533682已启动 proc_98b3a99e / PID2750059（.100 GPU1）；实际probe13FB/1Adam已rc0，预锁成本5642.17≤7200；15:21:59.012453已fresh正式868步（timeout PID2750096，deadline17:21:59+TERM30），随后完整1416推断。只恢复原job，无重启。

- 基线与原结果：Top8两完整端点负、独审9f6d1365；原终点/权重不修改。
- 唯一执行计划：[plan.json](plan.json)。过程：[实验记录](experiments/E-20261024-boundary.md)。
- 大产物外存 `R/research/mention-boundary-selection-v1`；关键代码/结果及时commit及push，来源/未知变更先恢复原意图。
