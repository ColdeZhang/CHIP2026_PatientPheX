# 四流明确阴性一致删除

2026-10-24 UTC，独占 explore/unanimous-relation-veto，main起点0878cbdf。

目标：仅删当前B4中被S1/S2/Q/C全部明确否定且G不支持的关系，检验全量综合Score；不把缺失/弃答当阴性。真实机制区别于旧N/R/P票数过滤，必须逐流真实CID、clinical患者覆盖、完整strict/noncap。

当前：两折固定控制已完整结束，F1 正、F0 负，关闭本配方的 full64 迁移。源/decision/实际SHA plan与31合成合同均先push再实际装配；没有阈值或子集扫描。

外存：R/research/unanimous-relation-veto-20261024。逐pair状态/删除证据只留外部，Git存源/短记录/终态指针。恢复按实际freeze/result，禁止重复评分。

F1唯一终点已正：13:34:36.223完整freeze后1v2/旧0，.701325377212→.702208481659（+.000883104447）；删9=2TP/7FP，micro+.002648676/mac+.000883742，其他对象/实体精确。S1/S2/Q各1557明确状态；C原40unknown保留。

F0原样复验已负：13:36:15.957完整freeze后1v2/旧0，.728195628911→.727598926304（−.000596702607）；343→339，删4=2TP/2FP，micro−.001481493/mac−.000905318，其他对象/实体精确。不得用F1收益绕过两折门，不迁移full64、不改投票数/患者子集。

恢复入口 evidence/terminal.json → 外部 terminal-manifest.json f73a671e（9424B）。两折共2新v2/旧0、0NN/fit/服务器调用；不需重评分。真实源码阶段提交59eca9fa、F0预锁d1af1008已先远端核同SHA，结果总结另提交。
