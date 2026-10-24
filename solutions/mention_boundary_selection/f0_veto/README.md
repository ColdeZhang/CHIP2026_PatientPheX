# 固定 F0 BOUNDARY veto：负结案

同head/同公共Top8/同解码规则的1085跨度已完整结束。Score **.740572140440 → .740492025895**，下降 .000080114546；删4=1TP+3FP。menF1 +.000487901，但docF1 −.000808359，两关系指标完全相同。未通过预设严格正门，**不迁原dev、不改cut、不追加子集或官方提交**。F1独立veto小正与原主头union/new-only两负结论均保留。

373必要Sap查询11.010秒；1085 nativeCPU6.673秒/616583tokens/max698，完全无截断；原514状态语法与已审veto公式复用。训练3470与负例祖先排F0，本次没有重训。原epoch2 adapter504 tensors逐值exact，同Torch2.7.1/TF4.51.3，推断770.098秒，17:15:54.981426 UTC实际释放GPU1（0MiB/noapps，child rc0）。完整解码991HPO/90NO/4BOUNDARY，无NIL/拒绝/cap。

20 query、24 CPU、2202 infer小成员全部SHA回收；完整1085 raw/parsed和结果保外部，不写仓库。共同完整预测冻结351f651c先于唯一1v2/旧0。终态864c1e24不可覆写；小备份和后继独审用单独pointer补充，不改原件。

[结果](result.json)、[固定协议](decision.json)、[推断计划](infer-plan.json)、[实验记录](../experiments/E-20261024-boundary-veto-F0.md)。外部目录：`/workspace/patientphex-runs/round2-20261023/research/mention-boundary-F0-veto-v1`。这是第二份开发集控制，不是盲测或官网结果。

64件小备份298302 B/SHA72af4dbc已17:18:23.563548 UTC完成远端逐成员核验和完整同SHA回读；见[backup-pointer.json](backup-pointer.json)，原terminal/result字节均保持。
