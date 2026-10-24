# 四来源关系共识

2026-10-24 UTC。目标：检验新患者编码输出能否补强现有三源二票，不能只超过较弱单模型。

已固定[决定](decision.json)及[F1计划](plan-F1.json)，源码沿已核open-concept控制，仅将交集机制改为新C与三旧源至少一源一致。F1已完整装配及唯一v2，507→524关系（17新7TP10FP），Score .701087332488→**.701325377212**；micro+.001233、macro−.000281，实体两项不变。原dev最佳.755240不变。

source/plan62565701已先推送核验，13:04:07共同freeze后唯一1v2、旧0、无NN/fit。完整预测/指标在外部，见[入口](evidence/entry.json)。后续：等待原strict coded58，按原固定公式做F0；双折正才提出full64。不同去重解码政策不混入本实验。

自身12内容小包（含新完整预测）已在.100持久并实际2次read_file回读、逐SHA/fsync/reopen通过，zip31982B，回执[1f66ef85](evidence/F1-backup.json)。旧四源raw/模型沿原完整包继承，不重复备份。
