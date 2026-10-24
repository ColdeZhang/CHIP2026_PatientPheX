# 工作台账

2026-10-24：从实际main20fa611b独立建explore/patient-competition-loss。旧geometry已经有效负结案；47新增FP中31为概念属于其他患者，驱动本次同概念跨患者目标控制。

当前：完整源码与配置已准备，CUDA隐藏CPU合同全部通过：1174组/93多正组，原3194袋/813正/1masked正不变；实际13:19:13.270747 UTC唯一proc9a3f/driver2741397已启动，截止13:34:13.270747+TERM30，768step/16/39/规范化共享900；首8步有限，3.77GB。尚未质量。只替换患者bag BCE为grouped MLCCE，原数据/模型/geometry/optimizer全部继承。源和CPU证据先push后唯一完整900+30。

原metadata、原8/24/geometry结果不改写。本线完整raw/模型/日志在仓库外新namespace。已知F1开发背景，非盲测。完整终点后统一增量审。
