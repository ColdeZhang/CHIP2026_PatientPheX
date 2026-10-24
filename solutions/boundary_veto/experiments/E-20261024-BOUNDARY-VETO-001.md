# E-20261024-BOUNDARY-VETO-001

- 目标：独立检验新 BOUNDARY 训练信号是否改善强 Task1 F1，从原train64/dev16公共v2 Score≥.8的研发目标出发；当前仅CPU准备，不称完成质量实验。
- 起点：main `bc8084e756f6f24fb9e633cc4fd31f8395b43f30`，分支 `explore/boundary-veto`；独占 `solutions/boundary_veto/`。
- 比较：内部 F1，16文献39患者，原强1313实体；基线173758B/SHA `8cb61ba94fab8de4223acafeece83ef5a243954312d008073a42e5ba80398d6f`，Score .7286975364690524。开发折已用于研发，不称盲测。
- 变量：固定新3470训练目标产生的finalepoch2四状态输出，唯一额外完整删除终点；父两终点不改。
- 方法、来源、无输出预锁和成功门：[decision.json](../decision.json)、[NOTES](../NOTES.md)。执行入口：[README](../README.md)。
- 最小验证：现成held/encoded与输入文献来源合同；小型合成 exact边界、NO/NIL/拒答、row歧义、复合、多对象、完整性与字节保护合同。均不运行模型或质量。
- 后续：等待原完整训练/ACK/全1416及所有raw/parsed回收；root核完来源后绑定唯一计划、完整freeze，再仅1次新v2/旧0，零变更0。无 partial、子集扫描或自动迁移。

准备阶段结果见台账与 evidence；实际质量尚未运行，耗时/资源仅CPU合同自行记录，不推定涨分。
