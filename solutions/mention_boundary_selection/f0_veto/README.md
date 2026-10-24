# 固定 F0 BOUNDARY veto

同 final epoch2 头作用于强 F0 全部 1085 个精确跨度，只删除完整、严格解析的 BOUNDARY；其余状态保留原实体。基线 .740572140440，原 F1 主头两负与独立 veto 小正均保持。

373 条缺失公共 Sap Top8 查询已完整通过（11.010 秒，无截断），16:57:09.811953 UTC 实际释放。165 缓存＋373 新查询覆盖全部538词面，未插 gold 候选、训练别名或 exact override。

新1085 native CPU 已6.673秒完整通过：616583 tokens，最长698，每条原文/模板/ID与原生HF一致，无截断；原514状态语法沿旧实际合同复用。24 CPU成员全部SHA回收。训练3470/负例祖先都排F0，当前不重训。

[固定推断计划](infer-plan.json)绑定原头 beecfb84、同Torch/runtime/greedy/128、独立1800+TERM30与过期独占启动。GPU1按root就绪优先安排，启动前fresh核。[唯一veto协议](decision.json)沿已审root公式，完整冻结后至多1v2（零删除时原基线字节与指标继承0v2）。不运行新headunion、原dev、阈值扫描或官方。

外部目录：`/workspace/patientphex-runs/round2-20261023/research/mention-boundary-F0-veto-v1`。完整原始输出待推断；本目录不把准备合同当质量结果。
