# 固定非重叠全局解码控制

假设：逐概率greedy不一定最大化一组互斥跨度的总效用。固定既有Bioformer四fold OOF完整概率池、cut .3，仅改解码为标准weighted interval scheduling；不用新top3候选/特征，不重fit，不读原dev。

每项效用为JSON十进制精确分数`p-3/10`，仅`p>.3`入选；p=.3零效用明确排除。首先最大化总效用，然后最大化与该篇原greedy实体键的交集数，最后选按(offset,length,identifier)排序后的键元组字典序最小者。半开区间允许接壤，任何重叠不允许。原greedy全局最优且没有exact .3时被保留；不把这一结论错误推广到零效用项。

合成全局优于greedy、空池/零效用/同界ID/接壤/并列、80组独立穷举及固定reference置换合同已通过。运行时先复现原greedy实体、保留全部非entity对象，四旧四新完整预测共同freeze后再读train64标签。旧cut.3 fullbaseline由本体同时完成首次评价，按相同预测SHA复用其完整指标/格式，不再重复旧质量评价；新top3池不进入本控制。

总CPU硬cap120秒，无GPU；唯一指标结论包含各fold/pooled全v2、实体及关系增删TPFP。cut .3使用过全train64标签，当前各折也参与过研发，属于开发控制，不是盲测。没有概率校准保证，不宣称效用必然等于F1。结果负则关闭本固定控制，不扫效用/门槛。

外部恢复：`/workspace/patientphex-runs/round2-20261023/sequence_crf/weighted-interval-control/plan.json`。入口`run_weighted_interval_control.py --directory <该目录>`；完整输入/源码SHA及CPU合同见[计划证据](../evidence/round2/weighted-interval-plan.json)。固定任务已完成，结果见下。

## 唯一终点

实际进程3.309秒（程序内部3.115秒），exit0，120秒预算内。2026-10-23T22:08:11.660616+00:00八份完整预测共同冻结，随后一次新臂评价；旧指标复用同SHA首测。四fold/64篇新完整输出与旧输出逐字节相同，原4315条实体、全部非entity对象不变；候选中exact p=.3为0，合计4583条正效用候选，所有文献效用增益为0。

pooled Score **.7256553887000612→.7256553887000612**；men .763831183/doc .796554009/micro .687846109/macro .654390253均不变，四fold全Δ0，实体和关联增删TPFP全0，格式0错0警。该候选池和固定效用下，greedy已达到相同全局最优；合成例能区分两算法，不是实现只回传旧实体。关闭本控制、保原decoder，不改效用/门槛，不迁移原dev。

[完整小摘要及实际恢复入口](../evidence/round2/weighted-interval-results.json)。代码计划2c34d39已推送；独立增量审PASS（4802573a）：248穷举小例、反向DP、64篇完整重放和新v2/格式一致。
