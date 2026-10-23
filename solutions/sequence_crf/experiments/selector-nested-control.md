# HGB池的固定严格嵌套结构政策迁移

真实问题：旧greedy输出不允许任何重叠，可能漏掉不同概念的嵌套表型。使用已冻Bioformer四fold完整score池，只对原HGB p>=.3/原排序的重叠拒绝项追加；保留全部原实体和所有非entity对象。

旧`append_nested_spans.py` f6d95355的`overlap/proper_containment/may_append`直接导入，不重写边界政策：必须至少碰撞一个已接受项，每个碰撞都严格包含（允许共享一侧边界）且原子HPO集合不交；同边界、交叉、共享ID、无碰撞均不加。已有新项也参与后来所有碰撞检查。沿用allowedHPO去root/NIL及旧否定筛选，仅用于新项；原NIL/否定实体不删除。

这是结构政策迁移，**不是旧GLiNER全链复现**：原GLiNER score排序/Sap .65/.92上游改用当前HGB p>=.3/原排序，不加Sap二次门。未改池/模型/特征/概率/阈值，不能与本体top3新池混合。15项合成合同覆盖包含方向/共享边/同ID/同界/交叉/NIL/所有旧新碰撞/否定/空池。逐候选拒绝计数及实际全部碰撞判定外存，不把大量细节写Git。

运行总CPU≤30秒；四旧四新完整预测共同freeze后一次新臂v2/格式，旧baseline按相同SHA复用同期首次完整指标。全train64曾选择.3且参与研发，当前是开发控制，无原dev/GPU/新fit；负或零即关闭，不扫阈值。

恢复：`/workspace/patientphex-runs/round2-20261023/sequence_crf/selector-nested-control/plan.json`；入口 `run_nested_selector_control.py --directory <该目录>`。[小计划证据](../evidence/round2/selector-nested-plan.json)。实际已完成，见下。

## 唯一四fold终点

实际进程2.631秒（程序主体1.348秒），exit0，8fullpred在新标签评价前共同冻结。268原门槛以上overlap拒项中，12被同原否定规则挡住、61不满足所有碰撞的严格包含/异ID、195追加成功；无新NIL/非法HPO，全部实际碰撞逐项记录。

pooled Score **.725655389→.727975656**（+.002320267）；men .763831→.766299、doc .796554→.803367，关系两项完全不变。新增实体86TP109FP、doc58TP48FP；无删除、关联增删0，0错警。四fold Score Δ `[+.002327052,+.004209081,+.003453136,-.001194855]`，最后一折men/doc都下降（13TP28FP新增），不得只写均值正。

该结果支持一次固定结构迁移的小收益，仍是开发结果，不是.8或原dev结论。root条件授权仅在本次独审通过后做唯一原dev同政策实体追加、现368条关系不变；当前尚未运行。完整摘要/源码/每次碰撞证据见[结果](../evidence/round2/selector-nested-results.json)。

## 固定原dev迁移计划

四fold独审f5482b85通过，3/4正，满足root事前条件。复用actual c032概率池/e9b30447当前最佳完整预测，仅同7f4484边界政策新增实体；全部原1004实体/368关系/其他对象保留，不用新实体扩大关系支持。两fullpred先freeze后唯一v2、30秒CPU；无fit/GPU/新候选/阈值扫描。计划与源码快照见[dev计划](../evidence/round2/selector-nested-dev-plan.json)，外存selector-nested-dev-transfer/。该唯一原dev终点已完成，见下。

## 原dev唯一终点

计划25420d5推送后执行2.173秒CPU（主体.643秒）、exit0；两完整预测22:13:11前共同freeze f1d7e752，之后唯一新v2，0错警。49个原overlap拒项中1否定、10未满足所有碰撞政策、38追加。全部原1004实体/368关系/非entity对象不变，新实体不扩关系支持池。

当前e9 baseline **.736526555→.737585917**（+.001059362），新预测d531f2ee；新增14TP24FP，men **.786729858→.785847300**下降，doc **.816874401→.821994408**上升，micro .708891596/macro .633610366保持。doc增加15TP15FP，与mention差异源于概念去重；无实体删除/关系增删。仍未.8，仅此固定结构政策迁移的温和收益，不再扫cut/顺序。

[完整结果/恢复与碰撞证据](../evidence/round2/selector-nested-dev-results.json)。独立增量审待回收；当前已保存原概率池/模型和旧best，主线程将在审查PASS后另外备份新完整输出，不把Git当大产物备份。
