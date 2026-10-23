# HGB池的固定严格嵌套结构政策迁移

真实问题：旧greedy输出不允许任何重叠，可能漏掉不同概念的嵌套表型。使用已冻Bioformer四fold完整score池，只对原HGB p>=.3/原排序的重叠拒绝项追加；保留全部原实体和所有非entity对象。

旧`append_nested_spans.py` f6d95355的`overlap/proper_containment/may_append`直接导入，不重写边界政策：必须至少碰撞一个已接受项，每个碰撞都严格包含（允许共享一侧边界）且原子HPO集合不交；同边界、交叉、共享ID、无碰撞均不加。已有新项也参与后来所有碰撞检查。沿用allowedHPO去root/NIL及旧否定筛选，仅用于新项；原NIL/否定实体不删除。

这是结构政策迁移，**不是旧GLiNER全链复现**：原GLiNER score排序/Sap .65/.92上游改用当前HGB p>=.3/原排序，不加Sap二次门。未改池/模型/特征/概率/阈值，不能与本体top3新池混合。15项合成合同覆盖包含方向/共享边/同ID/同界/交叉/NIL/所有旧新碰撞/否定/空池。逐候选拒绝计数及实际全部碰撞判定外存，不把大量细节写Git。

运行总CPU≤30秒；四旧四新完整预测共同freeze后一次新臂v2/格式，旧baseline按相同SHA复用同期首次完整指标。全train64曾选择.3且参与研发，当前是开发控制，无原dev/GPU/新fit；负或零即关闭，不扫阈值。

恢复：`/workspace/patientphex-runs/round2-20261023/sequence_crf/selector-nested-control/plan.json`；入口 `run_nested_selector_control.py --directory <该目录>`。[小计划证据](../evidence/round2/selector-nested-plan.json)。当前尚未正式运行。
