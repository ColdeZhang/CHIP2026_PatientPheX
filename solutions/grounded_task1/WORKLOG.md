# 生成跨度补充Task1

主线程独立控制，基于main9c85180。目标是补强Task1的实体与概念召回，完整F1固定Score严格高于 .728697536469；当前无新分数。

输入为已完成grounded39原始生成的固定normalizer结果786cc560，328接受记录。统一加入全部合法新跨度/ID，保留原1313实体、患者和关系，不筛选正例、患者、阈值或重规范化。逐原文定位、ID及冻结前拒读gold门在实际装配中检查。

入口：`python solutions/grounded_task1/run.py --plan solutions/grounded_task1/plan.json assemble`，成功后同入口 `evaluate`。全量预测在仓库外；CPU装配后仅一次v2，不重评旧baseline。
