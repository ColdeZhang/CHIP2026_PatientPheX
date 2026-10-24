# 生成跨度补充Task1

主线程独立控制，基于main9c85180。目标是补强Task1的实体与概念召回，完整F1固定Score严格高于 .728697536469；实际完整端点负结案。

输入为已完成grounded39原始生成的固定normalizer结果786cc560，328接受记录。统一加入全部合法新跨度/ID，保留原1313实体、患者和关系，不筛选正例、患者、阈值或重规范化。逐原文定位、ID及冻结前拒读gold门在实际装配中检查。

入口：`python solutions/grounded_task1/run.py --plan solutions/grounded_task1/plan.json assemble`，成功后同入口 `evaluate`。全量预测在仓库外；CPU装配后仅一次v2，不重评旧baseline。

11:20实际CPU装配0.186883秒，全部328接受记录按固定原子去重补42实体，完整16文献/39患者，原1313实体及全部非实体对象不变，格式0错误/0警告。共同冻结07d0d1db后唯一v2 0.038658秒：Score .728697536469→.727242335488，mention新增11TP/31FP，document新增6TP/12FP，关系两项完全不变。固定门失败，关闭该统一增实体配方，不筛正确子集、改阈值或迁原dev。

21成员小包已上传.100并实际读回全部成员，SHA/fsync/重开通过，2次read_file；receipt8a1f852f。未复制模型/heldgold/HPO；独审待单独汇入总索引。
