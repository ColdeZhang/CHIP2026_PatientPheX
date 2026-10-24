# 前缀名称一致性工作台账

- 目标：评估原12条截断响应中已经完整输出的名称是否能在固定 grounded 一致性约束下补充强基线。
- 范围：独立分支 `explore/canonical-prefix-agreement`，起点 `f3a86fd2`；不修改原名称路线的负结论或原55关系源。
- 固定策略：27条strict行原字节保留；12cap沿原native `CanonicalList` 文法提取已闭合的JSON字符串项。接受的是原字符串项，整个列表仍可不完整；不添加字符、猜尾部或放宽未知／重复名检查。
- 唯一端点：B507（F1 .701087332488）∪（全prefix名称源∩固定grounded282）。无新增时预测逐字节等于B、0gold/0v2；有新增完整freeze后仅1新v2，旧0、单头0。无F0、原dev或官方动作。
- 当前：14项本地边界合同通过；源／固定计划准备推送，然后一次.100隐藏CUDA、线程1、≤60+5秒纯tokenizer状态重放。0模型加载／0新NN，不使用GPU。
- 研发性质：这是观察原cap工程行为之后选择的新解码策略，F1属于开发折，不称盲测。
- 大产物：`/workspace/patientphex-runs/round2-20261023/research/canonical-prefix-agreement-20261024/`。源码/固定计划见 `evidence/fixed-plan.json`，实际原生计划见 `evidence/native-plan.json`。

完成后只保小元数据与集中交接入口，完整逐项证据／预测放外部并持久备份。
