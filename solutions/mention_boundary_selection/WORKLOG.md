# 当前状态

已完成固定训练侧539个BOUNDARY选择，3470总训练行；511新增词面中2个复用已有公共Top8，509需同公共银行补query。正在源/原生CPU准备，尚未新GPU启动。

- 基线与原结果：Top8两完整端点负、独审9f6d1365；原终点/权重不修改。
- 唯一执行计划：[plan.json](plan.json)。过程：[实验记录](experiments/E-20261024-boundary.md)。
- 大产物外存 `R/research/mention-boundary-selection-v1`；关键代码/结果及时commit及push，来源/未知变更先恢复原意图。
