# 文档支持或正共识的关系过滤

2026-10-24 UTC，分支 `explore/document-supported-relation-filter`，main 起点 `504659ec`。只改本目录；外部产物 `R/research/document-supported-relation-filter-20261024/`。

目标：检验当前 B4 全部关系的固定可靠性过滤能否提升完整公共 v2 Score。公式为 `B4 ∩ (Task1 文档概念支持 ∪ 至少两个 pure G/S/Q/C 正支持)`；只删除，零新增。S 是原 S1∩S2，不能把 Bunion 的旧基线算作正票。

当前：F1唯一固定终点负，关闭本配方，不做F0/full64。有限源码查重和37项合同后，源码/decision/实际SHA计划先push `82dbca4d`并核远端，才读取真实删除数。原dev最佳05cfa90f及旧veto结果保持。

结果：13:44:25.446完整freeze→13:44:27.690唯一1新v2/旧0。524→522，删1TP/1FP，Score .701325377212→.701225288549（−.000100088663）；micro−.000481684941、macro+.000081330290。16篇39患者1189实体/其他对象不变，非association字节精确。pure G282/S383/Q416/C395，S由408∩418求交。

恢复入口 evidence/terminal.json；完整逐关系审计/原预测/原baseline沿外存terminal保留。不扫票数或support-only，不追加质量。源代码与SHA绑定均完成核验，结果摘要另提交。

记录边界：逐关系支持和完整预测仅外存；Git 留源码、短摘要、证据指针。无模型、fit、服务器操作或官网动作。每折完整 freeze 后唯一 1 新 v2，旧0；零删除 byteexact、0v2 关闭。恢复按实际 freeze/result，禁止重复质量。
