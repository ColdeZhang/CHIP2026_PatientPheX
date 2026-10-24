# 方法和风险

新增监督变量是OOF候选拒绝标签，不换生成架构。现成纯F3→F2/F2→F3缓存同时排F0/F1，参照HANDOFF H40。原四fold候选不能默认复用。

20756可用池只是标注意义负例，仍有漏标风险；全部边界重叠、正span错概念、NIL/outbranch不进入新增NO。固定1:4是成本受限的保守选择，不称最优。原候选prob来自class-balanced词面LR，并非上下文/概念正确概率；只做文内排序，不输入模型。

完整来源分析R/research/mention-hard-negative-readiness-20261024/REPORT.md SHA7e8fba82；census81d46d98。F1输入proposer仍含训练F0的强48模型；F0后继必须替换排F0强池，不能把head纯train32误称全pipeline纯train32。
