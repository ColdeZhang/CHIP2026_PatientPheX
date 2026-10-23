# E-20261023-PUBLIC-PHENO-UNION-OOF

目标：检验已完成CPU计数后的公开候选扩展能否让已有学习选择器找回更多有效实体。真实成功标准：固定四开发fold pooled默认v2 Score超过已独审.95对照.7226510219708908，同时完整报告各fold；不原dev或官方。

在未看新质量前，root固定旧.95∪新.5 native canonical，共8221键，保留6064旧键（含原hybrid会移掉的3键），新增2157。相同canonicalkey真实来源flag取OR，分数取实际观察hybrid score最大值，HPO/headcolumn只作元数据，不编造NN后验。这是候选+来源的组合变化，不是纯阈值单变量。

复用已独审的四套外train48/held16，每外train48用三组真实train32→held16 OOF底层候选；公共原生模型无taskfit，纯input64缓存可共享。对每折重新拟合相同HGB160/.4/seed20261023/全部池greedy，旧.95已训树不复用。4个新fit/infer共120秒CPU硬限，旧全部控制不重训；4完整预测共同freeze再一次pooled/各fold评价。

[冻结计划](../evidence/round2/public-phenotagger-union-plan.json)绑定原3378/c258源码快照、全部旧reference/数据/指标、新canonical/provenance及CPU证明。当前尚未新fit或质量评估；数据准备和增量独审可并行，不额外增加逐步审批。
