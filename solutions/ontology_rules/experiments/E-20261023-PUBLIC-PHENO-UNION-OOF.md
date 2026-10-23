# E-20261023-PUBLIC-PHENO-UNION-OOF

目标：检验已完成CPU计数后的公开候选扩展能否让已有学习选择器找回更多有效实体。真实成功标准：固定四开发fold pooled默认v2 Score超过已独审.95对照.7226510219708908，同时完整报告各fold；不原dev或官方。

在未看新质量前，root固定旧.95∪新.5 native canonical，共8221键，保留6064旧键（含原hybrid会移掉的3键），新增2157。相同canonicalkey真实来源flag取OR，分数取实际观察hybrid score最大值，HPO/headcolumn只作元数据，不编造NN后验。这是候选+来源的组合变化，不是纯阈值单变量。

复用已独审的四套外train48/held16，每外train48用三组真实train32→held16 OOF底层候选；公共原生模型无taskfit，纯input64缓存可共享。对每折重新拟合相同HGB160/.4/seed20261023/全部池greedy，旧.95已训树不复用。4个新fit/infer共120秒CPU硬限，旧全部控制不重训；4完整预测共同freeze再一次pooled/各fold评价。

[冻结计划](../evidence/round2/public-phenotagger-union-plan.json)绑定原3378/c258源码快照、全部旧reference/数据/指标、新canonical/provenance及CPU证明。当前尚未新fit或质量评估；数据准备和增量独审可并行，不额外增加逐步审批。

第一次CPU准备在0.165秒因调用本体模块不存在的parse_hpo退出；本体真实接口名是load_hpo（parse_hpo属于sequence），只修函数名，原HPO/allowed政策不变。失败源/计划/执行/stderr保留于外部attempt1-HPO-loader；尚未候选处理、拟合或质量评估。

正式四fit共49.192秒/120限内，21:32:32.988四pred共同冻结后21:32:39唯一评。pooled .722651022→.718046744（−.004604278），四分项均降，四fold中三负；root判定本配方关闭，不扫阈值、组合Bioformer或原dev。完整终点独审待完成。随后另授权仅已有结果聚合，区分2157新public键是否原pool已有及真实新增候选被选择多少，不能因果区分重拟合/特征/greedy，不创建oracle预测或gold补丁。
