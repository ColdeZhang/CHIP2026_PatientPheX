# 公开Pheno候选并集的严格OOF控制

`select_public_candidates.py` / `select_public_candidates_folds.py` 是sequence已独审3378/c258入口的原字节快照，HGB160/.4、三组train32→inner16候选、旧患者关联规则不变；原来源Git分支round2/sequence-crf，文件实际SHA见计划。不修改其活跃文件。

本控制只把公开候选换成事前固定旧>.95∪新>.5池。同键只对真实flags做OR、真实observed hybrid score取max，原始HPO/column只作来源元数据；这是候选及来源变化，不声称单一阈值变量。新prepare入口验证旧池6064键全部保留、总8221键，全部坐标/来源真实。

`run_control.py`执行4个新fit/infer，共享120秒CPU硬限，四个完整预测一起freeze。`evaluate_control.py`在核全部freeze后唯一评价4fold/pooled，旧.95指标复用，不重训旧控制。计划与产物在仓库外ontology_rules/public-phenotagger-union-control；源码、计划、短结果进Git，输入/完整预测/全scores不进Git。
