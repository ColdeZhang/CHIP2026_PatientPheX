# 固定原门槛的 prehybrid 实体供给控制

全输入先验：两公开模型全64原raw top1，按原5位舍入后严格>.95且固定current-ID有效，绕过native hybrid/abbr。7397键/7355跨度；与旧池并集后四fold累计244794→252014行，仅新增7220重复来源行，旧对象完整原样、原顺序前缀保留，新增按(offset,length,ID)排序。全raw top3规模约20倍，未获准；无门槛top1方案也不启动。

实际raw top1已过门却在hybrid丢弃的217错误条件诊断为PubMed29/Bioformer23，其中正确literal并集27；没有abbr/current映射丢项。此证据只提出候选假设，不是保证的收益。

新增key的12个原native public/bioformer特征全部False，不能冒称已经过canonical输出。只新增raw key拥有4个独立字段：`raw_pubmed_observed/p`、`raw_bioformer_observed/p`。observed意为该模型原top1真实通过固定门，p取未舍入实际概率；缺失模型显式False/0。旧对象不添加字段，模型展开时默认False/0只表示“没有作为追加raw来源”，不声称旧候选的公共模型概率为零。

其他新增候选基础占位沿旧已审契约 probability=.5、similarity=0、exact=False、is_base=False、crf=False；crf_span仅在同训练来源原池确有该跨度时为True。既有训练/语境特征仍由同源码计算。此项是新增跨度及其真实来源信号的组合控制，不是只改解码。

复用严格每outer三个train32→16候选、原HGB160/seed/原模型参数。四fit+infer共同120秒硬限，固定九cut .1–.9另30秒；先冻全部36完整预测再唯一训练侧实体均值选择，tie近.4再高，同时报告固定.3。模板沿原cut.3，所有非entities逐对象相同，association实际来自旧.4。对照旧训练校准best .780192596；不能称独立泛化、比赛Score或.8达标。

源码从已审top3 fit/旧池Sap wrapper与9cut窄适配，旧入口不改；实际新增源SHA单独冻结。每fold保存模型和vectorizer到外部，供load复核。raw/projected/prepared/model/scores/predictions均留外部 `ontology_rules/public-phenotagger-prehybrid-control/`。无原dev、NN、GPU或官方操作。
