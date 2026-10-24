# E-20261024-GROUNDED-TASK1-001

假设：被旧强实体选择器遗漏的精确quote能带来净增益。基线为F1 lightgbm.nested.jsonl SHA8cb61ba9、Score .728697536469，非弱1189实体基线。固定唯一变量是全328接受记录按跨度/原子HPO去重后的新增实体；全39患者关系不变。

无fit/NN/新gold训练，接受策略沿用已冻结b59eb3f8。plan.json锁完整数据/来源SHA及唯一all/text v2；先完整装配与格式检查、冻结，再读gold进行一次新评分，负结果也结案。
