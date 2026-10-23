# F1二特征逻辑校准 → F0唯一期望F1控制

## 固定协议

依据[只读互补诊断](P-20261023-BINARY-SFT-COMPLEMENT.md)，主线程授权一个固定低成本外组控制。仅F1 1557个literal关系单位拟合 `[binary raw logit1−logit0, SFT32 hard]`；原两logit完整存在，不使用clip/logit(p)回退。固定带截距L2 LogisticRegression，C=1、lbfgs、max_iter=1000、class_weight=None，不扫描或追加特征；CPU≤60秒，序列化模型及coef/来源后先push。

F0用同binary32一次新前向与旧SFT48 hard，只有一个最终期望患者F1 top-k政策（允许空、稳定概率排序、exact tie较小k），原数学源码6537fadb字节快照复用。全部非association对象保留旧SFT48完整预测，只改各患者phenotype集合；完整新预测freeze后才读F0 gold，装配/评测≤30秒。保留标准：Δ(micro+macro)>0且macro不降，不另跑固定.5臂。F1未作拟合质量验收；F0是历史开发验证，SFT32→48及候选32→48规模变化明确保留。

F1候选/两模型均排除F0；F0旧SFT48和候选train48含F1但不含F0，对F0合法。F0无标签单位检查：58提示/415CID/47患者/1166 literal pair，无compound/重复，单普通文本`metabolic diseases`保持字面，旧hard276关联与旧完整pred集合一致。无原dev或官方操作。

## 已完成的F1拟合

唯一固定LR CPU主体0.0105秒，含导入1.054秒；11迭代收敛。系数依次 `[0.5166889924, 2.2208233493]`，截距 `−1.2482764434`。模型817字节仍留外部并实查SHA；序列化回读系数exact，没有F1训练质量评分。来源、实际文献集合和恢复见[fit证据](../evidence/round2/binary-sft-calibration-fit.json)。下一仅等待检索负责人完整F0前向冻结，再按固定政策装配唯一质量端点。
