# F1二特征逻辑校准 → F0唯一期望F1控制

## 固定协议

依据[只读互补诊断](P-20261023-BINARY-SFT-COMPLEMENT.md)，主线程授权一个固定低成本外组控制。仅F1 1557个literal关系单位拟合 `[binary raw logit1−logit0, SFT32 hard]`；原两logit完整存在，不使用clip/logit(p)回退。固定带截距L2 LogisticRegression，C=1、lbfgs、max_iter=1000、class_weight=None，不扫描或追加特征；CPU≤60秒，序列化模型及coef/来源后先push。

F0用同binary32一次新前向与旧SFT48 hard，只有一个最终期望患者F1 top-k政策（允许空、稳定概率排序、exact tie较小k），原数学源码6537fadb字节快照复用。全部非association对象保留旧SFT48完整预测，只改各患者phenotype集合；完整新预测freeze后才读F0 gold，装配/评测≤30秒。保留标准：Δ(micro+macro)>0且macro不降，不另跑固定.5臂。F1未作拟合质量验收；F0是历史开发验证，SFT32→48及候选32→48规模变化明确保留。

F1候选/两模型均排除F0；F0旧SFT48和候选train48含F1但不含F0，对F0合法。F0无标签单位检查：58提示/415CID/47患者/1166 literal pair，无compound/重复，单普通文本`metabolic diseases`保持字面，旧hard276关联与旧完整pred集合一致。无原dev或官方操作。

## 已完成的F1拟合

唯一固定LR CPU主体0.0105秒，含导入1.054秒；11迭代收敛。系数依次 `[0.5166889924, 2.2208233493]`，截距 `−1.2482764434`。模型817字节仍留外部并实查SHA；序列化回读系数exact，没有F1训练质量评分。来源、实际文献集合和恢复见[fit证据](../evidence/round2/binary-sft-calibration-fit.json)。随后按下述唯一端点完成，不再等待或重启。


## 唯一F0终点与决定

检索负责人同binary32完成58提示/1166 pair前向，程序66.246秒（纯forward43.478秒）、23:15:25Z释放GPU；全部raw两logit/p先冻结，无阈值臂。LR拟合与freeze已在 b7457c5 推送核验后，endpoint源码/计划又在72619d4推送；23:16:24Z开始唯一装配，完整预测 `47cf1e77…` freeze 后才读F0 gold，装配/评测0.414秒、含导入外层1.403秒（限30）。

| 同F0/同947实体 | micro F1 | macro F1 | Score |
|---|---:|---:|---:|
| 旧 SFT48 | .690076336 | .639022564 | .695443948 |
| F1 LR + F0期望F1 | .690076336 | .643429528 | .696545689 |

macro增加 **.004406964**、Score增加 **.001101741**；全局关系仍为226TP/50FP/153FN，micro完全不变。预定保留条件通过，但增益很小，仍低于同实体规则基线 .711353522。主线程决定关闭本固定试验，保留这份可组合候选，不自动做full64训练、原dev迁移或额外阈值/解码控制。

完整终审PASS（0b44186b）：独立F1固定LR refit的coef/intercept/11iter精确，1166校准概率全等；另一joint-count DP独立核47患者全部k与选点，完整预测byteexact/v2全指标精确，格式0错警。实体及所有非association对象、患者/association其他字段保持旧SFT48原样；旧基线指标直接绑定复用，未重评旧模型。输入和实际神经终点/释放复用已审来源，不重NN。完整[小证据/恢复入口](../evidence/round2/binary-sft-calibration-result.json)。
