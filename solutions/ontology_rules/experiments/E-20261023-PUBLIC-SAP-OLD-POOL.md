# 原Bioformer旧池 + 公开Sap两特征

## 目标与固定范围

上一完整top3+Sap控制虽比top3本身改善，却未超原Bioformer最佳。此独立对照只在**原固定Bioformer池**增加 `public_sap_cosine` 与 `public_sap_observed`，不引入任何top3新键或六rank/p/observed字段，区分扩池影响与候选指定ID真实匹配的价值。沿同冻结公开向量/alias-all-ID映射；不重新编码、不GPU、不读取原dev。

固定12套nested train32→16与4held输入、原候选全部键/字段/类型/顺序、HGB160/seed20261023；全部nonentities旧cut.3原样。四fit/infer共享120秒硬限。事前九点.1–.9／tie近.4再高、36完整pred先共同冻结后一次计算pooled(menF1+docF1)/2，与原已校准best .7801925961542502比较；另外报告固定.3分项。不扩cut/特征/来源，不以训练侧校准称泛化/完整Score/.8，不自动迁原dev。

## 最小CPU确认

9.919秒完成原池逐对象检查：244,794重复候选行，238,294 observed、6,500 missing（全部NIL），唯一38,674 phrase-ID、37,515 observed与1,159 missing。12nested/4held来源集合与原manifest完全一致，未添加新候选，未读取gold。原四fit manifest实际总50.213秒；按刚完成Sap扩展池217.670秒乘行数比例并加15秒，估86.314秒，低于120秒；估计不冒称实测。完整[准备证据](../evidence/round2/public-sap-old-pool-preparation.json)。

上一top3+Sap终审c5775cf4已通过；此原旧池对照随后正式运行，结果如下。

独立入口已冻结：fit仅原features+两Sap、逐doc去两列=oldprepared=legacy原池；120秒runner与30秒calibrator保留原数学，关键labels/greedy/评分/选择AST同旧。全部源/参数与[正式计划](../evidence/round2/public-sap-old-pool-formal-plan.json)先推送，随后按原预算运行。


## 唯一终点

四fold唯一fit全部rc0，共55.563秒（<120）。八完整预测共同冻结后，预定九点的36完整预测再次共同冻结，再进行一次训练侧实体评分，实际1.244秒。

| 设置 | menF1 | docF1 | 两实体F1均值 |
|---|---:|---:|---:|
| 原旧池best .3 | .763831183 | .796554009 | .780192596 |
| 原旧池+Sap固定 .3 | .760689354 | .796202252 | .778445803 |
| 原旧池+Sap best .4 | .761830531 | .796392334 | .779111432 |

best相对旧best **−.001081164**，四fold目标差值依次−.002198810、−.004857821、+.004763415、−.000980539。best .4减少72mentionTP/93FP、37docTP/55FP；固定.3则mention−6TP/+28FP、doc0TP/+2FP，均未改善两指标均值。候选集完全不变后仍无收益，关闭本配置及此前top3+Sap配置，不迁原dev、不扩大消融或阈值，不据此否定所有语义模型。

nonentities逐对象保持原cut.3模板，association原来自.4。本次是训练侧两实体F1校准，不是完整Score、独立泛化或.8达标。没有序列化HGB/vectorizer；实际源/计划/全概率/全预测留外部，可明确标记重建。完整[结果/恢复](../evidence/round2/public-sap-old-pool-results.json)。新增终点独审进行中，向量链完整复用前一审查，不重NN。
