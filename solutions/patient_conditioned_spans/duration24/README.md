# 固定24轮时长对照

原8轮F1固定双端点均负，原终审f74ef7e3保持。已有训练记录两项loss各轮下降，第7→8轮NER −.207988 / bag −.010728；只据训练证据检验优化时长，不能保证泛化收益。

本次唯一方法变量：fresh公开初始化24轮/768完整文档更新，比例10%warmup76，末轮一次。数据、全篇图、max-MIL+GP loss 1:1、LR3e−5、seed20261024、globalNER>0∩patient>0、规范化与两个完整端点都保持。三倍训练曝光；估263秒，共享900+TERM30。无新probe，不选epoch/cut，不覆盖原8轮。

本目录train.py/normalize.py在部署包替换同名入口，其余从source-contract.json逐SHA复用父目录；旧source留在原Git和原终包。effective-preparation.json仅引用原payload并改运行轮数说明，未重tokenize或改标注。末部署权重追加file+父目录fsync及CPU真实重载与当时model.state_dict全tensor键/shape/dtype/value一致检查，无额外forward、无optimizer/RNG。

完整两预测共冻后各1v2，旧0，主对照仍强B .686338981272；另外用已有旧8轮指标报告差值，不重评分。尚未执行时不声称成功。所有大产物独立外部duration24-v1。


已完整结束：264.731998秒rc0，12:59:13.938498 GPU0释放；205末权重tensor真实CPU重开逐键/shape/dtype/value相同。原58内容包260204B/3e2cf38c全SHA回收，质量包已实际远端fsync/readback。

| 端点 | 24轮Score | 比强B | 比旧8轮 |
|---|---:|---:|---:|
| B∪new | .669185282495 | −.017153698776 | −.009010836902 |
| new-only | .531756085992 | −.154582895280 | +.070532595047 |

主端点micro .697588126160 / macro .564214068033，87新增仅10TP77FP，全471旧关系/1189实体保留；source266关系。完整两端点共冻后2新v2、旧0，没有选择中间epoch或调cut。末NER3.175242 / bag.326286明显低于旧8，但NER第23→24轮3.032880→3.175242，并非仍单调下降。关闭这组时长配方，不继续加轮数；源码、部署权重、原raw和负质量保留。几何方向将是独立分支的另一个控制。

准确终点与恢复入口见 terminal.json。这里的弱source指标改善不能替代强B目标，未迁原dev/官方。

独立终审6929a048已通过，准确范围model-only、205tensor CPUexact与两指定目录fsync；见review.json。旧24全部负结案，后继geometry独立分支不混此批。
