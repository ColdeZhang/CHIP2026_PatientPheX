# 固定24轮时长对照

原8轮F1固定双端点均负，原终审f74ef7e3保持。已有训练记录两项loss各轮下降，第7→8轮NER −.207988 / bag −.010728；只据训练证据检验优化时长，不能保证泛化收益。

本次唯一方法变量：fresh公开初始化24轮/768完整文档更新，比例10%warmup76，末轮一次。数据、全篇图、max-MIL+GP loss 1:1、LR3e−5、seed20261024、globalNER>0∩patient>0、规范化与两个完整端点都保持。三倍训练曝光；估263秒，共享900+TERM30。无新probe，不选epoch/cut，不覆盖原8轮。

本目录train.py/normalize.py在部署包替换同名入口，其余从source-contract.json逐SHA复用父目录；旧source留在原Git和原终包。effective-preparation.json仅引用原payload并改运行轮数说明，未重tokenize或改标注。末部署权重追加file+父目录fsync及CPU真实重载与当时model.state_dict全tensor键/shape/dtype/value一致检查，无额外forward、无optimizer/RNG。

完整两预测共冻后各1v2，旧0，主对照仍强B .686338981272；另外用已有旧8轮指标报告差值，不重评分。尚未执行时不声称成功。所有大产物独立外部duration24-v1。
