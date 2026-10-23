# 原Bioformer候选池加公开 pair 均值

这是root在top3+pair终点独审后指定的唯一隔离控制。此前在扩展pool上三列有收益但仍未胜强基底；因此只保留原Bioformer旧pool，检查是否是top3扩池影响了排序。不是在原dev上选方案。

固定四fold OOF：每个outer训练48的候选来自三套train32→16；原prepared `87264eb8` 逐键、所有字段、类型和顺序保留。无top3新ID、六rank字段、Sap新cos或固定边界臂。只追加已审公开PhenoBERT `[Not Match, Related, Match]` 原子均值，原文surface→指定HPO preferred name；原子方向/compound均值/NIL三零政策与前轮字节同源，不重NN。

标签、HGB160参数/seed/无earlystop和.3 greedy同旧。四fit+infer共120秒，TERM5；已审同旧pool两Sap字段实际55.563秒，新估1.5倍+10=93.344秒，估算不是完训保证。四新+旧预测共同freeze后，九点[.1… .9]全36预测先freeze，再一次按pooled `(menF1+docF1)/2` 选择，tie距.4近再高，CPU30秒。原所有非entities对象/关联保留，关系不参与选择，不追加原dev。

CPU准备13.271秒：244794候选出现完整保留，238294编码、6500纯NIL；39576唯一(text,ID)中38385编码/1191NIL，没有缺失合法pair。全部旧键字段顺序及无top3列通过，prepared83c68083；静态labels/select/load_modules/public_sources/check_pool、calibration数学五函数AST与旧已审实现exact。实际来源均快照，不从活跃其他分支导入。正式plan667828ad，尚未fit/读本轮质量；原来源和27个pair合同/135000缓存独审复用，不重新审模型。

详细参数与精确路径：[计划与恢复](../evidence/round2/phenobert-old-pool-plan.json)。若无收益则关闭pair特征这条，不追加阈值/模型/原dev测试。
