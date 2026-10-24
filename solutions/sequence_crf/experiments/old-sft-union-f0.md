# 旧8B与9B SFT固定并集：F0负结案

真实目标是在规则确认允许多模型后，补核一个历史尚未覆盖的低成本互补控制。此实验与正在运行的患者优先F0训练完全独立，不改变后者的来源/策略/两臂。

先核历史与来源：指定旧Qwen3-8B预测c3157472、9B预测63aefe68，都为严格train48/204条OOF监督、固定2epoch/52calls，原58提示、16文献/47患者/947实体、415CID/1166候选对；三nested资源均排F0。实际所有nonassociation/患者元数据精确，8B297关系、9B276、交243、并330；8B414/415严格解析及1处既定LR回退如实保留。检索12个旧8B引用未找到两SFT直接并集终点，已有各自rules控制不能替代；不声称记录外绝对未做。

按root单项授权，预先冻结sourceaa0854c5/planfd96bda5，只做literal union，不加当前patient-first或其他规则；完整预测ace46852先freeze10eae725、beforegold ea3fe6b2，然后一次公共v2 all/text及格式检查，旧9B指标复用零次重评。CPU上限60秒，实际2026-10-24 03:26:49UTC内部0.222737/外层0.259816秒、rc0。没有新模型、训练、GPU或其他组合。

Score **.6954439475814587→.6885650228734785**（−.006878924708）；micro **.690076335878→.682651622003**，macro **.639022564483→.618931579526**，均下降；实体两项不变。新增54＝16TP38FP，无移除；格式16/47/947/330全覆盖、0错0警。按预设delta(micro+macro)>0且macro不降判据keep=false，负关闭，不扫组合、不F1/dev或官方提交。54条集合差异不能预示收益，本次主要增加了误关联。

完整源/预测/metrics外存`research/old8B-old9B-fixed-union/`；小[结果](../evidence/round2/old8B-old9B-union-results.json)、[清单](../evidence/round2/old8B-old9B-union-manifest.json)、[来源与历史](../evidence/round2/old8B-old9B-union-history.json)。独立增量终审c12e3ae5 PASS：literal集合/不变字段/330覆盖/冻结顺序/唯一v2和16TP38FP计数完整，未重评分/Gold/NN。见[终审](../evidence/round2/old8B-old9B-union-terminal-review.json)。
