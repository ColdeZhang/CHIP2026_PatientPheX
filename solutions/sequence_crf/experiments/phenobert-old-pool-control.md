# 原Bioformer候选池加公开 pair 均值

这是root在top3+pair终点独审后指定的唯一隔离控制。此前在扩展pool上三列有收益但仍未胜强基底；因此只保留原Bioformer旧pool，检查是否是top3扩池影响了排序。不是在原dev上选方案。

固定四fold OOF：每个outer训练48的候选来自三套train32→16；原prepared `87264eb8` 逐键、所有字段、类型和顺序保留。无top3新ID、六rank字段、Sap新cos或固定边界臂。只追加已审公开PhenoBERT `[Not Match, Related, Match]` 原子均值，原文surface→指定HPO preferred name；原子方向/compound均值/NIL三零政策与前轮字节同源，不重NN。

标签、HGB160参数/seed/无earlystop和.3 greedy同旧。四fit+infer共120秒，TERM5；已审同旧pool两Sap字段实际55.563秒，新估1.5倍+10=93.344秒，估算不是完训保证。四新+旧预测共同freeze后，九点[.1… .9]全36预测先freeze，再一次按pooled `(menF1+docF1)/2` 选择，tie距.4近再高，CPU30秒。原所有非entities对象/关联保留，关系不参与选择，不追加原dev。

CPU准备13.271秒：244794候选出现完整保留，238294编码、6500纯NIL；39576唯一(text,ID)中38385编码/1191NIL，没有缺失合法pair。全部旧键字段顺序及无top3列通过，prepared83c68083；静态labels/select/load_modules/public_sources/check_pool、calibration数学五函数AST与旧已审实现exact。实际来源均快照，不从活跃其他分支导入。正式plan667828ad在fit/新质量前冻结；原来源和27个pair合同/135000缓存独审复用，不重新审模型。

详细参数与精确路径：[计划与恢复](../evidence/round2/phenobert-old-pool-plan.json)。若无收益则关闭pair特征这条，不追加阈值/模型/原dev测试。


## 唯一终点：负结果，关闭本次 pair 特征路线

四fold实际fit/infer **64.882秒**<120，所有rc0；四新+四旧预测在 `f2d071cd` 统一freeze。九cut **1.113秒**<30，36完整输出 `401c7ad7` 先freeze再读本轮校准标签；唯一选点仍 **.3**。只使用公共v2的两个实体原语，不新增关系评分/原dev运行，也不把两实体目标当完整比赛Score。

| 指标 | 旧Bioformer .3 | +三pair均值 .3 | 差值 |
|---|---:|---:|---:|
| mention F1 | .763831183 | .759654305 | −.004176878 |
| document F1 | .796554009 | .796017699 | −.000536310 |
| 两项均值 | .780192596 | .777836002 | **−.002356594** |

四fold目标差值依次 **+.005387/−.001627/−.004979/−.008126**，三折负；mention增加34TP/96FP、去除55TP/80FP，净−21TP/+16FP；document净−4TP/−3FP。实体对象4315→4311，原nonentities/关联对象不变。该结果排除了“只因top3扩池干扰才没超过强基底”的简单解释；不扫其他cut、类别权重或复合方式，不迁原dev。

完整准备、源快照、四fold全部概率/预测、九cut36输出及时间均保留，未保存HGB对象，需按冻结资源重fit恢复。独立增量审35d9e67e已PASS：244794旧字段/类型/顺序及三均值全部核对，fold0一次重fit13.39秒、99列及所有概率/pred byteexact，其余同引擎实际证据及36decode/table exact、format0；未重4fit/NN。实际校准job后置补录，明确未另采launcher起始UTC；源码自写freeze/start/finished和工具rc0真实保留，不把后置记录冒充预先日志。[完整结果与恢复](../evidence/round2/phenobert-old-pool-results.json)。
