# 公开SapBERT真实候选匹配特征控制

独立于已关闭top3-ID配方：检验候选指定ID的公开语义匹配是否弥补真实信息缺失。固定扩展池、不新增span/ID、不覆盖旧similarity或六rawtop3特征，仅加入public_sap_cosine/public_sap_observed。公共HPO name/EXACT按canonical及本体声明有效alt映射至全部ID，排除训练best_id/训练别名；atomic max alias cos，compound min各atomicmax，NIL/无alias显式missing。

先实际256query缓存二进制SHA/shape/norm+文本digest和公开bank全SHA；仅缺799词面用同原Sap FP16/CLS/max32编码一次，fresh .3 A30 GPU0/UUID07726469，含加载≤60秒，独立缓存不覆盖旧。现有向量以排序缓存首个词面为确定来源。

四fold沿同严格nested来源/HGB160/seed20261023，fit+infer总≤240秒；九点.1–.9从事前列入，36完整预测共同freeze后一次选择pooled两实体F1均值（tie近.4再高），与旧best.780192596比较；另保留固定.3与原top3负终点分项。所有nonentities旧cut.3原样；解码校准≤30秒。校准是训练侧拟合，非泛化/完整Score/达标，不自动原dev。无收益关闭。

[阶段计划/资源/sourceSHA](../evidence/round2/public-sap-feature-plan.json)。实际向量及两特征准备已完成；正式fit/质量尚未开始。

CPU阶段首次启动早于异步文件部署完成，在读取脚本前因文件不存在退出；未读模型/数据/GPU。失败及job完整保留，transport增加部署source/plan收据与前阶段完成门。数学/请求/实际vector源未变。之后CPU已完成：256个查询缓存实际二进制SHA/shape/有限归一值/有序文本digest通过，公开bank SHA与旧报告一致，模型所有实际文件含FP16权重SHA全部匹配。34,094词面/136,409已有phrase-ID，缺799；[CPU证据](../evidence/round2/public-sap-feature-cpu-preparation.json)。

同源799词面一次编码含加载5.340秒（2026-10-23 22:22:12–18 UTC），向量799×768／SHA eae4acb2…；22:24:28 fresh目标GPU4MiB且无计算PID，已释放。候选cosine CPU4.164秒，136,409唯一phrase-ID中135,250 observed、1,159明确missing，共548,942次alias点积，无全库检索。747,178重复池行仅追加两列，旧键/字段/顺序/类型完整保留；21个固定SHA抽样与compound/NIL边界的实际query和全部alias向量留外部供独审。准备耗时与网络传输另记，不冒称GPU占用。

[实际准备证据/恢复](../evidence/round2/public-sap-feature-preparation.json)，[正式四fit与九点冻结计划](../evidence/round2/public-sap-feature-formal-plan.json)。按旧实际145.395秒及两列/额外开销估163.25秒；硬限仍四fit合计240秒。所有新预测冻结后才读质量标签。

四fold实际全部rc0、合计 217.670秒，仍在原240秒内；前两折较慢，主线程曾在未读质量前允许资源恢复，但后两折加快，因此没有更改live source/预算、没有重跑或使用额外授权。八完整预测已共同冻结；九点具体输出绑定见[校准计划](../evidence/round2/public-sap-feature-cut-plan.json)，接着只做此一次校准。


## 唯一终点与决定

36完整预测共同冻结后，一次九点训练侧校准实际3.883秒。最佳仍为cut .3：

| 对照 | menF1 | docF1 | 两实体F1均值 |
|---|---:|---:|---:|
| 原Bioformer旧池已校准best .3 | .763831183 | .796554009 | .780192596 |
| 已负top3池固定 .3 | .757437568 | .794195251 | .775816409 |
| 同top3池 + 公开Sap两列 best .3 | .759181020 | .797199737 | .778190379 |

相对原best均值−.002002217，四fold差值依次−.004658694、+.005809171、−.008541409、−.002697283。mention多11TP也多74FP，doc多19TP/25FP。与已负top3固定 .3相比，两分项均改善（均值+.002373970，mention+16TP/+5FP、doc+16TP/+7FP）；但改善不足恢复原最佳水平，不能选弱对照宣称成功。

按事前规则关闭这套配置，不迁原dev、不加cut/特征/新fit。结果是train64侧校准，所有标签参与cut选择；不是独立泛化、完整Score或.8达标。非entities逐对象保持旧cut.3模板（association原源自.4），本次只计算两Task1指标。全部实际源、预算、21数值样本、4fit概率和36预测留外部，[关键结果/恢复](../evidence/round2/public-sap-feature-results.json)。独立终点审查进行中，向量与全池两列增量已通过e3f47aa1；不将准备PASS当质量验证完成。
