# 公开SapBERT真实候选匹配特征控制

独立于已关闭top3-ID配方：检验候选指定ID的公开语义匹配是否弥补真实信息缺失。固定扩展池、不新增span/ID、不覆盖旧similarity或六rawtop3特征，仅加入public_sap_cosine/public_sap_observed。公共HPO name/EXACT按canonical及本体声明有效alt映射至全部ID，排除训练best_id/训练别名；atomic max alias cos，compound min各atomicmax，NIL/无alias显式missing。

先实际256query缓存二进制SHA/shape/norm+文本digest和公开bank全SHA；仅缺799词面用同原Sap FP16/CLS/max32编码一次，fresh .3 A30 GPU0/UUID07726469，含加载≤60秒，独立缓存不覆盖旧。现有向量以排序缓存首个词面为确定来源。

四fold沿同严格nested来源/HGB160/seed20261023，fit+infer总≤240秒；九点.1–.9从事前列入，36完整预测共同freeze后一次选择pooled两实体F1均值（tie近.4再高），与旧best.780192596比较；另保留固定.3与原top3负终点分项。所有nonentities旧cut.3原样；解码校准≤30秒。校准是训练侧拟合，非泛化/完整Score/达标，不自动原dev。无收益关闭。

[阶段计划/资源/sourceSHA](../evidence/round2/public-sap-feature-plan.json)。当前仅准备，尚未GPU/fit/质量。

CPU阶段首次启动早于异步文件部署完成，在读取脚本前因文件不存在退出；未读模型/数据/GPU。失败及job完整保留，transport增加部署source/plan收据与前阶段完成门。数学/请求/实际vector源未变。之后CPU已完成：256个查询缓存实际二进制SHA/shape/有限归一值/有序文本digest通过，公开bank SHA与旧报告一致，模型所有实际文件含FP16权重SHA全部匹配。34,094词面/136,409已有phrase-ID，缺799；[CPU证据](../evidence/round2/public-sap-feature-cpu-preparation.json)。
