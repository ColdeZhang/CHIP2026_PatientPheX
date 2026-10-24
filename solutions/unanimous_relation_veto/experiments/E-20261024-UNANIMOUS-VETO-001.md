# 唯一四流明确阴性删除控制

F1 B4固定524/.701325377212；S1、S2、原生Qwen3及原strict响应code每流在同patient/CID原子上有完整明确负，且pure grounded无positive，才删除。覆盖/弃答/compound政策见decision.json；31项合成合同区分缺失、任一正、cap、冲突、多患者、NIL及compound部分删除。

预先源码/plan push，再唯一装配。0删除必须原预测逐字相同、0gold/0v2关闭；否则完整16篇/39患者/1189实体freeze，再1新v2/旧0，报告删TP/FP、macro及总Score。严格胜B4才同政策F0当前343/.728195628911；两折正才讨论full64，当前无原dev/官方动作。

只复用原完整canonical和终止元数据，不重parse原响应或调用模型。各历史基座/候选排相应held；此F1/F0均研发折，不称盲测。未完成时不冒称收益。
