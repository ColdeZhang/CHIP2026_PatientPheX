# 当前实施决定

[原始CPU报告](REPORT.md)、[来源](sources.json)、[训练侧库存](census.json)按原字节保存；报告末尾的初始端点建议已在训练前被以下决定取代。

新工作线为 [explore/mention-canonical-normalization](https://github.com/ColdeZhang/CHIP2026_PatientPheX/tree/explore/mention-canonical-normalization)，base `bc939eeb`。仅已有Qwen3-8B，不调用外部商业模型API。当前为CPU实施，尚无新GPU/训练/held质量。

- train32完整2353可表示行，2222 HPO、44 NIL、87 NO；16compound与8条分支外正例排除明确记录。模型输入不含旧预测ID、候选或held标签。
- held F1跨度池固定为强8cb实体跨度与GLiNER2.5已接受886原子对应跨度的并集，精确去重后1416行；不按held错误挑样本。
- 非空canonical-name列表表示HPO，空列表表示NO，独立对象 `{"NIL":"原quote"}` 表示NIL，必须互斥并严格匹配原quote；完整实际token/mask及约束接口仍需核验。
- 两个预定终点：强B实体保留并加入全部新头输出；同跨度池按新头输出构造完整实体集合。文本、患者、关系全部保持，两个完整预测共冻后各一次v2；任一总Score严格胜强B才候选，报告全部正负分项。
- 不复活旧逐窗quote抽取；该路线两折已有正式负结果。可复用现成native训练循环和名称约束代码，不能把新模板CPU通过称作显存或模型质量通过。

准确运行状态、源与后继实测只由新分支台账维护；大目标、全输入输出留仓库外。
