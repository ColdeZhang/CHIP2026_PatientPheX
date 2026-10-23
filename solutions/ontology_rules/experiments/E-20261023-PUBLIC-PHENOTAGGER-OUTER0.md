# E-20261023-PUBLIC-PHENOTAGGER-OUTER0：固定政策外折迁移

nested16公开原生实体控制独审取得+.013595473后，root授权一个不同文献fold的固定政策迁移，检验互补召回能否复现。唯一base是outer0 CRF+Sap/rules `b44c10f1…`（947实体，Score .711353521870099），不是sequence的HGB .710359学习基底。原base全部实体和关系保留；直接复用已独审 `assemble_document`，原排序/否定/非重叠及原生阈值不变，不给新折增加规则或扫参。

输入outer0 `ec38c135…`，16篇与原nested16完全不相交，逐对象等于helper train64无标签缓存的对应文献。CPU准备只读64份pmc_id/pmid/patient/full_text，未读本实验heldgold。helper负责同公开原模型补余48篇并复用旧16，ontology不重复NN；必须将64→16记录为精确原缓存行投影，不能伪称原模型直接以outer16为原输入。全量cache和投影产物保留外部，Git保存源与小manifest。

[CPU迁移入口](../run_public_phenotagger_migration.py) 校验缓存来源、完整输入、同一政策/旧源SHA、折互斥后，按文献取原JSONL行，装配并冻结完整pred；随后唯一v2 all/text。比较固定原base，关系指标必须完全相同。独立审查仅新增缓存/投影/终点；不重复nativeNN与旧201变量审查。

当前准备：16/64对象相等、与旧16互斥、947 base实体和实际引用已检查，[准备证据](../evidence/round2/public-phenotagger-outer0-preparation.json)。准备时尚未生成本折预测或评价；随后实际完整终点见下。以同分项/完整冻结/来源独审判定；即使正增益也仍是训练内开发控制，不是原train64/dev16≥.8验收，无原dev或官方操作。

## 缓存完成与最终固定计划

helper唯一48篇native实际完成183.45秒，CPU合并4.07秒，GPU2核空释放。统一64缓存6064行/6064键，旧16各文canonical行字节不变，input-only全坐标/来源契约通过。manifest `adaedb5c…`、canonical `876317bd…`、input `ee252384…`；最终[迁移计划](../evidence/round2/public-phenotagger-outer0-plan.json)明确parent64/target16两SHA及old16/new48谱系。源和计划提交推送后只执行一次已授权CPU终点。

## 唯一终点与结论

21:14:35.593Z先冻结完整pred `19ad8816…`（freeze `53735b44…`），21:14:35.959Z才开始唯一v2。64→16原缓存行投影 `e33462eb…`，1418候选拒绝1091原实体重叠、55否定、31新增重叠，接受241实体；全部原实体及非entities字段原样，关系计数完全不变，格式0错/0警。

| 指标 | 原base | 固定补充 | 差值 |
|---|---:|---:|---:|
| menF1 | .724105461 | .725581395 | +.001475934 |
| docF1 | .728571429 | .735985533 | +.007414105 |
| micF1 | .704419890 | .704419890 | 0 |
| macF1 | .688317308 | .688317308 | 0 |
| Score | .711353522 | **.713576032** | **+.002222510** |

新增89TP/152FP（精度36.93%），文档+50TP/76FP；关系TP/FP/FN、micro/macro完全不变。独立CPU复装/完整v2/格式与冻结顺序终审见结果索引，不重复native模型推断。

**结论：固定原生实体补充在两组不同训练内开发文献上方向均为正，但本折增益显著变小，误报仍限制收益。** 两组CRF/Sap基底训练规模分别nested32/outer48，不把总分或增幅直接平均为统一泛化指标；也不宣称原dev≥.8达标。关闭本次固定迁移，不扩阈值/规则/原dev；后续严格OOF可学习删换由sequence单独推进，避免与本固定实体only控制混淆。

独审报告 SHA `a78d20138b371d8675836b26dde7cc49c736024d9a904601a88dca8375fa1ecf`；[完整小结果与恢复引用](../evidence/round2/public-phenotagger-outer0-results.json)。
