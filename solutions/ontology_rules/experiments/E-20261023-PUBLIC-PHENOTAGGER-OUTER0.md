# E-20261023-PUBLIC-PHENOTAGGER-OUTER0：固定政策外折迁移

nested16公开原生实体控制独审取得+.013595473后，root授权一个不同文献fold的固定政策迁移，检验互补召回能否复现。唯一base是outer0 CRF+Sap/rules `b44c10f1…`（947实体，Score .711353521870099），不是sequence的HGB .710359学习基底。原base全部实体和关系保留；直接复用已独审 `assemble_document`，原排序/否定/非重叠及原生阈值不变，不给新折增加规则或扫参。

输入outer0 `ec38c135…`，16篇与原nested16完全不相交，逐对象等于helper train64无标签缓存的对应文献。CPU准备只读64份pmc_id/pmid/patient/full_text，未读本实验heldgold。helper负责同公开原模型补余48篇并复用旧16，ontology不重复NN；必须将64→16记录为精确原缓存行投影，不能伪称原模型直接以outer16为原输入。全量cache和投影产物保留外部，Git保存源与小manifest。

[CPU迁移入口](../run_public_phenotagger_migration.py) 校验缓存来源、完整输入、同一政策/旧源SHA、折互斥后，按文献取原JSONL行，装配并冻结完整pred；随后唯一v2 all/text。比较固定原base，关系指标必须完全相同。独立审查仅新增缓存/投影/终点；不重复nativeNN与旧201变量审查。

当前准备：16/64对象相等、与旧16互斥、947 base实体和实际引用已检查，[准备证据](../evidence/round2/public-phenotagger-outer0-preparation.json)。最终cache终点未到，尚未生成本折预测或评价。完成后以同分项/完整冻结/来源独审判定；即使正增益也仍是训练内开发控制，不是原train64/dev16≥.8验收，无原dev或官方操作。

## 缓存完成与最终固定计划

helper唯一48篇native实际完成183.45秒，CPU合并4.07秒，GPU2核空释放。统一64缓存6064行/6064键，旧16各文canonical行字节不变，input-only全坐标/来源契约通过。manifest `adaedb5c…`、canonical `876317bd…`、input `ee252384…`；最终[迁移计划](../evidence/round2/public-phenotagger-outer0-plan.json)明确parent64/target16两SHA及old16/new48谱系。源和计划提交推送后只执行一次已授权CPU终点。
