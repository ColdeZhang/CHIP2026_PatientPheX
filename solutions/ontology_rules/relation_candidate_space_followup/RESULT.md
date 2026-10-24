# 已选 Task1 概念的关系提示覆盖审计

2026-10-24，仅 F0/F1 训练内开发缓存。目标是判断“把 Task1 已选概念补给关系模型”是否有足够潜在收益；没有改本次官方 B，没有新模型、训练或评分调用。

| 固定折 | Task1 canonical 文档 HPO | 现有 old＋public2 合法文档 HPO | Task1 缺口 | 对应全部患者候选对 | 覆盖参考 FN |
|---|---:|---:|---:|---:|---:|
| F0 | 477 | 717 | 1 | 6 | 0 |
| F1 | 680 | 1033 | 2 | 9 | 2 |

唯一 CPU 计数 0.716438 秒，外层 0.818804 秒，rc0。先保存双折全部 input-only 缺口、来源分组与合法空间，再读取对应训练折 gold。合法对严格按每条实际提示的 code_map × 该条 patient_ids 构建；旧实体与旧提示空间、public2 原生 canonical 差集与实际 public2 提示空间均完全一致，没有额外患者覆盖缺口。完整空间留仓库外，[小型冻结记录](input-only-freeze.json)绑定其 SHA。

三个缺失文档概念均属于“完整 public2 不提供、BioBERT 提供”，但全部已选 exact 候选同时带 semantic_source 与 biobert_neural；不能把它们归为 BioBERT 的独立因果贡献。三者观测到的 BioBERT hybrid 分数均有 >.95。现有 public2 本来就没有第二次 hybrid >.95 筛选，不能以低分筛选解释缺口。F1 Task1 来源训练48文献，而旧关系候选来源训练32文献，也是无法隔离模型贡献的差异。

关系参考是 root 指定的 public-new LGB-supported 固定开发端点 F0 `94504b97`（Score .711507580027）、F1 `2a65056f`（.686338981272），并非完整901两折组合。只 AST 抽取 evaluator 的 associations/compound 正则做原字面集合计数；TP/FP/FN 精确复核为252/63/127及366/105/154，未调用 evaluate/report、未算新Score。F0/F1 全部提示空间之外的 literal FN为63/52，其中HPO关联为47/45；当前Task1已选缺口只能覆盖0/2，说明主要供给问题不在这三个额外概念。canonical诊断另外保留，未替代literal分母。

**建议**：本轮关闭专门的“仅补Task1已选概念”GPU控制，2/281参考FN只是输入供给机会，并不保证模型能恢复。保留事前完整集合；如未来统一关系候选来源，可作固定、可证伪的控制：一次性补全三条 input-only 概念，沿各折原模型/codec/证据/生成与支持门，两完整端点共同冻结后仅按总Score判定。不能只挑两个有gold支持的F1概念，且当前没有授权或实施该扩展。已检查的历史范围未发现完全等价控制，详见来源盘点；这不等于全仓所有历史均被穷尽。

证据：[聚合结果](result.json)、[实际执行](count-execution.json)、[事前计划](count-plan.json)、[来源盘点](input-source-inventory.json)。没有原dev gold读取、阈值扫描、逐错例规则或新预测。
