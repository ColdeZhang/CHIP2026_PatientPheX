# 固定五维关系融合可行性：尚未 fit

**定义与隔离可行，但当前无标签五列缓存还缺 F1 的 LR 连续概率导出。** 本次不打开混合 target 文件，不输出新矩阵、不 fit/predict/score，也不访问当前患者优先 F0 终点。此前完整库存和旧 HGB/两列 LR 负或微小收益记录仍有效。

| 唯一列 | 精确定义 | F1 / F0 来源及状态 |
|---|---|---|
| `old9B_selected` | 固定候选 `(pmc_id, patient_id, concept)` 是否属于旧 SFT 关系集合；0/1，不是置信度 | fullpred `ffb2c54a / 63aefe68`，strict32→F1 /48→F0，已核完整同池 |
| `reverse_selected` | 同一 adapter、逆序候选输入所得关系集合的0/1选择 | `bc9d1b26 / 0c5cea7e`，仍32/48来源；不是新增独立基座；已核同池且确有不同决策 |
| `binary_probability` | frozen LM head token0/1两列 logits 的原生 softmax 正类概率 | `b14ef30b / 6fe3a795`，**两折都用同一 binary32**，只训练F2F3，排除F0/F1；不补训、不冒称48 |
| `oldLR_probability` | 旧 LR `predict_proba(... )[:,1]` 对已选实体每次出现评分，再按患者/概念取 **max**；原 .55 hard选择另存，本方案不另加它 | F0纯 rows `10ea2de8`完整；F1只在混 target 的 `meta_F1.rows.jsonl`（manifest SHA `f055b532`）中，**本次未打开/哈希它**，所以未声明实际无标签列已齐 |
| `rules_score` | 固定采用 `rule_selected_mean`：`aggregate_relation.document_rows` 同概念候选出现中，`pid in base.assign_patients(...)` 布尔值的算术均值 | F1纯上游 `217c380f` / F0纯 rows `10ea2de8`，两折已核全key。它是候选出现投票比例，**不是校准概率，也不等同最终rules hard选择** |

规则分数基于聚合器的候选出现池，不只最终保留的实体跨度；候选来源仍为 F1 train32(F2F3) / F0 train48(F1F2F3)。生成源先核纯输入、再按概念/患者对称聚合，未按标签挑出现位置。只选 `mean` 一列，不把 max/min/context 展开成新搜索。

实际新增纯缓存检查 **0.214772秒**：F1 1,557 / F0 1,166 固定key精确覆盖、两文件均无 target/labels/answer/gold；规则均值有限且∈[0,1]，分别89/80种值、357/267行严格在(0,1)。F0 LR概率有限且∈[0,1]，1,163种值，全部与既有 `.55` hard标记一致。旧3列与模型来源复用上一份已核库存，不重做其模型/概率推断。完整SHA/结果见 `unlabeled-column-check.json`。

## F1 LR 缺口及安全导出边界

`prepare_fusion_inputs.py` 的原源证明：先从严格train32 LR计算概率，按出现取max，写入 `features.relation_lr_probability`，然后**另在顶层**添加 `target`；该features不依赖target。这支持从冻结混合文件仅投影 `pmc_id / patient_id / concept / features.relation_lr_probability` 的语义可行性。现有模型文件可以重推，但那是新预测，不是当前准备范围，也没必要为了导出重复跑。

后继若明确授权该投影，应以独立受限导出过程、固定 `f055b532…` 原文件、严格字段白名单输出无标签新JSONL；只写四项、禁止target进入输出/日志/排序/过滤，拒绝缺key/重复key/非有限/越界，绑定原1557纯prompt key并保存新文件SHA。**普通JSON读取仍会打开包含target的容器，不能把“随后丢弃target”说成从未读过标签字节。** 因此本轮仅分析源/schema，不运行该导出，也没有伪造已核第五列。

这五列来源并非旧两列LR换名：它保留旧SFT和binary概率，增加已有不同决策的reverse、独立旧LR概率、规则出现均值。后两类在旧303列HGB中已经存在；新的具体组合是五个固定标量的低容量模型，不能称首次学习融合或预期必有提升。二进制/逆序并不保证统计独立。

未来最小模型可仅用5个原值、固定L2 LogisticRegression、截距和一个预先固定解码政策，不加context/交互/网格；当前未生成正式fit计划、未选阈值、未授权fit。F1仍是已使用的开发折，F0也有已知历史，不能称盲测。强对照必须同时列旧SFT48 **.6954439475814587** 和已通过的旧∪reverse **.70468744005924**，不能只赢旧SFT就建议dev；具体接受条件应在后继任何fit/新质量前冻结。

`plan.json` 先于本次纯缓存检查保存；新增只读取两个已由源和旧manifest证明无标签的文件。`source-snapshots/` 仅小源/metadata，混合F1 rows、gold、模型、全日志均未复制/读取。未改Git；当前任务在“F1 LR纯列待授权导出、正式fit未授权”处停止。
