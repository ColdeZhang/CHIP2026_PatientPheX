# 本地仓库与官方规则核对

核对日期：2026-10-23。官方依据为[天池赛题原文](sources/2026-10-23/tianchi-information.md)及[官方 GitHub 说明](sources/2026-10-23/official-github/README.md)。本文件记录仓库现状，相关代码和数据未在本次资料整理中修改。

## 本地数据

| 文件 | 文档 | 患者 | 患者 mention | 表型实体 | 关联列表项 |
|---|---:|---:|---:|---:|---:|
| [data/train/PatientPheX-train.jsonl](../../data/train/PatientPheX-train.jsonl) | 80 | 209 | 677 | 6,027 | 2,307 |
| [data/B/PatientPheX-V1-B.jsonl](../../data/B/PatientPheX-V1-B.jsonl) | 100 | 244 | 785 | 0 | 0 |

- 两份 JSONL 均可解析，各自 `pmc_id` 无重复，包含 `pmc_id`、`pmid`、`patient`、`full_text`、`entities`、`association` 六个顶层字段。
- B 集的 `entities`、`association` 每行都是空列表；患者和部分患者指称已给出。
- 未发现 A 榜开发集文件。
- train 的关联列表含重复项：原始列表项共 2,307，逐患者去重后为 2,130 对。12 位患者的金标表型集合为空。
- train 包含 69 个分号分隔的复合表型实体。
- `full_text` 中部分段落还有可选 `title` 字段。坐标应以各段 `offset` 为基准。

## HPO 版本差异

官方指定 **v2026-06-23，HP:0000118 分支**；本地 [data/hp.obo](../../data/hp.obo) 第 2 行是：

```text
data-version: hp/releases/2026-09-01
```

因此不能将当前本体文件称为比赛指定版本。现有增强词典 [dict/hpo_dict_augmented.json](../../dict/hpo_dict_augmented.json) 的 `combined` 含 21,155 个 surface forms；仅凭该规模不能证明它的本体版本或分支筛选符合比赛要求。之后若要按官方约束开展实验，应先核对词典来源及版本。

## 本地评测脚本差异

现有 [src/evaluate.py](../../src/evaluate.py) 的输出名称与官方相同，但计算并不完全一致：

| 项目 | 官方规则 | 当前本地实现 | 代码位置 |
|---|---|---|---|
| menF1 实体匹配 | 边界与 HPO ID 都正确 | 只比较 `offset`、`length`，不检查 HPO ID | `entity_match`，第 12–15 行 |
| docF1 | 每文档 HPO ID 集合比较，跨文档累加 TP/FP/FN 后算 F1 | 沿用跨度匹配，逐文档 F1 平均；预测或金标实体为空的文档会跳过 | `evaluate_entities`，第 43–56 行 |
| 否定表型 | 按 `note="NO"` 规则排除或计 FP | 不读取 `note` | `evaluate_entities`，第 22–37 行 |
| 复合实体 | 分号拆分成独立 HPO 评价单元 | 不拆分 ID | `entity_match` / `evaluate_entities` |
| 正文筛选 | 非正文、参考文献不参与 | 不读取 `full_text` 或章节类型 | `evaluate_entities` / `evaluate_association` |
| 空患者集合 | 金标、预测均为空时，该患者 F1=1 | 同一情况 F1=0 | `evaluate_association`，第 85–92 行 |

另有本地行为需要注意：

- 预测与金标通过 `zip(pred_docs, gold_docs)` 按行配对，既不校验 `pmc_id`，也不校验文件长度；较长文件尾部会被忽略。
- 关联列表转为集合，重复表型自动去重；只遍历金标患者，额外预测患者不参与统计；同一 `patient_id` 多条记录会被后者覆盖。
- 总分确实是四个输出 F1 的等权平均，但相同公式不能消除各分项定义的差异。

本地分数可用于当前脚本下的回归对比，不能直接作为官方成绩。本次已按官方定义修正仓库 README 的指标说明，评测脚本本身保留原样。训练集上的自拟合分也不应外推到 B 集。

## 尚未确认的细节

- 官方公开 GitHub 文件树只有 README、README_EN 和 LICENSE，本次未获得官方评测代码。
- 官方说明未展开 `-1` 实体如何计入 docF1 概念集合，也未列出正文筛选的完整 `section_type/type` 规则。
- 公开数据接口未提供完整压缩包名或下载链接；本次未获取新的数据包及提交样例文件。
- 赛后评测的每日次数、登录账号的具体权限、群内通知未获取。
- 仓库 README 中旧实验的分数、方法来源及“达标”等说法属于历史记录，本次公开榜单不能据此对应到用户队伍或验证每次实验成绩。

本地文件校验值记录于 [来源清单](sources/2026-10-23/manifest.json) 的 `local_repository_checks`，用于后续确认这些核对结论适用的文件版本。
