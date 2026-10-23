# 首日公共评测修正：复合 HPO 患者关联

核对时间：2026-10-23 11:20 UTC。状态：评测适配已修复，旧指标与预测保留；不训练模型、不调阈值。本文是首日各路线共用的评测维护记录。

## 目标、证据与改动

目标是让独立 dev16 比较正确处理真实金标中的患者关联，而不是通过改口径提高分数。成功标准：合法复合 HPO 关联与等价原子预测一致，部分预测正确计 TP/FN，普通文本与非 HPO 字符串原样保留，既有实体指标不变。

旧 `patientphex-public-approx-v1` 已拆分实体中的复合 ID，但 `associations()` 将 `phenotype` 列表直接转集合。真实 dev16 有 **6 个合法 HPO 复合关联项**，涉及 5 名患者、5 篇文献，拆分为 15 个原子 ID；旧实现无法将它们与合法的原子预测对应。

[evaluate.py](../../scripts/evaluate.py) 升至 **`patientphex-public-approx-v2`**：仅将完整匹配 `HP:[0-9]{7}(;HP:[0-9]{7})+` 的关联字符串拆分，并在每位患者内去重。自然文本、MESH、混合字符串、畸形 ID 保持原样。输出仍须遵守现有 validator 的原子关联格式。版本、拆分策略均写入新指标文件；旧 `metrics.json` 继续代表 v1，不覆盖、不改写。

该修正依据公开规则的复合概念独立匹配要求及项目已使用的原子输出格式，解决实现内部不一致；**v2 仍是公开规则近似评测，不是官方评测代码**。正文筛选、`-1` 的 docF1 表示、HPO 别名及异常金标处理仍有不确定性，不能把本次修正视为官方口径已完全确认。

## 同一预测的重算结果

全部使用相同 `dev16.gold.jsonl`（16 篇、36 位患者），默认 `body_scope=all`、`unlinked_document_mode=text`。v2 只变更关联指标；有旧文件的 5 个运行，其 menF1/docF1 均逐项不变。

| 预测运行 | menF1 | docF1 | v2 micF1 | v2 macF1 | v1 Score | v2 Score |
|---|---:|---:|---:|---:|---:|---:|
| ontology-rules/dev-v1 | 0.725815 | 0.737166 | 0.530303 | 0.457105 | 0.615095 | 0.612597 |
| fuzzy-ranker/dev-v1 | 0.665181 | 0.701987 | 0.587135 | 0.569492 | 0.632921 | 0.630949 |
| fuzzy-ranker/dev-v2 | 0.647191 | 0.681564 | 0.553699 | 0.542261 | 0.607838 | 0.606179 |
| sequence-crf/crf-dev-v1 | 0.732540 | 0.745303 | 0.648472 | 0.612620 | 0.685654 | 0.684734 |
| case-retrieval/dev-cpu | 0.731657 | 0.749220 | 0.542420 | 0.430774 | 0.615881 | 0.613518 |
| semantic-neural/dev-minilm | 0.690076 | 0.703665 | 0.380420 | 0.297210 | 本次核对时无旧文件 | 0.517843 |

`fuzzy-ranker/dev-v2-repro` 的预测 SHA-256 与 `dev-v2` 完全相同，也生成独立 v2 指标，结果相同。共保存 7 份新评测。v1 到 v2 的分数变化来自金标拆分后的分母及匹配计数变化，不能解释为模型能力变化；前三路线排序不变。`dev-minilm` 是已产生预测的 CPU 首版，不代表其他进行中的语义模型或 GPU 路线结果。

## 真实数据检查与保留的不确定性

- dev16 有 4 条完全重复的正实体记录；现有评测按跨度和拆分 ID 去重，本次不变。
- dev16 正实体包含 2 次可由指定本体 alt_id 解析的废弃 ID：`HP:0002370 → HP:0002311`、`HP:0001388 → HP:0001382`；还有 6 个 MESH 评价单元，以及 1 次非 HP:0000118 分支的 `HP:0005268`。本次不改金标、不自动规范化这些实体 ID。
- dev16 另有 1 个 `MESH:D009069;MESH:D014786;MESH:D003072` 关联串，v2 原样保留。不能为了拆 HPO 而对所有原文字符串直接 `split(';')`。
- dev16 金标中 14 条正实体位于 TITLE/front、136 条位于 ABSTRACT/abstract，全部与原始文本切片一致。现有 `all` 与启发式 `body` 敏感性仍同时报告，具体官方正文边界未确认。
- dev16 有 7 个 `-1` 正实体评价单元；36 位患者均有非空金标。空患者行为由反例测试覆盖，不能从该 dev 单独验证。
- train80 与官方原始训练集的 JSON 对象逐篇完全一致；异常并非本地划分过程引入。train64/dev16、train80/B 的 PMC 交集均为空，输入无实体或关联答案。

## 训练规范化遗漏：记录待改，不在本次重训

只读审查时，fuzzy 的 `gold_associations()` 与 CRF 的 `fit_relation()` 同样将关联金标保留为整串，而候选按原子 ID 匹配。这会把部分复合关联正例当成负例，属于训练标签处理遗漏，不是 dev 泄漏。

| 训练范围 | 合法 HPO 复合关联项 | 拆分原子数（未跨项去重） | 涉及患者 | 涉及文献 |
|---|---:|---:|---:|---:|
| train64 | 26 | 63 | 20 | 17 |
| train80 | 32 | 78 | 25 | 22 |

后续应在各路线训练入口采用相同的受限规范化规则，并重新生成独立 dev 与 full80 B 产物；不能给已冻结模型追写“已修复”。本次只评估原预测，未改训练代码或阈值。

## 复现、产物与验证

代码 SHA-256（`scripts/evaluate.py`）：`ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781`。金标 SHA-256：`c1f0f24a58b62bb898eb672550dc58247ddb9a0e8768c6f899812cfc327cc050`。本体检查使用官方 v2026-06-23，SHA-256 为 `a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b`；本次评分未增加本体转换步骤。

新文件均位于仓库外 `/workspace/patientphex-runs/day1-20261023/<预测运行>/metrics.eval-v2.json`。完整输入/输出 SHA-256、实际命令、旧新分项及旧文件校验值保存在 `/workspace/patientphex-runs/day1-20261023/evaluator-v2-review-manifest.json`，该 manifest SHA-256：`e2b17c0ba342d6fff97e5becb9d461ea66202122018b273da579ac46709de643`。这些小型指标可由下列命令重建；全量预测不入 Git。

```bash
python scripts/evaluate.py --gold /workspace/patientphex-data/day1/dev16.gold.jsonl \
  --pred /workspace/patientphex-runs/day1-20261023/<预测运行>/<预测文件> \
  --output /workspace/patientphex-runs/day1-20261023/<预测运行>/metrics.eval-v2.json
python -m unittest discover -s tests -v
```

预测文件通常为 `predictions.jsonl`；CRF 为 `pred.jsonl`。新输出路径已存在时，应校验版本和预测哈希后复用，避免覆盖其他版本。

验证结果：14 项测试通过，新增反例覆盖完整/部分复合关联匹配、跨项去重，以及自然文本、MESH、混合串、畸形和非 ASCII 数字不误拆。7 个完整 dev 预测已重算；原有 5 个 `metrics.json` 的哈希前后不变。本轮评测为 CPU 短任务，未训练、未使用 GPU、未提交比赛预测。
