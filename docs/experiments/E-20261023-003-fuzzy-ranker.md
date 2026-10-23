# E-20261023-003：字符近似候选检索与监督排序

状态：首版及一次候选训练修正均已完成；选择近似验证较好的 v1 交主线程筛选提交。分支 `day1/fuzzy-ranker`；起始 commit `ac8a752`；v1 源码 commit `3eff30a0cdfc454168ed0eb2d7913945a8766c6b`。主线程统一安排数据划分、评测和平台提交，本路线不操作网站。

## 目标、假设与约束

目标是提高四项比赛分数，不要求原创。假设：基于指定 HPO 的近似候选召回能覆盖词形和表述变化，再用训练折监督的稀疏上下文排序降低模糊匹配误报；独立学习患者归属可避免段内笛卡尔分配。

仅使用 HPO v2026-06-23 的 HP:0000118 非废弃后代，以及指定训练折的实体/关联标注。验证输入不读取标签；B 无金标。不得使用旧实现、旧增强词典或旧预测。当前 CPU 单线程，不依赖 GPU，不使用外部人工表型标注数据。

成功标准：在统一 64/16 文献划分上跑通，输出通过覆盖、原文切片及 HPO 检查，并按首日计划与本体精确基线比较；近似分数不能称官方成绩。派生资源仅来自本轮 --train 文件。

## 实际调研与方法取舍

2026-10-23 实际读取以下公开实现：

- [scispaCy candidate_generation.py](https://raw.githubusercontent.com/allenai/scispacy/main/scispacy/candidate_generation.py)：使用 TF-IDF 表示与近邻索引检索知识库别名。读取文件 SHA-256 `5d402c6cabf23f4d00c1cee90e23fb999b8d8b253df222b1d659227e13316a66`。
- [scispaCy linking.py](https://raw.githubusercontent.com/allenai/scispacy/main/scispacy/linking.py)：候选相似度阈值筛选，默认阈值 0.7；并未把字符串相似度当患者关联证据。读取 SHA-256 `350e846a6fb664b8926542979f6b038a1f707f8facd10fde595c5f7474601f57`。
- [QuickUMLS README](https://raw.githubusercontent.com/Georgetown-IR-Lab/QuickUMLS/master/README.md)：SimString 近似匹配、Jaccard/Dice/cosine 等相似度与窗口候选。读取 SHA-256 `277eff6c3f7b203a28d338a558fa00060e343c2fedbd3b10187f69997328fe86`。

采用上述检索思想，重新实现 HPO 专用索引；不下载其过期 HPO 或 UMLS 索引，也不复制第三方实现。使用 sklearn（BSD-3-Clause）、RapidFuzz（MIT）等通用库。

首版流程：

1. 从指定 HPO 重新解析 name、EXACT/BROAD/NARROW synonym、alt_id 和后代范围；添加当前训练折实体表面别名。
2. 扫描最长 8 词窗口，保持原始全局字符坐标；精确及简单复数归一匹配，并对同词头候选计算字符 trigram IDF cosine、编辑相似度和词序无关相似度。
3. 用训练折候选的精确 `(offset,end,HPO)` 标签训练稀疏 LogisticRegression；特征包括检索分数、HPO/训练资源来源、上下文、段落类别、否定、字形及跨文档别名支持。训练特征的别名支持剔除当前文档计数。
4. 按实体置信度解决重叠；对每个候选–患者对训练第二个分类器，使用患者 mention 距离、同句/同段、最近前序指称、竞争患者和上下文。实体–HPO 标签必须与候选 HPO 一致。
5. 输入包含的 entities/association 字段不参与预测；最终只输出实体与给定患者的去重关联。

固定初始阈值：实体 0.60，关联 0.55。v1、v2 均未调阈值，验证标签仅参与评分、误差归因及两版选择，没有加入别名或特征。B 运行使用全量 80 篇训练重新拟合。

## 运行与产物

实现：[run.py](../../solutions/fuzzy_ranker/run.py)；依赖：[requirements.txt](../../solutions/fuzzy_ranker/requirements.txt)。

统一入口：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python solutions/fuzzy_ranker/run.py \
  --train TRAIN.jsonl --input INPUT.jsonl --hpo hp.obo \
  --output RUN_DIR/predictions.jsonl --work-dir RUN_DIR
```

每次保存 `model.joblib`、`run.json` 和预测到仓库外 `/workspace/patientphex-runs/day1-20261023/fuzzy-ranker/`。`run.json` 保存真实命令、数据/HPO/源码/输出 SHA-256、阈值、候选数、耗时和最大 RSS。缓存仅在 `--reuse-model` 且训练/HPO/实现哈希一致时使用。

## 数据、检查与结果

统一文献划分 seed `20261023`：64 篇/173 患者训练，16 篇/36 患者独立验证；B 为 100 篇/244 患者。路径为 `/workspace/patientphex-data/day1/{train64.jsonl,dev16.input.jsonl,dev16.gold.jsonl,train80.jsonl,B.input.jsonl}`，HPO 为 `/workspace/patientphex-data/hp.obo`。数据及输出完整 SHA-256、大小、实际命令、指标和资源摘要见 [results.json](../../solutions/fuzzy_ranker/results.json)。HPO 实际解析 19,120 个有效分支概念；训练别名仅来自相应 `--train`。

- Python 语法检查已通过。
- 合成边界/否定/患者覆盖检查通过：两篇合成训练病例、第三篇独立输入；仅保留非否定 microcephaly，归属 P1，P2 输出空集合，所有 offset 切片与原文一致。首轮训练+推理 5.70 秒。
- 输入泄漏反例通过：向推理输入加入错误 entities/association 金标后，预测文件与无标签版本逐字节相同；缓存恢复推理 1.85 秒，最大 RSS 458,792 KiB，模型 4,645,949 bytes。此检查只验证工程性质，不代表真实比赛效果。
- 统一验证 v1、v2 及全训练 B v1 均通过格式校验：文献/患者完整覆盖、原文 offset 切片、HPO 版本与分支合法，0 错误、0 警告。
- 官方提交：无，子代理不提交。

以下为 `patientphex-public-approx-v1` 的独立验证近似分数，**不是官方成绩**：

| 运行 | menF1 | docF1 | micF1 | macF1 | Score | 拟合+推理秒 | 最大 RSS KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev-v1 | 0.665181 | 0.701987 | 0.591284 | 0.573232 | **0.632921** | 16.10 | 708,348 |
| dev-v2，训练候选排除当前文献别名 | 0.647191 | 0.681564 | 0.557692 | 0.544906 | 0.607838 | 20.73 | 700,420 |
| v2 − v1 | −0.017990 | −0.020423 | −0.033592 | −0.028327 | −0.025083 | | |

v1 实体 P/R=0.866473/0.539783；患者关联 P/R=0.624378/0.561521。官方未知细节的近似口径敏感性：文档 `-1` ignore/shared_id 得 0.634484/0.633504，正文范围变体得 0.632560，仍需官方测评确认。

### 候选构建问题及一次修正

v1 训练候选使用完整训练折别名，虽特征计数排除了当前文档，但当前金标别名仍将训练正例变为 exact，造成训练与推理候选分布不同。诊断发现 dev-v1 的 689 个输出全部 exact，不能把 v1 分数解释成模糊召回已经有效。

v2 在每篇训练文献生成候选时，排除只由该文献提供的别名；官方 HPO 别名及其他训练文献的支持仍保留。没有访问 dev 来构建资源。训练实体候选/正例从 9,748/4,710 变为 7,211/3,235，关联候选/正例从 13,437/4,434 变为 9,656/3,384。统一推理候选两版均为 1,738 个，修正只改变拟合分布。

| 误差诊断 | v1 | v2 |
|---|---:|---:|
| 输出 exact / fuzzy 数 | 689 / 0 | 667 / 7 |
| 严格正确 exact / fuzzy 数 | 599 / 0 | 576 / 2 |
| 缺候选导致漏检 | 310 | 310 |
| 候选低于阈值导致漏检 | 166 | 179 |
| 否定过滤导致漏检 | 23 | 23 |
| 重叠或 ID 选择导致漏检 | 12 | 20 |
| 关联漏检中概念未被抽出 / 已抽出 | 153 / 43 | 170 / 45 |

诊断脚本 [analyze.py](../../solutions/fuzzy_ranker/analyze.py) 使用严格 split-HPO span 匹配；`-1`/重复项细节与共同评测略有不同（如 v1 正确实体诊断为 599、共同评测为 597），因此不替代正式的共同指标。

**结论**：候选排除修正让 fuzzy 真正产生了 2 个正确输出，但总体下降 0.025083，失去的 exact 召回及关联收益更多。首日选 v1 原产物提交，不因一次修正无收益否定整条技术路线；v2 保留为失败证据。当前得分应归因于训练折别名、监督实体筛选和患者归属模型，而非未经证实的 fuzzy 增益。候选召回仍是主要上限，后续可在新的训练内部验证上研究扩大召回、分层正负采样或语义编码；本轮不继续扫 dev 参数。

进一步区分 fuzzy 子类型：v2 有 6 个字符近似输出、其中 1 个正确（`intellectual development disorder` → `intellectual developmental disorder`）；另有 1 个词形归一输出且正确（`dysmorphic face` → `dysmorphic faces`）。因此字符近似的独立正确贡献目前只有 1 个。错误近似如 `recessive developmental disorder` → `pervasive developmental disorder` 提醒我们不能把高字符串相似度直接当语义等价。

### 交付与复现

选定 B 文件：`/workspace/patientphex-runs/day1-20261023/fuzzy-ranker/full-v1/B.predictions.jsonl`，594,642 bytes；SHA-256 `a990730984eeb0f5d1f28f7282659141bcbbec548af031606dfe020fb95ac10a`。100 篇、244 患者、4,411 实体、2,391 关联；训练+推理 26.61 秒、最大 RSS 724,552 KiB。该文件保持不可覆盖，主线程登记提交凭据。

`dev-v1`、`dev-v2`、`full-v1` 全量预测、模型和日志在外部目录保留；Git 仅保存源码、小型摘要、哈希及复现命令。每次改动使用新运行目录，不能覆盖交付文件。首次 v1 命令来自 `3eff30a`，源码 SHA-256 `5f3614f3859a74d98dc989a963ed32f34c131f31ae30e40073006bb2b69d3c5f`；v2 实际运行源码 SHA-256 `7cfa3cb838f003eb5c679fa5947b9947c812aa95f005e396ee7a41a7531906fa`。主线程维护台账和 HANDOFF；已上报“训练特征排除自身支持，仍不能解决候选自包含”的新坑。

v2 源码快照在 `f1472b8`。当前统一 CLI 用 `--training-candidate-policy full-index`（默认）复现 v1，用 `leave-one-document-out` 复现 v2，策略纳入缓存指纹。当前源码 SHA-256 `72a5a992c830a9da52d34398bba70ac0fa45ba4eead1da6f85fb232873efad6c`；`full-v1-repro` 重新拟合得到的 B 与原交付逐字节一致，耗时 31.27 秒、最大 RSS 726,504 KiB；`dev-v2-repro` 也与原 dev-v2 逐字节一致。两种显式模式的重建验证均通过，没有改变原产物。共享基础 `90d52d0` 已 cherry-pick，11 个评测边界测试通过。

例如，在新目录重建选定 B（先安装 requirements 中的固定版本）：

```bash
python solutions/fuzzy_ranker/run.py \
  --train /workspace/patientphex-data/day1/train80.jsonl \
  --input /workspace/patientphex-data/day1/B.input.jsonl \
  --hpo /workspace/patientphex-data/hp.obo \
  --training-candidate-policy full-index \
  --output NEW_RUN_DIR/B.predictions.jsonl --work-dir NEW_RUN_DIR
python scripts/validate_predictions.py \
  --input /workspace/patientphex-data/day1/B.input.jsonl \
  --pred NEW_RUN_DIR/B.predictions.jsonl --hpo /workspace/patientphex-data/hp.obo \
  --output NEW_RUN_DIR/validation.json
```

独立验证将 train 改为 `train64.jsonl`、input 改为 `dev16.input.jsonl`；评分使用 `python scripts/evaluate.py --gold /workspace/patientphex-data/day1/dev16.gold.jsonl --pred PRED --output METRICS.json`。诊断脚本仅运行后分析，不能反向向训练资源添加 dev 表达。

## 已知局限与后续判定

模糊召回使用词头阻塞，完全不同的同义表达会漏检；首版没有神经语义编码器。训练集小，候选特征与同文档训练误差可能导致概率过于乐观。否定和重叠处理为简化规则；跨段患者共指仍依赖有限 mention 特征。需要用独立验证逐项检查，失败时区分候选召回不足、链接错误和归属误报。

主线程后续审计发现，关系训练 `gold_associations` 未拆分 `HP:...;HP:...` 合法复合关联串；train64 有 26 项，因此一部分原子概念正例被当作负例。该限制同时影响 v1、v2，应在下一轮修正并单独验证。本轮按统一安排停止训练，保持已交付 B 不变，不把未做的修复说成已完成。共享 evaluator 的同类复合串问题由主线程修复；原指标文件保留，修正版会另存并标注版本，不能拿两个口径直接比较。
