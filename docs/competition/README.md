# CHIP2026 PatientPheX 比赛资料

查询日期：**2026-10-23（UTC）**。比赛 ID：`532512`。

本文整理官方公开资料，供后续离线查阅。比赛正式名称为 **CHIP2026-面向生物医学全文的特定患者表型概念识别任务**。赛程、榜单和网页规则均为查询当日快照。

## 资料入口

| 内容 | 本地文件 |
|---|---|
| 任务、数据、评分、提交要求和赛程速查 | 本文 |
| 仓库数据及本地评测与官方的差异 | [旧仓库核对记录](../../archive/legacy-20261023/docs/competition/LOCAL_NOTES.md) |
| 天池比赛介绍原文：规则、提交、赛程、联系方式 | [tianchi-introduction.md](sources/2026-10-23/tianchi-introduction.md) |
| 天池赛题与数据原文：完整字段、指标公式 | [tianchi-information.md](sources/2026-10-23/tianchi-information.md) |
| 天池参赛协议原文 | [tianchi-agreement.md](sources/2026-10-23/tianchi-agreement.md) |
| 官方 GitHub 中文说明及英文说明 | [README.md](sources/2026-10-23/official-github/README.md)、[README_EN.md](sources/2026-10-23/official-github/README_EN.md) |
| A/B/赛后完整榜单 | [榜单索引](#榜单快照) |
| 来源 URL、抓取时间、文件校验值 | [manifest.json](sources/2026-10-23/manifest.json) |

官方在线入口：[比赛介绍](https://tianchi.aliyun.com/competition/entrance/532512/introduction)、[赛题与数据](https://tianchi.aliyun.com/competition/entrance/532512/information)、[排行榜](https://tianchi.aliyun.com/competition/entrance/532512/rankingList)、[官方 GitHub](https://github.com/DUTIR-BioNLP/PatientPheX)。

## 任务与硬性约束

输入为 PMC 生物医学文献全文及指定患者信息，已核对的比赛数据为英文文本。

1. **Task 1：全文表型概念识别。** 找出表型实体在全文中的字符位置，并标准化为 HPO 概念。
2. **Task 2：特定患者表型概念识别。** 为每位给定患者输出其表型概念集合。患者指称仅提供部分有效 mention，需要处理其他指称及跨句、跨段关联。

官方要求：

- 模型参数 **不超过 10B**。允许传统机器学习、深度学习、预训练语言模型、大语言模型、提示学习、监督微调、RAG 等方案。
- HPO 固定为 **v2026-06-23**，使用 **Phenotypic abnormality（HP:0000118）分支**。
- 允许使用公开可获取的外部代码、工具及知识库，须注明来源，在技术报告中说明模型和外部知识资源。
- **不允许使用外部人工标注表型数据集**。
- A 榜答案不公开，也不用于 B 榜模型训练。
- 比赛数据仅用于本次评测；其他用途需联系组织者。

已归档的旧 `archive/legacy-20261023/data/hp.obo` 为 **2026-09-01** 版，与指定版本不同；旧评测脚本也不等价于官方口径，新项目不会默认沿用，详见 [旧仓库核对记录](../../archive/legacy-20261023/docs/competition/LOCAL_NOTES.md)。

## 数据规模

下表为官方发布的完整标注统计。A/B 的实体和关联数量是官方金标统计，下载的测试文件不含这些答案。

| 统计项 | Train | A / Dev | B / Test | 总计 |
|---|---:|---:|---:|---:|
| 文献数 | 80 | 20 | 100 | 200 |
| 患者数 | 209 | 53 | 244 | 506 |
| 表型实体数 | 6,027 | 1,563 | 7,603 | 15,193 |
| 唯一 HPO ID 数 | 1,662 | 551 | 1,863 | 2,874 |
| 否定表型数 | 171 | 38 | 226 | 435 |
| 复合表型数 | 69 | 5 | 138 | 212 |
| 患者表型关联数 | 2,307 | 563 | 2,786 | 5,656 |
| 患者相关唯一 HPO ID 数 | 1,091 | 351 | 1,252 | 1,995 |

前任保存的 [训练集](../../archive/legacy-20261023/data/train/PatientPheX-train.jsonl) 和 [B 集](../../archive/legacy-20261023/data/B/PatientPheX-V1-B.jsonl) 已移入历史归档，未找到 A 集；新项目的数据入口尚未建立，后续独立核验来源与版本。当前未登录的公开数据接口仅显示两个脱敏压缩包名称，未返回下载地址；本次保存了[返回内容](sources/2026-10-23/tianchi-datasets.json)，未补下载比赛数据。

## 数据字段与坐标

JSONL 为 UTF-8 编码，每行一篇文献。

| 字段 | 含义及使用要点 |
|---|---|
| `pmc_id` / `pmid` | 文献标识；保留输入原值 |
| `patient[].patient_id` | 该文献内的指定患者 ID；有文中编号时为 `O+编号`，其他通常为 `P` 开头 |
| `patient[].mention[]` | 患者部分有效指称，含 `text`、`offset`、`length` |
| `full_text[]` | 分段全文，含 `section_type`、BioC `type`、`offset`、`text` |
| `entities[].identifier` | HPO ID；复合表型用分号分隔；无有效 ID 为 `-1` |
| `entities[].type` | `Phenotype` |
| `entities[].offset` / `length` | 原始完整全文中的全局字符位置及字符长度 |
| `entities[].text` | 实体原文 |
| `entities[].note` | 通常为 `null`；`NO` 表示否定表型 |
| `association[].patient_id` | 输入中给定的患者 ID |
| `association[].phenotype` | 患者表型字符串列表：有效 ID 保存 HPO ID；ID 为 `-1` 的表型保存原文文本 |

工程使用提示：段内匹配位置需加上 `full_text[].offset` 才是全局位置；不要用直接拼接所有段落后得到的下标替代。患者应以 `(pmc_id, patient_id)` 区分。上述提示来自官方字段定义及本地结构核对。

## 官方评测口径

仅评价**正文**中的预测，参考文献及其他非正文内容不参与评价。公开说明未枚举完整的正文段落筛选规则。

| 指标 | 榜单字段 | 官方定义 |
|---|---|---|
| menF1 | `task1_mention_f1` | 实体边界与 HPO ID 均正确才算 TP；全局汇总 TP/FP/FN 后计算 F1 |
| docF1 | `task1_document_f1` | 每篇文档分别对 HPO ID 去重，比较预测与金标概念集合；汇总全部文档的 TP/FP/FN 后计算 F1；不比较位置、边界和出现次数 |
| micF1 | `task2_micro_f1` | 全局汇总患者–表型关联的 TP/FP/FN 后计算 F1 |
| macF1 | `task2_macro_f1` | 对全部金标患者分别计算 F1，再按患者等权平均 |
| Score | `score` | `0.25 × (menF1 + docF1 + micF1 + macF1)` |

对一般非空情形，`P = TP/(TP+FP)`，`R = TP/(TP+FN)`，`F1 = 2PR/(P+R)`。docF1 的统计单位是“文档–概念”，同一概念在不同文档中分别计数；它不是逐文档 F1 的平均值。

特殊情况：

- **否定表型 `note="NO"`**：金标不作为正例、不计 FN；正确预测为 `NO` 不参与评价；漏掉该否定实体不扣分；误报为普通表型计 FP。
- **复合表型**：将 `HP:...;HP:...` 按分号拆成独立评价单元，分别匹配。
- **无 ID 表型 `-1`**：官方说明为实体位置正确即正确；其 docF1 集合表示未进一步说明，不自行推定。
- **患者金标和预测集合均为空**：该患者 Precision、Recall、F1 都为 **1**。

完整公式和原文见 [官方赛题快照](sources/2026-10-23/tianchi-information.md)。截至本次查询，官方 GitHub 仅提供说明和 LICENSE，没有公开评测脚本；归档中的旧 `src/evaluate.py` 不能被当作官方实现。

## 提交要求

- 提交对应测试集的 **UTF-8 JSONL** 预测文件，每行一篇文献，格式与训练结果字段一致。
- A 榜每日最多 **5 次**，B 榜每日最多 **3 次**；评测失败不扣次数。
- 排行榜整点刷新，按得分降序排序。
- A 榜用于开发调优，不计最终成绩；**正式 B 榜历史最优成绩**用于最终排名和奖项评定。
- 赛后评测的每日次数、文件大小限制和额外文件命名要求，本次公开资料未明确，不套用正式 A/B 阶段的限制。

以下为合法的单行 JSON 格式示意（文献 ID 为占位值，不能直接提交；实体例子取自官方格式说明）：

```json
{"pmc_id":"<输入PMC编号>","pmid":"<输入PMID>","entities":[{"identifier":"HP:0000519","type":"Phenotype","offset":206,"length":30,"text":"bilateral congenital cataracts","note":null}],"association":[{"patient_id":"P1","phenotype":["HP:0000519"]}]}
```

准备输出时，应覆盖测试集每篇文献并保留文献 ID，使用输入给定的患者 ID，校验全局位置和原文一致，患者表型列表按集合去重。这些是格式自检建议；平台对缺失文档、重复条目的具体处理未在公开说明中展开。

## 赛程与当前阶段

日期按 2026 年赛程整理。赛季接口时间使用 `CST`，本表按中国标准时间（UTC+8）记录；原始时间字符串见 [tianchi-detail.json](sources/2026-10-23/tianchi-detail.json)。

| 阶段 / 事项 | 时间 | 平台标识 |
|---|---|---|
| 正式报名 | 2026-08-20 至 2026-09-27 | 赛题正文日程 |
| A 榜评测 | 2026-08-20 00:00:00 至 2026-09-27 23:59:59 | `season=1`，`seasonId=1860` |
| B 榜评测 | 2026-09-28 00:00:00 至 2026-10-01 23:59:59 | `season=2`，`seasonId=1861` |
| 赛后评测 | 2026-10-03 00:00:00 至 2029-10-03 23:59:59 | `season=3`，`seasonId=1862` |
| 评测论文截止 | 2026-10-15（天池标注“暂定”） | 赛题正文日程 |
| CHIP 会议 | 2026-10-30 至 2026-11-01 | 赛题正文日程 |

**2026-10-23 查询时处于赛后评测阶段。** 平台顶层报名/结束时间已经对应赛后阶段，不能据此认为正式 B 榜延长到了 2029 年。未登录响应中的 `signed=false`、`showSubmit=false` 仅说明本次匿名访问状态，不代表用户账号的报名或提交权限。

## 榜单快照

抓取窗口：**2026-10-23 09:25:29–09:25:36 UTC**（北京时间 17:25:29–17:25:36）。所有榜单均已翻页，保存排名、队伍名、队伍 ID、总分、四项 F1 和最佳成绩提交时间。下表条数是有成绩的榜单记录数，不是报名队伍总数。

| 榜单 | 已保存 / 接口总数 | CSV | 带来源和分页信息的 JSON |
|---|---:|---|---|
| A 榜 | 189 / 189 | [ranking_A.csv](rankings/2026-10-23/ranking_A.csv) | [ranking_A.json](rankings/2026-10-23/ranking_A.json) |
| B 榜 | 103 / 103 | [ranking_B.csv](rankings/2026-10-23/ranking_B.csv) | [ranking_B.json](rankings/2026-10-23/ranking_B.json) |
| 赛后评测 | 55 / 55 | [ranking_post.csv](rankings/2026-10-23/ranking_post.csv) | [ranking_post.json](rankings/2026-10-23/ranking_post.json) |

正式 B 榜前三名及官方基线（此表保留 6 位小数；完整精度见文件）：

| 排名 | 队伍 | Score | menF1 | docF1 | micF1 | macF1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Agriv | 0.738920 | 0.771912 | 0.828626 | 0.705541 | 0.649602 |
| 2 | 医诺未来 | 0.724669 | 0.748349 | 0.823076 | 0.689004 | 0.638247 |
| 3 | 天色天歌天籁音 | 0.716945 | 0.741979 | 0.813420 | 0.683441 | 0.628939 |
| 44 | 官方Baseline | 0.655178 | 0.721091 | 0.793349 | 0.587425 | 0.518845 |

官方 Baseline 在 A 榜为第 28 名 / `0.725094`，赛后榜为第 15 名 / `0.655178`。榜单快照不替代最终获奖资格审核；赛后榜与正式 B 榜分别保存。

## 官方基线、组织与材料

官方 GitHub 补全的基线流程：用指定 HPO 版本训练的 **PhenoTagger** 完成 Task 1；用 **Qwen3-8B** 通过提示识别指定患者相关提及，并关联到已识别表型完成 Task 2。整体方法未使用训练集优化。天池说明还提到分段处理后使用完整章节补充跨句、跨段共指和遗漏表型。

参赛要求：每人一个账号，队伍 **1–5 人**；个人需依托单位报名；队长和全部成员需在报名截止前实名认证；组织方成员不得参赛。正式报名还要求在 A 榜结束前填写[报名表](https://docs.qq.com/form/page/DRE5IVEJJbnRvZkhK)。

奖励为获奖证书、会议汇报和英文评测论文（官方说明为 EI 收录）。获奖队伍需注册参会、口头汇报并撰写技术论文；会议注册、交通和食宿自理。未提交所需材料会失去评奖资格。

需提交材料：代码及说明、评测论文、额外公开资源的文件或来源地址。代码和文档打包为 zip/rar 等附件，发送至 `dutir_bionlp@163.com`；天池指定邮件标题为 `CHIP2026Task1-评测材料-参赛队伍名称`。

- 组织单位：大连理工大学；组织者：罗凌、王健、孙媛媛、林鸿飞。
- 联系人：朱旋律，`xlzhu@mail.dlut.edu.cn`。
- [CHIP 2026 官网](http://cips-chip.org.cn/2026/)。
- [官方钉钉群二维码本地副本](sources/2026-10-23/dingtalk-qr.png)。群内通知未纳入本次快照。
- [数据集论文](https://doi.org/10.64898/2026.10.07.757319)、[评测综述](https://doi.org/10.48550/arXiv.2610.29606)。两篇 BibTeX 已保存在[官方 GitHub 原文末尾](sources/2026-10-23/official-github/README.md)。论文全文未在本次下载。

## 来源差异与维护

- 天池 baseline 段落从“随后”开始，缺少 Task 1 前半段；本文采用官方 GitHub 的 PhenoTagger 说明补全，并保留两份原文。
- 天池论文截止标注“10 月 15 日（暂定）”，GitHub 未写“暂定”；本文保留天池标注。
- 天池材料邮件标题为 `CHIP2026Task1-...`，GitHub 为 `CHIP2026Task-...`；本文采用天池版本。
- 天池引文作者末尾写作 `Hongfei Linn`，GitHub 为 `Hongfei Lin`；引用可查阅已保存的 GitHub BibTeX。
- 字段表中的否定标记示例出现 `null/No`，评分正文明确使用 `NO`；实现应以评分正文为准。

以上说明是原文摘要；完整规则及参赛协议也已本地保存。来源快照保留原文，包括原始排版、相对链接和可能的笔误。GitHub 快照对应提交 `ae5b5e9b77233a7b05169448b2645f94bf02b3b3`；其代码许可证不替代赛事数据使用规则。

后续默认先查本目录。只有需要最新公告、当前提交额度/权限或更新后的榜单时再联网；若刷新，建议新建日期目录并保留本次快照。
