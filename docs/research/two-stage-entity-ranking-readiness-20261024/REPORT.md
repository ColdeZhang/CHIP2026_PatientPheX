# 两阶段实体排序：历史核查与 train48 准备度

2026-10-24；仅只读来源核查及一次获准的 train48 标签聚合。结论：**当前不立项 same-span 两阶段 ranker。a465 的严格嵌套训练材料可用，但实际能教会同跨度排序的正负竞争组只有 116 个。** 不改当前两个训练，不运行 fit、predict、解码或 evaluator，不打开 F1 gold / 原 dev gold。

## 方法库存与官方依据

缓存 `R/research/overview.txt` 388–392 行描述获选 Agriv 的 two-stage LightGBM ranking，另配 BioLinkBERT cross-encoder；公开摘要没有给出两阶段的精确目标、分组方式、特征合同。因此“先 span 检测，再 same-span HPO 排名”是可研究的解释，不能宣称已复现该团队具体方法。

有限检索三条活跃方案的 Python/Markdown，未找到 LGBMRanker / lambdarank / rank_xendcg 实际训练端点，也没有找到独立 span-existence 模型配同跨度 HPO ranker 的完整受控结果。当前 a465 是 **binary objective、101 列**；103 列是其他/full64 模型，不能套用到该折。已有 Sap/双头跨度门控、nested/global decoder、关系信号堆叠、CE rerank、普通 LGB 二分类及相对候选特征，各自改变了不同环节，不能冒称 grouped ranking 已测试。

但这并非全新研究提议：`official_best_B_901/followup_two_stage_research` 已做过旧四折汇总，root 已关闭“固定已选 span、仅换同 span 排名”。其旧聚合有 155 个正确/错误 ID 竞争组、仅 5 个现有 FN 能通过替换已选跨度的 ID 挽救。本次只读取既有汇总，没有重开四折 gold。18 列相对候选来源特征亦已负结案（pooled .731545429370 → .730783429343），不能换名重跑。

## H40 来源核查：a465 不是直接复用全局四折 OOF

`audit_lineage.py` 在 **0.442111 秒**内通过 54 项文件身份/来源核查，结果 `lineage-result.json`。训练池、标签及源文件完整 SHA 见该结果和 `lineage-sources.json`：

| 资源 | SHA256 / 关键事实 |
|---|---|
| a465 模型 | `a465e716fedab408a027f4c3af20f2e9870f88ac3a02209e7206d2ebf6b41227`；只哈希，不加载 pickle |
| 原 train48 标签 | `b1e0d953db49e7ecdb28de8588fe39dd996883610158c0a43cd93193d6c1be44` |
| 原训练池 `public-biobert-selector-control/prepared/fold1/train.pool.jsonl.gz` | `72aa15825a2b719ac06e4ecf1bb813a9defe4df0c1cfd1e28a2d0f446ba5f987`；45,403 候选 |
| 原标签函数 `fit_helpers.py` | `e8bac3d485074052175a5a365d750fd3d9fdb8d5b131260ba42bf10af6578aff` |

outer train48 为 F0+F2+F3。`nested-outer1/calibration-manifest.json` 的 inner0/2/3 分别使用其余两个内部折 train32 生成对应 16 篇候选；这三组 train32 均排除 outer F1 和自己的目标16。检查了实际文献 ID、训练文件 SHA，以及 baseline 词典、SapBERT 候选检测器和 CRF manifest 对同一 train32 / input16 / 输出的引用。原 CRF 路径包含自己的链接及关系模型；未偷用全局模型。特征源码与 a465 保存的依赖哈希一致。普通上下文/重叠/文内频数来自当前输入和上述 baseline；公共 PhenoTagger、Bioformer、BioBERT 是 manifest 明确未做本任务训练的冻结发布权重，区别于后来 H5 自适配模型。

这证明现存文件来源链足以支持在原 train48 上做此普查；并非重新执行或取证所有历史训练。固定公共模型的外部预训练来源仍是其发布证据，不是本次重新验证。没有发现 H40 所担心的 outer F1 泄漏，不保留无证据泄漏猜测，更不否定既有 train64→dev16 结果。

## 唯一新普查：0.414232 秒

`census_train48.py` 仅加载上述池和 train48 标签，用 AST 提取原文件的 `key` / `labels` 两函数原样执行，不导入训练模块。一次 `timeout 60s` 调用完整完成，正例 2,978 与原 manifest 一致。未读 a465 的 held-F1 scores。

| 项目 | 数量 |
|---|---:|
| 候选 / exact-span 组 | 45,403 / 44,865 |
| 单候选组 | 44,333（98.81%） |
| 多 literal-ID 组 | 532（1.19%） |
| 有效正负竞争组 / 正负配对 | 116 / 118 |
| 多候选全阴性组 / 多候选全正组 | 414 / 2 |
| 全部全阴性组 / 至少一个正候选组 | 41,889 / 2,976 |
| 含多个正候选的组 | 2 |
| 金标正跨度 | 3,424 |
| 有 exact candidate / 无 exact candidate 的金标跨度 | 3,183 / 241 |
| 有 legacy-supported candidate 的金标跨度 | 2,976（86.92%） |
| 已有 exact span 但候选均不支持金标的跨度 | 207 |

组键是 `(pmc_id,offset,length)`；同跨度 literal identifier 不拆分当作多个候选。原标签函数排除 `note=NO`，对复合 ID 按原子子集接受，对 gold `-1` 接受同跨度任何 ID。因此上表不是 evaluator 原子计数；81 个金标跨度带 NIL、39 个有多个原子，候选池55组含 compound literal。覆盖比例是训练供给描述，不是可达到的分数或关系覆盖。详细直方图及三 origin 聚合在 `census-result.json`，没有逐错例造规则。

## 判断与停止条件

分组 ranker 在单候选和全阴性组上基本没有内部排序监督；当前仅 116 组提供正负竞争。固定池排序无法补回 241 个缺跨度和 207 个缺支持 ID 的跨度。分离 span 检测有不同的可能作用：它改变“是否为表型”的监督/校准，可能影响大量单候选组，但该普查没有证明它优于已存在的二分类及跨度模型，也没有提供新的检测输入证据。仅因官方方法名字相同而 fit 两阶段，证据不足。

**此次到此结束，不启动模型。** 如未来有独立候选机制显著增加同跨度有效竞争，才重新评估排名目标；若未来单独研究检测/链接分离，必须声明新监督目标、NIL/compound 多正语义及独立概率校准，不能把 ranker 原始分数套旧 .3，也不能把已有跨度门控重命名成新实验。当前优先等待在跑的两项正式训练终点。

过程说明：来源核查脚本最初误把引用中的可选 `bytes` 当必填而提前中止；确认所有 SHA 一致后修正引用检查，未涉及候选标签计数。随后来源核查通过，**新的 train48 普查仅运行一次**。所有新增文件位于本外部研究目录，没有修改路线源文件、共享文档或模型。
