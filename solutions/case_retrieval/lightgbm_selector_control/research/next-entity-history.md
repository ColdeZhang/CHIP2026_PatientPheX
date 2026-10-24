# 下一实体增量：有限历史盘点（仅只读，2026-10-24）

建议优先考虑**固定候选池的相对竞争特征**，不是再扩大 top-k、直接拼 GLiNER 或降 cut。当前 LGB dev 的 859TP/202FP/247FN 复用 owner 已冻结诊断：126 FN 的正确候选低于 .3，4 个高于门限未选；50 个仅有错概念的精确 span，64 个仅有重叠 span，3 个无重叠。132 FP 仅说明与真实体重叠但边界不同，**不能推定同概念或都能靠边界修正解决**。本盘点未读取原始 gold、拟合、预测、评分或启动模型。

下文 O=/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules，S=/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf；R=/workspace/patientphex-runs/round2-20261023。

| 已实跑变量 | 实际结论与准确记录（旧弱池结果不冒充当前强池控制） |
|---|---|
| SapBERT Top8 + 上下文 BiomedBERT CE 改 ID | 旧 CRF+Sap 固定 span：改 13 ID，Score .642034→.637749；已识别 exact span 原 ID 768/778 正确，Top8 覆盖777仅为上限。S/experiments/E-20261023-009.md，S/run_linker_rerank.py。 |
| public top3 ID、Sap 候选专属 cosine | 原 Bioformer 四折 top3 .725655→.723467；top3+Sap 与原池+Sap 均未超过原 entity mean .780193，均关闭。O/experiments/E-20261023-PUBLIC-PHENO-TOP3-ID.md、E-20261023-PUBLIC-SAP-FEATURE.md、E-20261023-PUBLIC-SAP-OLD-POOL.md。PhenoBERT 三类 pair 特征也试过 top3/原池，后者 .777836<.780193；S/experiments/phenobert-old-pool-control.md。 |
| GLiNER 供给/扩大模型 | medium 在旧 CRF 上曾正；large/BioMed/context/同义替换/focal 不是空白方向。更关键：**强三公共源 HGB+nested 后追加已有 GLiNER**，F0/F1 新增23条仅4TP/19FP，pooled .733753→.732487，已关闭；S/experiments/gliner-strong-pool-supply.md。这不等于所有 GLiNER 学习融合都测过，但重做同一直接拼接没有新变量。 |
| 规则与公共候选放宽 | exact/related aliases、缩写/形态、精确本体 gate 均已有实现/控制；精确 gate 净 +40TP/+58FP，pooled −.0008865。O/enhance.py、O/evidence/round2/cpu-stage1.json、S/experiments/ontology-miss-mechanism.md。native .95∪.5 四折 .722651→.718047；unigram 直接补391条=17TP/374FP。O/experiments/E-20261023-PUBLIC-PHENO-UNION-OOF.md、E-20261023-PUBLIC-PHENO-UNIGRAM.md。 |
| 绕过 hybrid / 数学解码 | raw top1>.95 补候选：原诊断27个潜在可补原子，实际 held 新1806键 **0接纳**，旧键重选净−7TP/+9FP；O/experiments/E-20261023-PUBLIC-PHENO-PREHYBRID.md。这里不是已训练的边界修正器；有限文件搜索未找到 PHENO-SPAN-ADJUSTMENT 命名实验。固定强池全局兼容集合已负，R/validation/global-compatible-decoder-review/review.json。 |

**唯一候选假设：在同强8池、原103列上增加18列输入侧相对证据。** 对每个候选，只比较两类其他候选：①同 exact span、不同 literal ID；②不同但重叠 span、相同 literal ID。分别对 public/Bioformer/BioBERT 三个已缓存来源，加入“该来源竞争项存在、竞争项最高原始 score、自身原始 score 减竞争项最高 score”三值；无竞争项置零并由存在位区分。完整原候选、原字段、旧 nested 兼容规则、.3 和固定 LGB 参数保持。使用候选来源 score，**不用在自身训练文档上拟合出的 selector 概率**；每篇仍来自原 OOF 生成链。只是后续可注册的 CPU 特征控制，当前未构造/拟合。

依据：当前130个可供给 FN 与大量重叠 FP 值得检查筛选；实际 O/calibrate.py::features 的 overlap/contained/covers/same-id 只相对 base，O/public_phenotagger_biobert_selector/fit_control.py::feature 仅追加逐候选18个公共来源字段，**未计算全候选间来源分差**。有限三路线源码/实验及 R/research、R/ontology_rules 记录检索未发现该相对特征或 LGBMRanker 实跑。曾提过“概念竞争”不算运行；旧 Top8 CE 与此次原始公共来源相对数值不同。不要把排名目标直接换入 .3：其输出不再具有当前二分类概率语义。

限制：候选覆盖989/1106仅是乐观覆盖；上述特征不能创造117个缺失的正确候选，也未证明能改善130个可供给 FN。旧边界普查另有 >8词53原子/27 span 未供给，但不是当前247 FN分解；因此不据此优先重跑9–12gram或宣称潜在提升。先用既有纯输入池检查新增相对特征是否非退化及成本，再由 root 决定一个固定四折控制；不按原 dev 单例做规则补丁。

当前证据：R/research/lightgbm-selector-readiness/fixed-entity-error-inventory.json（owner 0.122秒固定诊断，source b79da46a…）；旧长跨度证据 O/experiments/P-20261023-PUBLIC-PHENO-BOUNDARY.md。所有数值直接引用已冻摘要，未新读答案或重算质量。
