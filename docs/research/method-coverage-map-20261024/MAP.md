# 已实际尝试机制与可重开条件

2026-10-24 UTC快照；状态以当时主台账13:08及owner记录为准，执行前再读当前运行入口。此页是查重目录，不是新实验授权或完整审计清单。
**目标与比较：** 原train64/dev16公共v2目标≥.8；当时原dev最佳ee39/.755239909。下表F0/F1为内部开发折，各行基线/实体壳可能不同，不横向排名；负结果只关闭对应固定配方。

| 家族／检索词 | 已实际做过、结果与边界 | 重开前应有什么证据／入口 |
|---|---|---|
| 01 实体候选、规则/CRF/Sap、PhenoTagger/Bioformer/NCR、selector | 公共源＋严格OOF树/LGB及文档概念支持已有原dev增量；LGB支持.748583→.754217，再合患者增量至901/.754361。公开分类头Top3扩ID四折pooled .725655→.723467；它**不是Sap Top3**。 | 先证当折train-only新跨度/新ID可修复供给与所有下层祖先隔离，不能重复“多加公开模型必涨”。[S01][S02] |
| 02 上下文规范化、Sap Top8、cross-encoder | 固定跨度＋Sap新ID Top8＋±180上下文Biomed CE已train48/F0：.642034→.637749；严格nested CE入树.714808→.701372。不是只做过旧池排序。 | 新已选跨度池的可修复错误/召回、exact/NIL保护影响及严格OOF特征来源先证明；改K/loss不自动算缺口。[S02] |
| 03 GP／GLiNER2.5跨度与联合schema | GP直接并集.701293＜强.728698；GP置信列F1正/F0负，补齐四折pooled .731545→.731170关闭。GLiNER2.5 Task1真实10轮后并集.724624＜.728698；预训练联合关系并集.672407＜N .676546。 | 新独立供给或可解释特征控制才讨论；不能把GP/GLiNER当仅readiness，亦不据新增FP率否定整个头。[S03][S04][S25] |
| 04 生成式mention规范化、HPO/NIL/NO、困难负例 | **进行中**：Qwen3-8B片段→canonical/NIL/NO，2353行；独立2931行只增578个train-only零重叠负跨度。两线均固定完整1416 F1跨度，尚无本快照完整质量。 | 先收原两轮/两ACK/完整推断与双端点；不从partial判断、别重复开训练。[S05][S06] |
| 05 患者身份、共指、nearest、identity bridge | F-coref markers完整80：.660942→.660757；identity bridge真实锚点/距离文本F0 .652285→.641438。旧LR/病例检索早已学习距离、同句/同段、nearest等。 | 证明新增且可信身份信息；nearest只是几何，不能充当人工证据归属。新PCS显式几何另列，不能称项目从没用位置。[S07][S08] |
| 06 神经关系CE、max-MIL、pair-query、DocRE/ATLOP | Biomed局部pair CE、occurrence max-MIL、全文pair-query均实际做过；DocRE全提及/marker8→30轮最佳并集.669662＜N .676546，meta-OOF＋Doc .674069＜强.686339。 | 候选、目标或真正新增信号有依据才重开；AP上升不等于可部署Score，零交集context不冒称跨窗交互。[S09][S11] |
| 07 患者条件跨度PCS、latent span bag、显式geometry | 全文anchor语义池化＋NER/潜在max袋，8轮并集.678196、fresh24轮.669185均＜B .686339。**新50参数全锚点距离×同段残差正在跑**，零初始、旧LR3e-5/新表1e-2。 | 先收固定24轮geometry终点；不改cut/挑epoch，不将139/560更近其他患者的训练几何当错误标签。[S08][S10] |
| 08 连续似然、binary、query-token／fixed-slot | 530子集NLL已做；更关键：9B fresh全无答案query共享prefill、末hidden0/1两列CE已124行2506pair/2轮→F1全80。固定.5/.630248、expected-F1/.648033均＜旧JSON/.660942；CE两列等价BCE差值。 | 全文/独立head/YesNo是额外变量，不代表核心机制未试；无新依据不重命名重训。选位全vocab投影与直接小head的工程差别仍保留。[S11] |
| 09 LLM基座谱系：Qwen3.5、原生Qwen3、R1、Med1.5 | 9B多配方是主力；原生Qwen3-8B strict32强B并集F1+.002973、F0−.023321，独立迁移关闭、仅因固定融合双折正做full64组件。R1准确为DeepSeek-R1-0528-Qwen3-8B：SFT后.606546/并集.657384＜N。Med**1.5-4b-it**短/全文fresh32均完整负。 | 实际revision/训练任务先查，不把原生Qwen与R1混同、不称首次全注意力；医学名或小参数不保证收益/低显存。[S12][S13][S14] |
| 10 输入全文、候选顺序、RAG/ICL | 旧短训＋全文推断N .676546高短输入.660942；train+infer全文SFT .675134未胜N。全文逆序alone .664751/并集.667933负；train32 TF-IDF top1完整训练示例ICL .645783＜.660942。 | 旧RAG query为局部证据非整篇，不能证明所有RAG无效；但不据该差异立即重复。输入完整性与模型资源需成对披露。[S14][S15] |
| 11 生成目标：开放HP-ID＋quote、grounded quote、canonical-name/trie | 开放HP-ID/quote与逐窗quote任务已有负结果；新grounded患者全文quote＋固定规范化强B并集F1 .695950、F0 .713748为正。8B患者canonical-name集合并集.672946＜B，52新增全FP；合法cap前缀∩grounded零增量。 | exactquote和标准名的覆盖边界不同，不伪造证据；完整grammar禁重复已实现，前缀策略亦已做。Grounded full64仍未完成。[S16][S17][S18][S23][S26][S27] |
| 12 目标转置、patient-first、响应PT编码、集合解码 | patient-first与逆序曾产生有效组合；保原clinical IDs的可逆PT输出编码F1 .686619＞N .676546、仅略高B .686339，**同权重F0全58仍在跑**。原80合法重复PT稳定去重只增0TP/5FP，.685663负。 | 等原strict F0，不把别名映射称无语义置换；独立去重控制不热改正在跑的decoder。[S01][S19][S20] |
| 13 损失与训练时长：value×2、患者均衡、warm-start/fresh4、indexed CE | value×2、normal再训2轮、患者α=.5均已完成负；fresh patient4轮组合.662072＜匹配P2组合.671703及N；DocRE30/PCS24亦非未试。indexed CE为同目标工程，不代表新监督。 | 曲线下降不证明held欠拟合、2轮也非已证最优；新时程需具体训练证据/固定最终端点，不能忽略更长训练旧反证。[S07][S14][S21][S09] |
| 14 学习校准/融合：HGB303、LR5、meta-OOF、binary校准 | 303列HGB/五维LR等已有控制；DocRE meta负。binary logit＋SFT hard的F1拟合→F0仅+.001102仍弱于同实体规则。新三raw-bit LR F0 .722874虽高B .711508，仍低固定2of3 .724457，按双门关闭。 | 学习元数据须严格OOF/排当前held；“已用F1元训练”不称盲测。弱对照正不能替代强集合门。[S07][S09][S11][S22] |
| 15 固定集合共识、双seed、2of3、四来源、原dev迁移 | 固定旧多数两折负；双seed交集原dev901/.754361→ee39/.755240正。grounded/sample/Qwen三源2of3 F1 .701087、F0 .724457双正，全并集负；**root四源F1 .701325小正、F0待response**。 | 组合按完整固定公式，不能按患者/TP择源；Qwen组件已完，groundedfull64未完，root原dev两端点尚未齐冻，不能冒称新原dev最好。[S14][S23][S24] |

**待完成清单（不是再次启动指令）：** 2353规范化→全1416；2931负例版→全1416；PCS geometry→完整F1双端点；response同权重→全58 F0；groundedfull64→双ACK及原dev36；root四源→固定F0，三源full64→预锁原dev融合。恢复只查各owner原job/intent，不从本页旧状态重启。
**使用规则：** 新提案先按候选供给、上下文、监督、保护/解码、祖先和实际终点逐项对照H59；单一负配方不否定整族。此页没有提出新GPU实验，未读新gold/NN/fit/评分。精确文件SHA和只读快照在[refs.json](refs.json)，活动状态以当前主台账更新为准。

[S01]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/6418be86d03ac22a897c73ebcbd59114269d4af0/docs/campaigns/round2-20261024-stage6-decisions.md
[S02]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/6418be86d03ac22a897c73ebcbd59114269d4af0/docs/research/context-normalization-gap-20261024/REPORT.md
[S03]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/89f31cca08ab83ea9dbc37c47a16af85f04b5787/solutions/span_normalization/WORKLOG.md
[S04]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/f1ff2c81f2c1ca8e90e6e73ac22cd9489e22f853/solutions/gliner25_task1/WORKLOG.md
[S05]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/d5691a67c22b11199849c2e7c10e2f3cbc506e92/solutions/mention_canonical_normalization/WORKLOG.md
[S06]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/fa76133b389f679f6abfea05e0a52f58378ac066/solutions/mention_hard_negatives/WORKLOG.md
[S07]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/6418be86d03ac22a897c73ebcbd59114269d4af0/docs/research/round2-next-method-gap-20261024/REPORT.md
[S08]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/98fd3528edff43df4265f42ce6d29c917c53d560/solutions/qwen3_relations/evidence/patient-span-geometry-readiness.json
[S09]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/17734aa4f01074e84e8aef8c06e61dab0cbca22e/solutions/document_relations/WORKLOG.md
[S10]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/36aa309d15329e196b825fc0f980b0fc542cba45/solutions/patient_span_geometry/WORKLOG.md
[S11]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/98fd3528edff43df4265f42ce6d29c917c53d560/solutions/qwen3_relations/evidence/query-token-classifier-duplicate.json
[S12]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/98fd3528edff43df4265f42ce6d29c917c53d560/solutions/qwen3_relations/evidence/medgemma15-duplicate.json
[S13]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/98fd3528edff43df4265f42ce6d29c917c53d560/solutions/qwen3_relations/experiments/E-20261024-QWEN3-002-F0.md
[S14]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/6418be86d03ac22a897c73ebcbd59114269d4af0/docs/campaigns/round2-20261024-stage7-local-optimization.md
[S15]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/98fd3528edff43df4265f42ce6d29c917c53d560/solutions/qwen3_relations/evidence/retrieval-demonstration-duplicate.json
[S16]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/b85c0135522e742eac9bd0f8463a160d1c52c675/solutions/grounded_generation/WORKLOG.md
[S17]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/a297f89ea4d3084a57897991ff778146f404e93e/solutions/canonical_names/WORKLOG.md
[S18]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/4f07bb5cf32100d9dbde2e46d60c4a60c5a467db/solutions/canonical_prefix_agreement/WORKLOG.md
[S19]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/1cf04539f8fa0fa23ca60fe802d4420e5e585ee3/solutions/patient_response_code/WORKLOG.md
[S20]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/edfe9d1b2de0b1227565971989f4936e9f8e0376/solutions/response_set_decoder/WORKLOG.md
[S21]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/a21698aa05ce4e09199f0afdce4af57be001c8f9/solutions/patient_schedule/WORKLOG.md
[S22]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/a5afe80d71185b83a568a61749516eb4b41a9db1/solutions/relation_calibration/WORKLOG.md
[S23]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/6418be86d03ac22a897c73ebcbd59114269d4af0/docs/campaigns/round2-20261024-stage8-parallel-expansion.md
[S24]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/ae1ea26644be3874ba279afaad1c58c351cd3623/solutions/four_source_fusion/WORKLOG.md
[S25]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/de08d1fc8987586ea4e6a0fd946dd566d6f46d35/solutions/schema_extraction/WORKLOG.md
[S26]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/3940d0068065423e9ac634da67f532308ce314f8/solutions/grounded_task1/WORKLOG.md
[S27]: https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/9be16bec96820aedac3d18c33524f037a703c0d0/solutions/case_retrieval/evidence/round2/patient-set-paired-control-summary.json
