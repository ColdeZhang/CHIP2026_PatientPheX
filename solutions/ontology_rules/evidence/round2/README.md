# 第二轮证据索引

- [CPU第一阶段](cpu-stage1.json)：14组内验证关键结果，包含无收益与失败配置。
- [语义阈值内选](semantic-inner-sweep.json)、[冻结四折及dev](semantic-frozen-results.json)：生成与装配来源分别记录；独立逐字节复现通过。
- [文内缩写](abbreviation-results.json)、[关系候选负结果](relation-pool-sweep.json)、[宽候选微收益](broad-candidate-result.json)。
- [OOF候选校准初版](oof-calibration-v1.json)、[固定160对照与初版复现](oof-calibration-fixed160.json)：目前最好dev .646244，二阶段训练层级清晰区分。
- [嵌套训练内校准](calibration-nested-fold0.json)、[关系覆盖诊断](fold0-relation-diagnostic.json)：独立内验证与oracle诊断明确分开。
- [Qwen全文关系负结果](llm-fold0-negative.json)：原始、单轮格式修复、规则交集均保留，未达标。
- [CE1epoch、训练先验修正与固定融合](context-ce-epoch1.json)：训练48嵌套候选，固定内fold0对照，无有意义增益，不跑dev。
- [CE3epoch负结果](context-ce-epoch3.json)、[CRF多来源候选树](multi-candidate-fold0.json)：完整对照与来源隔离，未达标。
- [A800运行时实测](a800-runtime-verification.json)：CUDA矩阵、Qwen全模型加载与最小生成通过。
- [远监督双头检测准备](distant-detector-preparation.json)：CPU构造和隔离审核，训练尚未运行，不能当性能证据。
- [CRF token/link置信度](crf-confidence-fold0.json)：来源与匹配控制字节复现，fold0 .714808，尚待另一外折复验。
- [首个GPU作业](gpu-job.json)：启动命令、固定GPU与managed进程；后续作业详见实验与外部manifest。

本目录只存小型证据。模型、全量预测、候选与完整日志保存在方案台账所列外部目录；当前未达0.8，不上传官方。

- [outer1固定三控制](crf-confidence-outer1.json)：跨折方向一致，原dev待冻结选型。
- [远监督双头五控制](distant-detector-results.json)：同宽池仅微小增益，保留负结果，不升级。

- [原dev前冻结的四折选型](multi-fourfold-strategy-freeze.json)：预设最简选择规则，source入选，含全部12预测SHA。

- [唯一预选原dev](multi-selected-dev.json)：.639287，低于v1最好分，不升级默认。

- [OOF聚合归属准备](aggregate-relation-preparation.json)与[fold0固定控制](aggregate-relation-fold0.json)：高召回池诊断、单模型/并集负结果及新增错误类型。

- [聚合归属第二折固定复验](aggregate-relation-fold1.json)：替换及union仍负，不推广原dev。

- [真实PhenoTagger结构与设计](phenotagger-design.json)：与旧二元冻结encoder区分，当前仅成本测量准备。

- [PhenoTagger训练材料](phenotagger-preparation.json)与[100步成本测量](phenotagger-measure100.json)：真实encoder梯度/更新、耗时/显存，尚无效果分数。

- [PhenoTagger batch256×30吞吐测量](phenotagger-measure256x30.json)：分组正类/None训练loss，非验证分数。

- [PhenoTagger固定30ep计划](phenotagger-formal-plan.json)：预先固定训练预算与恢复策略，不按fold0选epoch。

- [PhenoTagger真实中断恢复核验](phenotagger-resume-check.json)：direct4 vs split2+2逐张量/全RNG一致。

- [PhenoTagger正式作业](phenotagger-formal-job.json)：15:00:33Z启动；持续状态看方案WORKLOG，不将启动回执当完成。

- [PhenoTagger终点推理冻结计划](phenotagger-inference-plan.json)：固定p.95与既有宽池，仅增量保留基线。

- [PhenoTagger终点训练自检预设](phenotagger-training-probe-plan.json)：固定512+512训练样本，非验证指标。

- [PhenoTagger固定30ep终点结果](phenotagger-endpoint-results.json)：fold0仅+.000094，未升级默认；训练曲线和固定训练自检独立列出。
- [PhenoTagger独立审查索引](phenotagger-audit-index.json)：实际数据、真实resume及冻结推理接口，完整报告仓库外。

- [100ep终点结果与诊断](phenotagger-endpoint100-results.json)：独立复评通过，误报仍多；[Sap一致性控制预设](phenotagger-sap-consensus-plan.json)仅处理191新增、保持base。

- [Sap固定一致性结果](phenotagger-sap-consensus-results.json)：训练内fold0 .725539，仅一个固定点，原dev不变。

- [真实肯定词面3ep适配冻结](phenotagger-adaptation-plan.json)：仅train48、原回放保留、实际CPU token/init检查，尚不代表训练结果。

- [真实词面3ep raw负结果](phenotagger-adaptation-results.json)及[唯一配对Sap过滤计划](phenotagger-adaptation-sap-plan.json)：固定预算与负结果保留。

- [真实词面适配同Sap负结果](phenotagger-adaptation-sap-results.json)：.721696低于父过滤.725539，关闭当前配比，无原dev/官方上传。

- [原dev固定两臂预设](phenotagger-original-dev-plan.json)：Pheno48+HGB/Sap64混合规模，raw/filter预测均先冻结再评，非新增训练。
- [原dev两臂结案](phenotagger-original-dev-results.json)：raw .636028负、filtered .646360仅微增，独立CPU复现通过；默认稳定base .646244保留，不追加Pheno-only预算或调阈值。
- [强base共享组合计划](phenotagger-crf-gliner-dev-plan.json)与[两臂结果](phenotagger-crf-gliner-dev-results.json)：CRF+GLiNER上实体-only .712586、fullLR .712071；两pred冻结后统一评，不冒称本体独立分。
- [GLiNER弱链接有限选择CPU提案](gliner-rejected-link-proposal.json)：运行前冻结的176输入跨度、原Sap缓存top5、9B真实token预算与ABSTAIN/NIL边界；实际负结果见下方终点。
- [有限链接冻结运行计划](gliner-rejected-link-run-plan.json)与[解析/无操作检查](gliner-rejected-link-parser-checks.json)：900秒cap、严格候选码、完整base保留，合成检查不代表模型效果。
- [有限链接真实终点](gliner-rejected-link-results.json)：完整176响应且格式正确，Score下降.010365；负结果保留，不继续修提示或原dev迁移。
- [OOF doc-HPO袋训练诊断](relation-bag-training-diagnostics.json)、[固定MIL预算](relation-mil-run-plan.json)、[真实CPU检查](relation-mil-cpucheck.json)和[独审入口](relation-mil-audit-index.json)：纯关系背景过滤，准备与真实终点独审均通过；质量为下方负结果。
- [固定3ep MIL真实终点](relation-mil-results.json)：同实体纯关系删除控制.711354→.702753，删除27TP/45FP；不调阈值/epoch，原dev未运行。

- [严格OOF CE特征入树可行性](ce-oof-feasibility.json)：旧outer模型/分数资产、二级候选来源泄漏核对、两种隔离方案成本；仅CPU提案，未新增训练。

- [CE唯一OOF运行计划](ce-oof-run-plan.json)与[CPU实际准备](ce-oof-preparation.json)：严格3×16/6池隔离、旧84列/.4 byteexact、总720秒预算；已完成固定终点，结果见下方。

- [CE实际作业](ce-oof-training-job.json)与[独审索引](ce-oof-audit-index.json)：18:38:18Z开始，720秒硬cap；作业状态看WORKLOG，不把启动等同终点。

- [CE严格OOF真实终点](ce-oof-results.json)：全管线289.183秒，原84列/.4→85列/.4使.714808→.701372，负结果不回扫。
- [训练侧链接availability聚合](nested-link-diagnostic.json)：复用原train48错误分类，只核同跨度全池/精确字典；无新top-k或预测，不替代模型分数。

- [受控NER增强只读研究](ner-augmentation-research.json)：两份论文、固定MIT源码、训练32聚合预筛与复算；无样本生成或新模型分数，[可执行备选提案](../../experiments/P-20261023-NER-AUGMENTATION.md)。

- [NER增强CPU准备](ner-augmentation-preparation.json)与[唯一固定计划](ner-augmentation-plan.json)：1150原窗+353新增、真实collator全量核验，fresh medium10/3760步；独审PASS，未将准备当效果分数。

- [NER增强真实启动](ner-augmentation-job.json)：已推送源d9ab108，fresh medium1503×10，训练deadline19:37:55Z；包含旧32兼容推理冻结，不是新效果结果。

- [NER增强固定后链](ner-augmentation-postlink-plan.json)：旧32/new32均走同A30 Sap环境、原阈值/装配；谱系轻审ce6847d4通过，1378旧跨度键一致，浮点分数最大差1.72e-5；完整分数待终点。
