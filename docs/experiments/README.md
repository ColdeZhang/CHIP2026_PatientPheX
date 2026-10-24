# 实验索引

这里只保存会影响决策的短实验记录及必要的小型配置、指标和划分信息。执行约定见 [AGENTS.md](../../AGENTS.md)，当前进度见 [WORKLOG.md](../../WORKLOG.md)。

## 新项目实验

首日五路线从 `ac8a752` 创建独立分支/worktree，统一评测见[修正记录](day1-evaluator-review.md)，最新状态见[首日计划](../campaigns/day1-20261023.md)。下表同时列出固定 dev16 近似评测 v2 与已回收的官方 B 成绩，两者数据和评测器不同；详见[官方分析](../campaigns/day1-20261023-official-analysis.md)。

| 实验 | 方法 / 分支 | 当前结论 |
|---|---|---|
| [E-20261023-002](E-20261023-002-ontology-rules.md) | 本体规则 / `day1/ontology-rules` | .612597；官方 .603338（五路线最高） |
| [E-20261023-003](E-20261023-003-fuzzy-ranker.md) | 候选监督排序 / `day1/fuzzy-ranker` | v1 .630949；v2 .606179，保留负结果，选 v1，官方 .545653 |
| [E-20261023-004](E-20261023-004-sequence-crf.md) | 序列标注 / `day1/sequence-crf` | CRF .684734，官方 .590650；BiomedBERT .613313 未选用 |
| [E-20261023-005](E-20261023-005-case-retrieval.md) | 病例检索 / `day1/case-retrieval` | CPU .613518，官方 .599433；Qwen 下载受限，未运行 |
| [E-20261023-006](E-20261023-006-semantic-neural.md) | 语义编码 / `day1/semantic-neural` | SapBERT .545275，官方 .578946；MiniLM .517843 未选用 |

## 第二轮（方案目录维护完整过程）

| 实验 | 方案记录 | 状态 |
|---|---|---|
| E-20261023-007 | [本体规则最新实验](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/experiments/E-20261023-007.md) | 进行中，本地目标 ≥ 0.8 |
| E-20261023-008 | [病例检索最新实验](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/case-retrieval/solutions/case_retrieval/experiments/E-20261023-008.md) | 进行中，本地目标 ≥ 0.8 |
| E-20261023-009 | [CRF最新实验](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/experiments/E-20261023-009.md) | 进行中，本地目标 ≥ 0.8 |

总指挥只维护[协调记录](../campaigns/round2-20261023.md)和根台账；每方案的台账、实验、笔记、证据在其目录维护，避免重复。上表指向当前工作分支；达标实现合并前，main 中方案文档仅代表启动基点，不能当作最新过程。下一新实验 ID 为 `E-20261023-010`（跨日使用实际日期）。

## 第八阶段方案目录

当前执行状态、下一步与近期终点只维护在[运行入口](../campaigns/round2-current-work.json)和根台账；完整实验 ID、方法、正负结果与证据由下列各线台账索引。此目录不重复维护容易过期的“正在训练”等状态。旧原三线“进行中”指总体冲分目标，不能据此重开已关闭配方。

| 方案 | 分支中的完整记录 |
|---|---|
| `span_normalization` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/span-normalization/solutions/span_normalization/WORKLOG.md) · `explore/span-normalization` |
| `document_relations` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/document-relations/solutions/document_relations/WORKLOG.md) · `explore/document-relations` |
| `grounded_generation` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/grounded-generation/solutions/grounded_generation/WORKLOG.md) · `explore/grounded-generation` |
| `patient_schedule` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-schedule/solutions/patient_schedule/WORKLOG.md) · `explore/patient-schedule` |
| `schema_extraction` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/schema-extraction/solutions/schema_extraction/WORKLOG.md) · `explore/schema-extraction` |
| `qwen3_relations` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/qwen3-relations/solutions/qwen3_relations/WORKLOG.md) · `explore/qwen3-relations` |
| `relation_consensus` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/relation-consensus/solutions/relation_consensus/WORKLOG.md) · `explore/relation-consensus` |
| `sampled_consensus` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/sampled-consensus/solutions/sampled_consensus/WORKLOG.md) · `explore/sampled-consensus` |
| `grounded_task1` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/grounded-task1/solutions/grounded_task1/WORKLOG.md) · `explore/grounded-task1` |
| `parallel_relation_fusion` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/parallel-relation-fusion/solutions/parallel_relation_fusion/WORKLOG.md) · `explore/parallel-relation-fusion` |
| `gliner25_task1` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/gliner25-task1/solutions/gliner25_task1/WORKLOG.md) · `explore/gliner25-task1` |
| `relation_calibration` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/relation-calibration/solutions/relation_calibration/WORKLOG.md) · `explore/relation-calibration` |
| `patient_response_code` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-response-code/solutions/patient_response_code/WORKLOG.md) · `explore/patient-response-code` |
| `canonical_names` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/canonical-names/solutions/canonical_names/WORKLOG.md) · `explore/canonical-names` |
| `mention_canonical_normalization` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/mention-canonical-normalization/solutions/mention_canonical_normalization/WORKLOG.md) · `explore/mention-canonical-normalization` |
| `patient_conditioned_spans` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-conditioned-spans/solutions/patient_conditioned_spans/WORKLOG.md) · `explore/patient-conditioned-spans` |
| `mention_hard_negatives` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/mention-hard-negatives/solutions/mention_hard_negatives/WORKLOG.md) · `explore/mention-hard-negatives` |
| `open_concept_consensus` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/open-concept-consensus/solutions/open_concept_consensus/WORKLOG.md) · `explore/open-concept-consensus` |
| `canonical_prefix_agreement` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/canonical-prefix-agreement/solutions/canonical_prefix_agreement/WORKLOG.md) · `explore/canonical-prefix-agreement` |
| `patient_span_geometry` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-span-geometry/solutions/patient_span_geometry/WORKLOG.md) · `explore/patient-span-geometry` |
| `four_source_fusion` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/four-source-fusion/solutions/four_source_fusion/WORKLOG.md) · `explore/four-source-fusion` |
| `response_set_decoder` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/response-set-decoder/solutions/response_set_decoder/WORKLOG.md) · `explore/response-set-decoder` |
| `patient_competition_loss` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-competition-loss/solutions/patient_competition_loss/WORKLOG.md) · `explore/patient-competition-loss` |
| `span_document_support` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/span-document-support/solutions/span_document_support/WORKLOG.md) · `explore/span-document-support` |
| `response_code_equivariance` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/response-code-equivariance/solutions/response_code_equivariance/WORKLOG.md) · `explore/response-code-equivariance` |
| `patient_competition_calibration` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-competition-calibration/solutions/patient_competition_calibration/WORKLOG.md) · `explore/patient-competition-calibration` |
| `unanimous_relation_veto` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/unanimous-relation-veto/solutions/unanimous_relation_veto/WORKLOG.md) · `explore/unanimous-relation-veto` |
| `mention_candidate_selection` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/mention-candidate-selection/solutions/mention_candidate_selection/WORKLOG.md) · `explore/mention-candidate-selection` |
| `document_supported_relation_filter` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/document-supported-relation-filter/solutions/document_supported_relation_filter/WORKLOG.md) · `explore/document-supported-relation-filter` |
| `grounded_sampled_consensus` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/grounded-sampled-consensus/solutions/grounded_sampled_consensus/WORKLOG.md) · `explore/grounded-sampled-consensus` |
| `grounded_weight_average` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/grounded-weight-average/solutions/grounded_weight_average/WORKLOG.md) · `explore/grounded-weight-average` |
| `mention_no_veto` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/mention-no-veto/solutions/mention_no_veto/WORKLOG.md) · `explore/mention-no-veto` |
| `patient_swap_dpo` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/patient-swap-dpo/solutions/patient_swap_dpo/WORKLOG.md) · `explore/patient-swap-dpo` |
| `copy_grounded_decode` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/copy-grounded-decode/solutions/copy_grounded_decode/WORKLOG.md) · `explore/copy-grounded-decode` |
| `quote_set_completion` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/quote-set-completion/solutions/quote_set_completion/WORKLOG.md) · `explore/quote-set-completion` |
| `mention_boundary_selection` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/mention-boundary-selection/solutions/mention_boundary_selection/WORKLOG.md) · `explore/mention-boundary-selection` |
| `conditional_stacking` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/research/conditional-stacking/solutions/conditional_stacking/WORKLOG.md) · `research/conditional-stacking` |
| `verified_rationale` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/research/verified-rationale/solutions/verified_rationale/WORKLOG.md) · `research/verified-rationale` |
| `supervised_sapbert_linker` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/research/supervised-sapbert-linker/solutions/supervised_sapbert_linker/WORKLOG.md) · `research/supervised-sapbert-linker` |
| `training_seed_ensemble` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/research/training-seed-ensemble/solutions/training_seed_ensemble/WORKLOG.md) · `research/training-seed-ensemble` |
| `adapted_quote_linking` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/adapted-quote-linking/solutions/adapted_quote_linking/WORKLOG.md) · `explore/adapted-quote-linking` |
| `boundary_veto` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/boundary-veto/solutions/boundary_veto/WORKLOG.md) · `explore/boundary-veto` |
| `patient_set_risk` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/research/patient-set-risk/solutions/patient_set_risk/WORKLOG.md) · `research/patient-set-risk` |
| `biomedical_long_context` | [方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/research/biomedical-long-context/solutions/biomedical_long_context/WORKLOG.md) · `research/biomedical-long-context` |

机制查重先读[机制地图](../research/method-coverage-map-20261024/MAP.md)（明确为13:08历史快照），再读相应路线的最新记录；不把旧快照的待完成项当作重启指令。各链接目标已按本地对应worktree核实存在；不为目录整理访问官网、重评分或下载模型。

## 历史诊断索引

| 实验 ID | 类型 / 状态 | 要回答的问题 | 结论与证据 |
|---|---|---|---|
| [E-20261023-001](../../archive/legacy-20261023/docs/experiments/E-20261023-001-legacy-audit.md) | 已归档 / 接手诊断，历史补录 | 旧项目能否复现，现有结果是否足以支撑继续沿用 | 历史审查：CPU 版可复现，但评测及验证不可信；已整体归档，用户选择从头开始 |

## 如何记录

- 新实验复制 [TEMPLATE.md](TEMPLATE.md)，使用 `E-YYYYMMDD-NNN` 和简短主题作文件名；相关子运行归入同一记录。索引在开始及结案时更新。
- 运行前填问题、假设、基线、主要变量、数据划分与评测版本、成功标准；结束后补结果、成本、局限及决定。关键失败、中断、负结果同样保留。
- 通常一份记录约 30–80 行即可；重复过程用表格，复杂证据链接单独文件，不转贴原始长日志。没有运行的字段写“未运行/未记录”，不要填虚构数字。
- 小型配置、split manifest、关键 metrics 及少量错误例子可以入 Git；大产物默认保存在 `../patientphex-runs/<experiment_id>/<run_id>/` 或其他明确的持久位置。
- 对需保留的大产物记录路径、大小、SHA-256 和重建方式。只有 `/tmp` 副本的产物，应在交接前保存关键证据或明确为可丢弃，不写成已持久保存。
- 每项指标注明训练/验证/交叉验证/诊断/官方及统计口径。与基线比较前确认划分、指标和代码版本一致；有代码改动未提交时记录文件/补丁校验值。
- 碰到可复用的坑，同步更新 `HANDOFF.MD`；本文件只索引实验，不重复工作台账。

- 长上下文 GP 原跨度删除控制：[路线台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/63aa3829/solutions/long_context_span_veto/WORKLOG.md)；固定 max<0 全17窗负结果，关闭且不扫描阈值。
