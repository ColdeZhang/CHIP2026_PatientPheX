# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev**：名义最高 **.646360012**（Pheno48 + HGB/Sap64固定过滤），比稳定基线OOF HGB v1 **.646244333**仅+.000115679，不视为实质提升；默认仍保留稳定base。首日 .612597。v1独立重训字节复现；auto的10%随机候选验证不是文献隔离，但无dev泄漏。
- **最新内fold0候选**：真实HPO100ep + 固定Sap .65/.92/纯top1一致，**.725539421**（原base .710359060），新增36实体22TP14FP/16关系12TP4FP；labelmap/模型谱系/CPU装配与v2独审PASS。只是一折开发控制，未原dev验收、不合并。
- **共享组合结案**：CRF+GLiNER fullLR base .711324992上固定补Pheno实体、保留旧relations得 **.712586161**；同train64 LR重新关联得 **.712070726**。两pred先冻结，双方独立复评一致，12新实体4TP8FP；不称本体独立.712，本路线名义最好仍.646360。[固定计划](evidence/round2/phenotagger-crf-gliner-dev-plan.json)和[结果](evidence/round2/phenotagger-crf-gliner-dev-results.json)。关闭当前Pheno继续训练与组合扫描。
- **当前任务**：outerfold0弱链接[CPU提案](evidence/round2/gliner-rejected-link-proposal.json)获root批准一次176条9B base控制：固定top5/C1..C5/ABSTAIN，单条2048输入/64输出、greedy/no-thinking、900秒硬cap。已冻结[运行计划](evidence/round2/gliner-rejected-link-run-plan.json)与[严格解析/装配检查](evidence/round2/gliner-rejected-link-parser-checks.json)，等retrieval的A8002ep+58adapter推断明确交卡后运行；尚未LLM推理/评分。不读gold选例、不跑原dev，不新增-1。
- **最近结案**：真实词面3ep raw .694907353、同Sap .721696137均低于父对照，data/init/两终点独审PASS；配比关闭，不追加词面/轮次/阈值。[raw](evidence/round2/phenotagger-adaptation-results.json)和[配对过滤](evidence/round2/phenotagger-adaptation-sap-results.json)完整保留。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：当前无自有运行作业；176链接源码/输入已上传A800并核SHA，准备[独审PASS](evidence/round2/gliner-rejected-link-audit-index.json)。明确等待retrieval的训练与58条adapter推断两项完成交接，再运行外存 `gliner-rejected-link-fold0/start_a800.py`；不因GPU暂时空闲自行抢卡。组合外存 `phenotagger-crf-gliner-dev16/`，所有父模型/负结果保持原位。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：4083244（176严格协议/合成检查及正常合并main a0e9d0d共享规则），2026-10-23约18:00UTC，origin/round2/ontology-rules；合并未改本路线源码或评测器。准备独审与旧关系CE结论在下一检查点保存。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已按同数据/顺序/源码做明确标记的CPU精确重建；原14,480池p完全相同、阈值与原ID策略0差异。本路线只读系数，不重复重建；旧原ID含词典优先，不能冒充纯nearest ID。
