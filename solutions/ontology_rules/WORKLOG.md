# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **最新内fold0候选**：真实HPO100ep + 固定Sap .65/.92/纯top1一致，**.725539421**（原base .710359060），新增36实体22TP14FP/16关系12TP4FP；labelmap/模型谱系/CPU装配与v2独审PASS。只是一折开发控制，未原dev验收、不合并。
- **当前任务**：root授权一次原固定dev验证，复用Pheno父100（48篇、原train64子集）+ 最好冻结HGBv1 base .646244（64篇）+原Sap64，明确混合训练规模。[两臂计划](evidence/round2/phenotagger-original-dev-plan.json)已冻结：原rawp.95与相同Sap .65/.92/pureID过滤，两个pred先冻结再一起评价，禁止结果后改参。已生成77329无标签候选/53034词面；Sap64由原缓存CPU精确恢复，12811旧池p0差、阈值/ID0差，无新编码。sequence已释放，17:40:08Z fresh核空后原dev两臂pipeline启动；只推理，未评价gold。
- **最近结案**：真实词面3ep raw .694907353、同Sap .721696137均低于父对照，data/init/两终点独审PASS；配比关闭，不追加词面/轮次/阈值。[raw](evidence/round2/phenotagger-adaptation-results.json)和[配对过滤](evidence/round2/phenotagger-adaptation-sap-results.json)完整保留。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3GPU0原dev pipeline proc_be26d844-cb88-4825-95ca-d34d2024de53、PID1134792；原Pheno100推理→raw→Sap64→filtered，最后写双预测freeze marker。[启动/接手入口](evidence/round2/phenotagger-original-dev-job.json)，完整日志/指纹外存phenotagger-dev16/。所有父模型保持原位不覆盖。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：cb8d085（原dev两臂冻结、含main512cf8d），2026-10-23约17:38UTC，origin/round2/ontology-rules。本次原dev作业入口随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已按同数据/顺序/源码做明确标记的CPU精确重建；原14,480池p完全相同、阈值与原ID策略0差异。本路线只读系数，不重复重建；旧原ID含词典优先，不能冒充纯nearest ID。
