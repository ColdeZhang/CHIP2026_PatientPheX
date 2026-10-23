# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **正在推进**：真正HPO多类远监督全encoder正式30ep训练（15:00:33Z启动），不是先前冻结双头。固定batch256/Adam2e-5/max32，从通用BiomedBERT fresh初始化，指定HPO2026全scope+lemma和train48弱负；不使用GSC/外部成品。
- **下一步**：root已冻结batch256/Adam2e-5/30ep，从通用base fresh初始化；正HPO/None分项曲线、每240秒原子latest及最终final。真实4步连续vs2+2恢复13项全通过（model/Adam/全RNG逐张量相同），759aa24推送后已开正式训练；仅终点评价fold0，不选epoch。终点固定p.95保留原source HGB全部实体/关系，冻结政策及宽池SHA见[evidence](evidence/round2/phenotagger-inference-plan.json)。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3 GPU0作业proc_f1cb3a1a-273a-44b0-81f2-b850593be9a3运行中，预计15:23左右训练终点。远端ontology-round2/phenotagger-fold0/formal30ep.log与formal30ep/latest.pt；每240秒原子保存，恢复用原命令加--resume。实际命令/运行位置见[evidence job](evidence/round2/phenotagger-formal-job.json)。输出只自有目录，共享Biomed/解释器只读。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：759aa24（含main 811b8cc共享更新），2026-10-23约15:00UTC，origin/round2/ontology-rules。本次记录随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已获root授权，正在只读原缓存按相同数据/顺序/源码做明确标记的CPU精确重建，必须全池复核才可用于新跨度。本路线不重复重建。
