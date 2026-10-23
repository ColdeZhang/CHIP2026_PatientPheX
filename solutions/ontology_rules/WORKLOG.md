# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **最新阶段完成**：真正HPO多类远监督固定总100ep已结束，30ep精确续70ep；HPO训练probe512/512全对且全部p≥.95。固定fold0增量Score **.712548706**，较source .710359060提高.002190，但191新增实体仅60TP/131FP，84关系27TP/57FP；men/micro下降，doc/macro上升，0错警。弱于已知fold0置信度树.714808，不推广、不跑原dev。
- **当前任务**：固定Sap一致性控制已完成，fold0 **.725539421**，较base +.015180、较原100ep +.012991；191新增留36（22TP/14FP），新增16关系（12TP/4FP），0错警，[结果](evidence/round2/phenotagger-sap-consensus-results.json)。本次只用原train48 detector/bank、固定.65/.92+pure nearest ID一致，无重训/扫参，原base全保留；Sap labelmap/装配/v2独审已PASS。当前真实词面适配数据与CPU初始化已核：1749对×16+原回放=145102行；真实max22词元，无新增token标签冲突；全模型从100ep精确载入，新Adam状态0。固定3ep/1701步/5e-6，[计划](evidence/round2/phenotagger-adaptation-plan.json)已冻结，只终点原rawp.95同池对照；源码905cc41推送后fresh核卡已于17:19:37Z启动，实际GPU初始化日志确认parent权重精确/newAdam0/newstep0；训练数值正常，不看中间heldout。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3 GPU0适配managed proc_42257843-82c3-431f-afec-7c1756b0cca0、PID1133377；新目录phenotagger-fold0/adaptation3ep/trained/，每120秒latest，训练后同进程固定终点推理，预计约17:24结束。[作业/恢复入口](evidence/round2/phenotagger-adaptation-job.json)。原100ep及30ep不覆盖。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：905cc41（真实词面适配冻结、含main512cf8d），2026-10-23约17:19UTC，origin/round2/ontology-rules。本次作业恢复入口随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已按同数据/顺序/源码做明确标记的CPU精确重建；原14,480池p完全相同、阈值与原ID策略0差异。本路线只读系数，不重复重建；旧原ID含词典优先，不能冒充纯nearest ID。
