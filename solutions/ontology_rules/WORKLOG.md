# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **最新阶段完成**：真正HPO多类远监督固定总100ep已结束，30ep精确续70ep；HPO训练probe512/512全对且全部p≥.95。固定fold0增量Score **.712548706**，较source .710359060提高.002190，但191新增实体仅60TP/131FP，84关系27TP/57FP；men/micro下降，doc/macro上升，0错警。弱于已知fold0置信度树.714808，不推广、不跑原dev。
- **当前任务**：Sap控制fold0 .725539421已独审PASS。真实词面3ep适配完成且raw结果为负：**.694907353**（父同raw .712548706），新增实体87TP239FP/关系31TP98FP，0错警，[结果](evidence/round2/phenotagger-adaptation-results.json)。训练真实词面acc .99825并未改善泛化。按root授权只做唯一相同Sap过滤的公平对照：326新增请求已冻结，不改base/阈值/模型，[计划](evidence/round2/phenotagger-adaptation-sap-plan.json)；若不胜父过滤.725539，则关闭本3ep配比，不再按本折调词面/轮次。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：适配proc_42257843…已结束，1701steps/190.42秒纯训练；终点7文件全SHA回收，GPU0 fresh核空。final SHA a4d25265…保留adaptation3ep/trained/，原100ep与30ep不覆盖。下一326 Sap短批使用独立adaptation3ep/sap-consensus/，当前仅CPU准备。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：ad5ebbb（适配运行恢复入口、含main512cf8d），2026-10-23约17:21UTC，origin/round2/ontology-rules。本次负结果与固定过滤计划随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已按同数据/顺序/源码做明确标记的CPU精确重建；原14,480池p完全相同、阈值与原ID策略0差异。本路线只读系数，不重复重建；旧原ID含词典优先，不能冒充纯nearest ID。
