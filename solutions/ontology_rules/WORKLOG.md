# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **最新阶段完成**：真正HPO多类远监督固定总100ep已结束，30ep精确续70ep；HPO训练probe512/512全对且全部p≥.95。固定fold0增量Score **.712548706**，较source .710359060提高.002190，但191新增实体仅60TP/131FP，84关系27TP/57FP；men/micro下降，doc/macro上升，0错警。弱于已知fold0置信度树.714808，不推广、不跑原dev。
- **当前任务**：Pheno100独立CPU复评已PASS；固定Sap一致性过滤191新增实体（p≥.65、cos≥.92、纯nearest ID等于Pheno ID），不改base、不扫阈值。原fold0 train48 LR精确重建证据已核，bank42,236的文本顺序/缓存key完全一致；只构造label映射、不fit。全拒绝/全接受CPU fixture分别逐字节恢复base/100ep结果。[运行前冻结](evidence/round2/phenotagger-sap-consensus-plan.json)；等GPU短编码后冻结预测再评分。真实训练肯定词面适配仅并行统计提案，尚未训练。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3训练proc_fbc1734b…及终点proc_d2304157…均退出，16:52复查GPU0空并归还root。远端ontology-round2/phenotagger-fold0/formal100ep-from30保留latest+final（final SHAd3aaf718…）；原formal30ep未覆盖。7项终点产物/2.96MB候选已全SHA回收到本地外存collected，当前无GPU作业；[启动/恢复入口](evidence/round2/phenotagger-continuation100-job.json)。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：675ecab（100ep结案、含main512cf8d），2026-10-23约16:58UTC，origin/round2/ontology-rules。当前Sap控制源码/计划随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已按同数据/顺序/源码做明确标记的CPU精确重建；原14,480池p完全相同、阈值与原ID策略0差异。本路线只读系数，不重复重建；旧原ID含词典优先，不能冒充纯nearest ID。
