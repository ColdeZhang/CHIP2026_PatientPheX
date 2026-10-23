# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **正在推进**：独立从通用BiomedBERT重训真正的HPO多类远监督分类器的成本测量。采用指定HPO全scope同义词+lemma、train48原文弱负，先核标签/冲突与100步throughput、显存、encoder更新；不下载外部PhenoTagger成品、不用GSC、暂不评价fold0性能选epoch。
- **下一步**：root已冻结batch256/Adam2e-5/30ep，从通用base fresh初始化；正HPO/None分项曲线、每240秒原子latest及最终final。真实4步连续vs2+2恢复13项全通过（model/Adam/全RNG逐张量相同），配置与源码推送后开约22分钟正式训练；仅终点评价fold0，不选epoch。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：100步proc_e1df12a9…与batch256×30步proc_87fef356…测量完成，待root协调固定训练预算；日志ontology-round2/phenotagger-fold0/measure100.log，结果measure100/measurement.json。先前全部候选/CE作业已结束。只读Biomed源和semantic-neural/venv，输出放自有ontology-round2/phenotagger-fold0；本地外存研究与设计入口 `research/phenotagger-classifier-design-stats.json`。A800由root协调其他路线。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：2ee7856（已正常合入main 811b8cc），2026-10-23约14:57UTC，origin/round2/ontology-rules。本次记录随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已获root授权，正在只读原缓存按相同数据/顺序/源码做明确标记的CPU精确重建，必须全池复核才可用于新跨度。本路线不重复重建。
