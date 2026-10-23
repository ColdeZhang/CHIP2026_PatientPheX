# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **正在推进**：独立从通用BiomedBERT重训真正的HPO多类远监督分类器的成本测量。采用指定HPO全scope同义词+lemma、train48原文弱负，先核标签/冲突与100步throughput、显存、encoder更新；不下载外部PhenoTagger成品、不用GSC、暂不评价fold0性能选epoch。
- **下一步**：CPU构建材料并保存来源；.3 GPU0由root分配给此100步测量，启动前再次核空。测量后报告30epoch预算再定固定训练长度；不会把小预算pilot冒称官方100epoch复现。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：当前新任务尚未GPU启动。.3 GPU0已由root重新分配，先前全部候选/CE作业已结束。只读Biomed源和semantic-neural/venv，输出放自有ontology-round2/phenotagger-fold0；本地外存研究与设计入口 `research/phenotagger-classifier-design-stats.json`。A800由root协调其他路线。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：58f780e，2026-10-23 14:29UTC，origin/round2/ontology-rules。本次记录随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已获root授权，正在只读原缓存按相同数据/顺序/源码做明确标记的CPU精确重建，必须全池复核才可用于新跨度。本路线不重复重建。
