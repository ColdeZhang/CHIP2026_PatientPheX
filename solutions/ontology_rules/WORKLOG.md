# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **最新阶段完成**：真正HPO多类远监督全encoder固定30ep已结束（15:28前退出），未使用GSC/外部成品。固定p.95增量fold0 **.710453046**，相对source基线 .710359060仅+.000094；11新增实体4TP/7FP，6关系2TP/4FP，0错警。不升级默认、不跑原dev。
- **当前任务**：root按训练欠拟合证据已授权从完整30ep精确续至总100ep（额外70ep），同数据/head/batch256/lr2e-5/seed/p.95，保持30ep模型结果。独立source100/CPU恢复与准备审查已通过；15:48:52Z核空后已启动，实际model/Adam/CUDA RNG恢复exact，从第31轮继续，当前31轮正类acc.77194。计划见[续训冻结](evidence/round2/phenotagger-continuation100-plan.json)，不在中间轮评价heldout。预计额外纯训练59.67分钟+存档开销；原型初始化仅保留假设。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3训练proc_f1cb3a1a…与终点proc_34e31551…均退出，15:29复查GPU0空并交root/sequence。远端ontology-round2/phenotagger-fold0/formal30ep保留latest+final（最终SHAd024b9e5…）；全部小结果/2.74MB候选已SHA回收到同名本地外存collected。实际命令、恢复入口与结果见[evidence](evidence/round2/phenotagger-endpoint-results.json)。100ep当前作业proc_fbc1734b-e1e9-484a-a917-0ad5f4249823/PID1127483在.3 GPU0运行，预计16:50–16:55UTC终点；独立目录formal100ep-from30，首次240s保存后用计划内--resume命令恢复。见[实际启动/恢复入口](evidence/round2/phenotagger-continuation100-job.json)。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：cd4adac（含main 811b8cc共享更新），2026-10-23约15:43UTC，origin/round2/ontology-rules。本次100ep运行记录随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已获root授权，正在只读原缓存按相同数据/顺序/源码做明确标记的CPU精确重建，必须全池复核才可用于新跨度。本路线不重复重建。
