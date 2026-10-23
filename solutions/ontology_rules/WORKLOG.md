# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev**：名义最高 **.646360012**（Pheno48 + HGB/Sap64固定过滤），比稳定基线OOF HGB v1 **.646244333**仅+.000115679，不视为实质提升；默认仍保留稳定base。首日 .612597。v1独立重训字节复现；auto的10%随机候选验证不是文献隔离，但无dev泄漏。
- **最新内fold0候选**：真实HPO100ep + 固定Sap .65/.92/纯top1一致，**.725539421**（原base .710359060），新增36实体22TP14FP/16关系12TP4FP；labelmap/模型谱系/CPU装配与v2独审PASS。只是一折开发控制，未原dev验收、不合并。
- **共享组合结案**：CRF+GLiNER fullLR base .711324992上固定补Pheno实体、保留旧relations得 **.712586161**；同train64 LR重新关联得 **.712070726**。两pred先冻结，双方独立复评一致，12新实体4TP8FP；不称本体独立.712，本路线名义最好仍.646360。[固定计划](evidence/round2/phenotagger-crf-gliner-dev-plan.json)和[结果](evidence/round2/phenotagger-crf-gliner-dev-results.json)。关闭当前Pheno继续训练与组合扫描。
- **最近链接结案**：176条9B base控制完整运行74.032秒，125有效选择/51ABSTAIN、无格式失败/截断，新增124实体36TP88FP与59关系14TP45FP。18:11:44Z先冻pred bfc28bf3…再评，Score **.721504431→.711139780**、四分项均降、0错警。[完整负结果](evidence/round2/gliner-rejected-link-results.json)，不修prompt/重采样/扫参/原dev迁移。proc_93a9cd78已退出、GPU2核空明确交回retrieval。
- **MIL结案**：固定3ep/559step真实完成53.45秒，encoder/head梯度与更新非零，414终点分数完整。18:20:11Z先冻纯删除pred7f443fad…再评：**.711353522→.702752693**，移除27TP/45FP，men/doc不变但两关系分项下降；0错警，[负结果](evidence/round2/relation-mil-results.json)。不扫阈值或epoch，不跑原dev，终点独审PASS（CPU预测字节与指标精确复现，见[审查索引](evidence/round2/relation-mil-audit-index.json)）。
- **当前任务**：root已授权方案A一次严格OOF CE特征控制：3×train16生成6池、3×train32原recipe CE3ep、旧outer48只补32缺键。真正原树为84列/.4（不是299列/.5）；原树98ffd7c5… CPU已字节复现，只加1列CE，固定85列/160iter/.4，双预测先冻后评。额外GPU全阶段总cap720秒，CPU数据/来源/源已冻结并交独审，尚未启动。[运行计划](evidence/round2/ce-oof-run-plan.json)，禁止旧11180泄漏切片训练。
- **最近结案**：真实词面3ep raw .694907353、同Sap .721696137均低于父对照，data/init/两终点独审PASS；配比关闭，不追加词面/轮次/阈值。[raw](evidence/round2/phenotagger-adaptation-results.json)和[配对过滤](evidence/round2/phenotagger-adaptation-sap-results.json)完整保留。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源/恢复**：当前无自有GPU作业；sequence已18:31:30Z明确释放.3GPU0，待CE准备独审后fresh核卡再启动。MIL已完整回收且18:20Z进程退出/.3GPU0核空报root释放，A800也已归还retrieval。MIL外存 `relation-mil-fold0/` 留完整来源/commands/metrics；远端 `ontology-round2/relation-mil-fold0/formal3ep` 留latest model/Adam/RNG、final27c6d1a1…及tokenizer；未实现自动resume。所有父模型/负结果保持原位。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：5dfbc38（严格嵌套CE成本核对），2026-10-23约18:22UTC，origin/round2/ontology-rules；本次保存MIL终点独审与CE严格嵌套可行性，不启动新作业。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已按同数据/顺序/源码做明确标记的CPU精确重建；原14,480池p完全相同、阈值与原ID策略0差异。本路线只读系数，不重复重建；旧原ID含词典优先，不能冒充纯nearest ID。
