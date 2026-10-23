# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **正在推进**：CRF候选+同源置信度加入固定160树。outer0 三控制 .705023/.710359/.714808；outer1 .659436/.663220/.664419，顺序均为无CRF/来源标记/再加置信度。outer2/3 全部CPU上游和GPU Sap缓存已完成并SHA回收，正在固定三组评分，未看原dev新结果。
- **下一步选型规则**：比较完整四折 pooled Score；最高者胜，差距≤.001时取更简单者。保存各折方向与输出SHA，先冻结策略文件及源码，再仅一次 train64→原dev。不得对新折追加阈值。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3 GPU0全部本路线作业已结束，14:09左右复查无该卡compute进程并向root释放；outer3作业 proc_4339063f-c003-4308-bae6-c55534f60ab0。后续当前只CPU。A800环境已CUDA/全Qwen实测，通过root交序列路线，只读共享。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：3542ec3，2026-10-23约13:53UTC，origin/round2/ontology-rules。本次记录随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已获root授权，正在只读原缓存按相同数据/顺序/源码做明确标记的CPU精确重建，必须全池复核才可用于新跨度。本路线不重复重建。
