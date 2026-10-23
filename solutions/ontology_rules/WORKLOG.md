# 本体规则工作台账

实验 `E-20261023-007`；分支 `round2/ontology-rules`。先读根 HANDOFF.MD，本方案只记录关键状态，完整过程见[实验](experiments/E-20261023-007.md)。

- **目标与约束**：固定 train64/dev16、公共 evaluator v2，Score ≥ .8；训练资源按文献隔离。未达标，不合并；本轮禁止官方上传。
- **当前原 dev 最好**：OOF HGB v1 **.646244333**，首日 .612597。v1、固定160对照与嵌套旧树均已独立重训字节复现；auto 的10%随机候选验证不是文献隔离，但无 dev 泄漏。
- **正在推进**：CRF候选+同源置信度加入固定160树。outer0 三控制 .705023/.710359/.714808；outer1 .659436/.663220/.664419，顺序均为无CRF/来源标记/再加置信度。四折已齐，source每折都优于无CRF；pooled .692527 vs置信度 .692974，差≤.001故选简单source，14:14:44Z策略冻结。唯一预选source原dev .639287，低于best .646244，不升级。
- **下一步**：冻结LR归属迁移fold0/1均下降，已停止。下一为自有3×32 OOF候选→文献/概念/患者聚合归属，训练侧覆盖88.27%，首固定HGB单独 .672308、规则并集 .704175均低于rules .710359；第二折同配置亦负（pool .633039/union .656995 vs rules .663220），已结案不跑dev。阈值校准与全文共指仍待新实验，未扫参数。已完成四折预冻结选型及唯一原dev，不回扫另两配置。
- **已关闭负结果**：零样本Qwen归属 .581179；CE1/3epoch及先验/融合无实质收益。HPO双头宽池最佳 .706632、同池旧LR .706174，增益仅.000459，弱于CRF置信度树；不升级默认、不跑dev。
- **审计**：3542ec3 的 fold0 多来源/置信度三控制独立重放全部字节一致；3个train32真实模型导分、outer0导分、3455候选独立BIO/TFIDF复算通过。见外部 validation/calibrate-multi-review.json，SHA77da3227…。
- **资源**：.3 GPU0全部本路线作业已结束，14:09左右复查无该卡compute进程并向root释放；outer3作业 proc_4339063f-c003-4308-bae6-c55534f60ab0。后续当前只CPU。A800环境已CUDA/全Qwen实测，通过root交序列路线，只读共享。
- **恢复入口**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer{1,2,3}/evaluate_controls.py` 与各 controls-results.json；全量预测/模型/日志不入Git。小证据见[evidence索引](evidence/round2/README.md)，研究与踩坑见[NOTES](NOTES.md)。
- **最近已核验远端**：dd50158，2026-10-23约14:28UTC，origin/round2/ontology-rules。本次记录随后提交推送。

来源澄清：冻结首轮Sap候选生成器为016bb77/0084d33的 gpu_candidates.py（SHA dc5cc0…）；0673f85是后续装配证据检查点。原LR没有序列化；检索路线已获root授权，正在只读原缓存按相同数据/顺序/源码做明确标记的CPU精确重建，必须全池复核才可用于新跨度。本路线不重复重建。
