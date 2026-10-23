# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **固定dev最佳**：CPU四折选出的 k21/.35 为 **.625073738**（首日 .613517597）；独立重训/逐字节预测复现通过。后续GPU研究仅在train64内部fold0/fold1，尚未重评dev。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：整患者开放集合SFT完整47×2配对严重负：base grounded .375189、epoch002 grounded .396343，同实体规则 .711354；adapter 44合法中26空数组，3触顶，停止这套配置。原文quote存在不代表自由HPO ID正确。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：9B原58 base **.633859674→SFT .695443948**，关系208TP/102FP→226TP/50FP；仍低同实体rules .711353522。58/58严格JSON、415/415候选合法。额外schema投影使base反降 .628353579，SFT不变；两主预测保留。A800已18:09:25明确交本体176短批，归还后仅一次9B全文probe/固定控制；原dev未动。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. **9B原58完整配对已结束**。原adapter作业 `proc_301c8be3-bba6-4c69-9634-bf4a0dbf752b` 与本地watcher93066均正常结束；58/58严格JSON、0触顶、415/415候选合法、0格式错，生成203.10秒。原base .633859674→adapter .695443948，仍低rules；小证据见[配对结果](evidence/round2/qwen35-relation-paired-result.json)，完整raw/预测/SHA在外存 `qwen35-preparation/relation-quality-plan/`。不重启，不挑epoch。
2. **训练已结束**。fresh204行×2ep/52updates，原作业 `proc_c35c5440-2f32-4842-b4aa-8e444173673b` 与monitor89754已结束。实际1855.10秒，allocated37.48GB/reserved78.50GB分开记录，无OOM；epoch002 adapter SHA8e32daee…，两epoch及Adam/scheduler/RNG全实际SHA在外存 `relation-training-preparation/checkpoint-actual-hashes.json`。完整配置、命令与终点见[完成证据](evidence/round2/qwen35-relation-training-complete.json)，不挑epoch1质量，不宣称新的数值resume等价。
3. **A800当前运行9B full58**。本体176短批18:11:29明确归还后，最长16245→374tokens probe通过：13.151秒、allocated22.58GB、8/8CID合法（不评分/不复用）。full58于18:13:50 UTC启动 `proc_9e7169c1-cbe9-466e-ae83-d61a6d29c80b`，本地session26431，25分钟timeout+30秒TERM宽限；外存 `full-context9b-a800/full58-plan.json` / job / monitor。cap等价证据支持复用原A800 base58。后继是已批准的204 OOF官方base推理25分钟；源码/契约已冻结，[树准备证据](evidence/round2/llm-feature-tree-preparation.json)，不得用SFT回推替代。
4. **A30原dev迁移准备与probe**。root批准一次同A30 base+epoch002迁移，固定c1c2e508基底955entities/466concepts/36patients，64条局部prompt总202743/max5194。源码/输入纯度已独审；base最长probe已通过（5.708秒、19.46GB、8/8CID），adapter最长probe进行中，本地session43829。外存 `dev-relation-transfer-preparation/probe-plan.json` / `probe-base-job.json` / `probe-adapter-job.json`。两probe后才冻结两臂共享35分钟预算；不复用probe、不用schema投影，双预测完整冻结后统一评分。此时没有原dev新模型分数，混合训练规模如实记录见[准备证据](evidence/round2/qwen35-dev-transfer-preparation.json)。
5. **8B全文控制已结束且释放A30**。同A30两臂58/58、0触顶/删demo，原58 .618120269→全文 .610344254；947实体与所有候选不变。配对墙钟805.38秒；FP的‘不属任何指定患者gold概念’293→323，‘属其他指定患者’53→70，前者不能直接认定真实背景。双raw与双pred先冻结再评测，原解析回退2→7，未应用schema控制。外存 `full-context8b-a30/paired-summary.json` 及[结果](evidence/round2/full-context8b-paired-result.json)，独立终点审查待完成。
6. **模型/adapter复制均已结束并关源**。A80015文件→A30共19,329,392,091B，2116.35秒；原monitor96986已退出，独立重读15文件全部size/SHA同源，原source进程停止、端口41236拒绝连接。另本轮epoch002两adapter文件173.18MB以独立allowlist复制18.62秒，全SHA与训练检查点一致，自有41237源也已停止/核关闭。没有残留复制/GPU自动等待任务；[持久副本证据](evidence/round2/qwen35-a30-persistent-copy.json)。A30模型实际目录 `case-retrieval/qwen35-preparation/models/Qwen3.5-9B`，新版runtime保持与旧8B隔离。
7. **环境与来源**。A800只读Python `ontology-round2/qwen-venv/bin/python`，私有 `case-retrieval/qwen35-preparation/a800-migration/runtime` 提供TF5.20.2/PEFT0.21.2；模型 `/dev/shm/patientphex-qwen35-1045-c20223623576` 为已全SHA核验15文件的易失缓存，持久源在A100 `476f81cf-09e7-445b-a7f0-d0e12b12e50f:case-retrieval/qwen35-preparation/models/Qwen3.5-9B`。HF c202236…/MS aa2aaf…，LICENSE差异单独处理；A100 GPU等待计划已停止。
8. **预设CPU schema控制已完整负**。e9867a0a源码在两原始响应冻结后共同应用，base3行/11CID/30声称归属投影使Score .633860→.628354；adapter全合法0变化。原raw/原解析所有SHA前后一致。未知CID/HPA不猜修，原解析仍为主结果；外存 `relation-quality-plan/schema-control/summary.json`。事后聚合显示SFT减少无任一目标gold概念FP60→37、其他患者FP42→13，但比rules少29TP；FN153中113不在固定实体概念池。聚合不用于改提示/候选。
9. **已结案**：整患者自由HPO47×2负在 `patient-set-fold0-bf16-v1/evaluation-summary.json`；抽取跨fold1七臂在 `sft-extract-fold1-bf16-v1/fold1-final-report.json`，均独立CPU重放通过。不继续无收益网格，Qwen3-8B旧模型/环境/checkpoint不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `6e9a1b9`（9B配对与schema负结果；此前144325e正常merge main@512cf8d共享规则），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
