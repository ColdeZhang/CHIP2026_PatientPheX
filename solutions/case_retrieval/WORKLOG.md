# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **原dev两种口径**：原独立检索 CPU k21/.35 **.625073738**（首日 .613517597）；共享CRF+GLiNER955实体 + Qwen3.5关系SFT48 **.711210414**，低于同池fullLR **.711324992**。后者是共享组件组合进展，不冒称纯检索分；两者均未达.8。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：9B微调能降低关系FP，但直接替换仍损召回；原dev微调271TP/64FP/182FN vs fullLR331/203/122。204严格OOF base决策作为4事实的固定HGB对照也负：无LLM .653192→有LLM .649124，低于内fold0 rules .711354。均冻结，不依据heldout调阈值。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：fresh full64关系SFT实际运行中；A800 GPU2 `proc_ccb3fa32-ddf7-424e-b15f-a3c9a7003ef8` 于18:55:25 UTC启动，19:55:25 UTC训练硬截止（TERM后30秒KILL）。独立CPU/data/guard审查与6153最长fresh probe均已通过；265行×2ep、68updates/4warmup、cap6400只拒超限，唯一epoch002，不续旧48或probe。正式已进入训练；尚无完成或质量结论。恢复入口外存`full64-relation-cpu-preparation/formal-job.json`与`formal-monitor.json`，本地watcher8857，勿重复启动。
- **已授权后继，尚未启动**：full64完整终点后仅一个原dev新adapter臂；同A30旧64提示/955实体/466CID/原decoder与parser，900秒+30秒grace，旧base/SFT48/fullLR按SHA复用。新权重SHA待真实完成；先等本体NER明确释放.100 GPU1再fresh核空。单臂操作源码/10合成检查/静态审查已冻，先freeze预测再一次评新臂，见[准备证据](evidence/round2/qwen35-full64-dev-single-arm-preparation.json)。
- **补充固定并集也负**：内fold0 rules∪9B SFT .711354→.704652，15新增只有1TP/14FP；实体不变，无门控/阈值策略，关闭且不评原dev。
- **严格CPU校准已结案**：F1 meta-val固定grid选.30，outer旧.5 .653192→.3 .669003，但低rules .711354，关闭且不迁移原dev。实际2506训练行/594正/297列，完整CPU约44秒，无GPU；原模型/基线字节未变，见[结果](evidence/round2/strict-meta-calibration-result.json)。
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
3. **A800204与固定两树已完成**。原作业 `proc_0adf6588-bcd4-4b60-a490-a22e0152c958` 正常退出；204strictJSON/0hit，648409实际prefix精确、生成735.192秒。原1477训练CID全合法，验证397合法/18旧格式失败。609374固定两HGB先双freeze再评：无LLM .653192044→4facts .649124234，分别201TP/66FP与188TP/60FP，均弱于rules .711354；[终点证据](evidence/round2/llm-feature-tree-result.json)，外存 `llm-tree-fold0-results/summary.json`。原GPU2于18:38:45.877Z fresh核空已交root，无自动GPU后继。
4. **A30原dev配对已完成并释放**。两臂64完整/466CID全合法/0触顶回退，same955实体。base .646040894→epoch002 .711210414，仍低fullLR .711324992；无schema/阈值修改。实际339.672+337.989秒，全编排748.516秒；双pred freeze `ad68570c…`先于读gold。正确终态入口外存 `dev-relation-transfer-preparation/paired-completion.json` / paired-summary.json；早期paired-plan是被freeze绑定的历史回收快照，不覆盖其SHA。A30于18:36:44.771Z fresh核GPU1空后交root；[终点证据](evidence/round2/qwen35-dev-transfer-result.json)。原独立检索.625074与本组合.711210分开保留。
5. **8B全文控制已结束且释放A30**。同A30两臂58/58、0触顶/删demo，原58 .618120269→全文 .610344254；947实体与所有候选不变。配对墙钟805.38秒；FP的‘不属任何指定患者gold概念’293→323，‘属其他指定患者’53→70，前者不能直接认定真实背景。双raw与双pred先冻结再评测，原解析回退2→7，未应用schema控制。外存 `full-context8b-a30/paired-summary.json` 及[结果](evidence/round2/full-context8b-paired-result.json)，独立终点审查已通过，报告外存validation/full-context8b-terminal-review。
6. **模型/adapter复制均已结束并关源**。A80015文件→A30共19,329,392,091B，2116.35秒；原monitor96986已退出，独立重读15文件全部size/SHA同源，原source进程停止、端口41236拒绝连接。另本轮epoch002两adapter文件173.18MB以独立allowlist复制18.62秒，全SHA与训练检查点一致，自有41237源也已停止/核关闭。没有残留复制/GPU自动等待任务；[持久副本证据](evidence/round2/qwen35-a30-persistent-copy.json)。A30模型实际目录 `case-retrieval/qwen35-preparation/models/Qwen3.5-9B`，新版runtime保持与旧8B隔离。
7. **环境与来源**。A800只读Python `ontology-round2/qwen-venv/bin/python`，私有 `case-retrieval/qwen35-preparation/a800-migration/runtime` 提供TF5.20.2/PEFT0.21.2；模型 `/dev/shm/patientphex-qwen35-1045-c20223623576` 为已全SHA核验15文件的易失缓存，持久源在A100 `476f81cf-09e7-445b-a7f0-d0e12b12e50f:case-retrieval/qwen35-preparation/models/Qwen3.5-9B`。HF c202236…/MS aa2aaf…，LICENSE差异单独处理；A100 GPU等待计划已停止。
8. **预设CPU schema控制已完整负**。e9867a0a源码在两原始响应冻结后共同应用，base3行/11CID/30声称归属投影使Score .633860→.628354；adapter全合法0变化。原raw/原解析所有SHA前后一致。未知CID/HPA不猜修，原解析仍为主结果；外存 `relation-quality-plan/schema-control/summary.json`。事后聚合显示SFT减少无任一目标gold概念FP60→37、其他患者FP42→13，但比rules少29TP；FN153中113不在固定实体概念池。聚合不用于改提示/候选。
9. **已结案**：整患者自由HPO47×2负在 `patient-set-fold0-bf16-v1/evaluation-summary.json`；抽取跨fold1七臂在 `sft-extract-fold1-bf16-v1/fold1-final-report.json`，均独立CPU重放通过。不继续无收益网格，Qwen3-8B旧模型/环境/checkpoint不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `d0513c2`（严格CPU校准终点/full64实际成本；此前144325e正常merge main@512cf8d共享规则），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
