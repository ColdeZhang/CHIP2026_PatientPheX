# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **固定dev最佳**：CPU四折选出的 k21/.35 为 **.625073738**（首日 .613517597）；独立重训/逐字节预测复现通过。后续GPU研究仅在train64内部fold0/fold1，尚未重评dev。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：整患者开放集合SFT完整47×2配对严重负：base grounded .375189、epoch002 grounded .396343，同实体规则 .711354；adapter 44合法中26空数组，3触顶，停止这套配置。原文quote存在不代表自由HPO ID正确。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：9B原58 base Score **.633859674**，同实体rules .711353522；fresh204×2epoch已完整52step、1855.10秒（30.92min）结束，唯一epoch002同58 adapter推断18:04:47 UTC启动，尚待质量结果。旧8B同A30全文控制 **.618120269→.610344254**，仅3额外TP却47额外FP，冻结负结果，不调提示。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. **正在运行：9B原58 adapter推断**。A800 `69b991ee-183f-4add-af1a-579af92479c9`、GPU2 `GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc`，作业 `proc_301c8be3-bba6-4c69-9634-bf4a0dbf752b`，18:04:47 UTC开始；本地session93066。外存 `qwen35-preparation/relation-quality-plan/adapter58-plan.json` / `adapter58-job.json` / `adapter58-monitor.json` 为恢复入口，完成自动回收及原解析/v2评分。禁止重复启动；原协议batch1/maxinput14000/maxnew1600/greedy/thinking=false，唯一epoch002。
2. **训练已结束**。fresh204行×2ep/52updates，原作业 `proc_c35c5440-2f32-4842-b4aa-8e444173673b` 与monitor89754已结束。实际1855.10秒，allocated37.48GB/reserved78.50GB分开记录，无OOM；epoch002 adapter SHA8e32daee…，两epoch及Adam/scheduler/RNG全实际SHA在外存 `relation-training-preparation/checkpoint-actual-hashes.json`。完整配置、命令与终点见[完成证据](evidence/round2/qwen35-relation-training-complete.json)，不挑epoch1质量，不宣称新的数值resume等价。
3. **GPU交接队列**。本批adapter58全量冻结并fresh核空后，明确交本体176条9B base短链接；待其明确归还后，本路线做一个最长16245token全文probe，再依据成本跑一次9B full58。旧原58 cap14000与16384在实际CPU的input_ids/attention_mask/生成参数全同（max6040、0CUDA），所以复用已冻结同A800 base58，不重复基线；证据见[cap等价](evidence/round2/full-context9b-cap-equivalence.json)。外存 `full-context9b-a800/probe-plan.json` 尚未启动，不得跳过交接。
4. **8B全文控制已结束且释放A30**。同A30两臂58/58、0触顶/删demo，原58 .618120269→全文 .610344254；947实体与所有候选不变。配对墙钟805.38秒；FP的‘不属任何指定患者gold概念’293→323，‘属其他指定患者’53→70，前者不能直接认定真实背景。双raw与双pred先冻结再评测，原解析回退2→7，未应用schema控制。外存 `full-context8b-a30/paired-summary.json` 及[结果](evidence/round2/full-context8b-paired-result.json)，独立终点审查待完成。
5. **模型持久副本复制仍进行中，CPU任务**。A800 SHM公开15文件allowlist源 `proc_4bf81387…`/端口41236 → .100下载 `proc_44414248…`，本地monitor96986，外存 `a30-fullcontext/`；约8.5MiB/s，保留Range状态不重启。全15SHA完成后必须停止自有源并核端口关闭。A30私有TF5已CPU通过，CUDA未初始化；只读旧runtime补regex、新私有safetensors .8，不改共享环境。复制只是持久备份，不在A30重复同9B质量任务。
6. **环境与来源**。A800只读Python `ontology-round2/qwen-venv/bin/python`，私有 `case-retrieval/qwen35-preparation/a800-migration/runtime` 提供TF5.20.2/PEFT0.21.2；模型 `/dev/shm/patientphex-qwen35-1045-c20223623576` 为已全SHA核验15文件的易失缓存，持久源在A100 `476f81cf-09e7-445b-a7f0-d0e12b12e50f:case-retrieval/qwen35-preparation/models/Qwen3.5-9B`。HF c202236…/MS aa2aaf…，LICENSE差异单独处理；A100 GPU等待计划已停止。
7. **完成原解析后才做CPU schema控制**。`normalize_pair_schema.py` 冻结e9867a0a，仅精确合法patient_id/evidence对象列表投影，两臂一同应用，原raw/原解析不覆盖；外存 `relation-quality-plan/run_schema_control.py` 要求adapter原协议先完整。未知CID/HPA不猜修，完整格式错误/回退纳入分母。关系误类另作事后聚合，不用于调提示。
8. **已结案**：整患者自由HPO47×2负在 `patient-set-fold0-bf16-v1/evaluation-summary.json`；抽取跨fold1七臂在 `sft-extract-fold1-bf16-v1/fold1-final-report.json`，均独立CPU重放通过。不继续无收益网格，Qwen3-8B旧模型/环境/checkpoint不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `9ff661e`（8B全文资源/配对预算；此前144325e正常merge main@512cf8d共享规则），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
