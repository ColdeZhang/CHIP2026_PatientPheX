# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **固定dev最佳**：CPU四折选出的 k21/.35 为 **.625073738**（首日 .613517597）；独立重训/逐字节预测复现通过。后续GPU研究仅在train64内部fold0/fold1，尚未重评dev。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：整患者开放集合SFT完整47×2配对严重负：base grounded .375189、epoch002 grounded .396343，同实体规则 .711354；adapter 44合法中26空数组，3触顶，停止这套配置。原文quote存在不代表自由HPO ID正确。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：上述负结果均已独立CPU重放通过；在新A100自有目录准备固定Qwen3.5-9B通用基座与隔离runtime。15文件已实际全SHA通过、隔离runtime与CPU接口已就绪。首次GPU前检发现A100 GPU3被他人占用，未启动任何新GPU任务。已获root分配A800自有SHM缓存，19.329GB内网复制及15文件实际SHA已完成，源服务已关闭，新隔离runtime已CPU通过。A800短smoke于17:10:28启动proc_d81cb8a8-82fe-461c-a570-854e6e06e763；同一冻结短smoke→最长6040输入探针→最长5787训练样本单步成本，均须前一步通过再执行。204正式训练未启动。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. 整患者47×2已完成，A800 GPU2已重新查询无进程并释放。外存`patient-set-fold0-bf16-v1/`的`training-plan.json`、`checkpoint-actual-hashes.json`、`inference-plan.json`、`evaluation-plan.json`、`evaluation-summary.json`是完整入口；base作业`proc_8eb8c4d2-940d-4f37-a376-7238b3625fb5`、adapter作业`proc_a5346f44-e7d1-4b09-bfee-841b6a30ec11`均结束，本地session96434也已退出，不要重启。
2. fold1抽取全链也已完成，.100 GPU1由研究子代理重新查询无进程并释放。外存`sft-extract-fold1-bf16-v1/fold1-final-report.json`、`linked-original/summary.json`绑定7臂预测、自己的train48重训/原88窗/唯一23子窗恢复/HPO56/冻结Sap来源。研究子代理已完成最长关系训练样本的完整r16单步成本入口与实际CPU审计，不管理运行中GPU。
3. 新模型准备：10.5.3.12 mount`476f81cf-09e7-445b-a7f0-d0e12b12e50f`，自有cwd`case-retrieval/qwen35-preparation`，唯一模型目标`models/Qwen3.5-9B`；下载作业`proc_b557721e-c2e8-4cb1-89d6-b432e3211eba`于15:45:37 UTC启动。固定HF revision`c202236235762e1c871ad0ccb60c8ee5ba337b9a`；完整来源在`qwen35-preparation/model-source-manifest.json`与仓库小摘要，16:22:53 UTC已完成15文件实际全SHA，marker SHA`b9c52e927851a157d49f895fb3c368704ee7b824026644b60028278e4aa802b8`。本地monitor session2154已退出；完整权重可用，不能重启下载。
4. A100系统glibc2.17，默认Python无torch；两套旧peft环境导入libcurand失败。已找到只读`/dataset/miniforge3/envs/vlm-det/bin/python`（Python3.10.21/torch2.6.0+cu124）CPU import成功，CUDA未初始化。私有`runtime/`安装TF5.20.2/PEFT0.21.2/Accelerate1.15.0/hfhub1.31/tokenizers.23.1，作业`proc_947a0f6b-0a9d-416b-989f-997ba971e357`；只写本目录，不修改共享/系统环境；实际Qwen3_5ForCausalLM/torch/PEFT CPU import全通过、CUDA未初始化。
5. 短smoke源码已冻结SHA615fd0c6…，真实CPU305行mask/EOS全过；最短非空366tokens（350prefix+16监督），9项不变量检查通过。外存`qwen35-preparation/gpu-smoke-plan.json`完整env/命令，状态prepared_not_started；本地driver13960因GPU前检占用退出，未创建GPU作业。先由root重新协调，确认空闲后再执行同plan；`smoke-preparation/implementation-review.json`绑定CPU证据。质量入口`qwen35_runner.py`另有5项恢复/身份隔离检查，未GPU执行。
6. 迁移恢复入口为外存`qwen35-preparation/a800-migration/`，source-server作业`proc_8408c689-2c42-454d-aaec-3de9951fb60d`仅公布15模型文件，transfer作业`proc_3472f814-4998-49d7-874e-905aea730145`向`/dev/shm/patientphex-qwen35-1045-c20223623576`做8线程Range/pwrite续传；monitor_transfer.py本地session84187已结束，2106.18秒完成15文件实际SHA后关闭自有server并核端口关闭。SHM是易失缓存，持久副本仍在A100，不能当远端备份。A800私有runtime为`case-retrieval/qwen35-preparation/a800-migration/runtime`，原qwen-venv Python3.11/torch2.7.1只读复用，TF5.20.2/PEFT0.21.2 CPU导入成功。sequence已16:52:41复查GPU2空闲并明确交回。本地driver session74031已在fresh GPU空闲核验后启动短smoke作业proc_d81cb8a8-82fe-461c-a570-854e6e06e763（17:10:28）；成功后顺序最长推断，失败即停。A800 gpu-smoke-plan.json已started，gpu-smoke-job.json为实际恢复入口；A100旧prepared plan始终未启动，不要混淆两个机器。
7. 完整r16成本入口`qwen35_relation_cost.py`只支持一个最长5787token样本，204实际数据mask/EOS/因果loss及248模块43,278,336参数的meta核验通过；不等于GPU可训练，且不支持正式204×2训练。外存`relation-cost-preparation/`计划须在短smoke和最长推理成功后才启动，helper已交只读审查。
8. 新A100 GPU3 UUID`GPU-f5417237-913f-06e6-77bc-00d0ce84b68b`原由root预留但16:23前检已有他人占用9110MiB；本路线尚未使用，不得强占或终止该进程。A800/.100已释放，.3 GPU0由本体持有；任何历史running字段只是启动快照。原Qwen3-8B/runtime/两个SFTcheckpoint继续保留，不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `498988f`（完整r16单步成本入口、CPU证据与A800迁移恢复；此前144325e正常merge main@512cf8d共享规则），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
