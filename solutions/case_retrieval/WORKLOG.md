# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **固定dev最佳**：CPU四折选出的 k21/.35 为 **.625073738**（首日 .613517597）；独立重训/逐字节预测复现通过。后续GPU研究仅在train64内部fold0/fold1，尚未重评dev。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：整患者开放集合SFT完整47×2配对严重负：base grounded .375189、epoch002 grounded .396343，同实体规则 .711354；adapter 44合法中26空数组，3触顶，停止这套配置。原文quote存在不代表自由HPO ID正确。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：完成上述负结果独审/诊断；在新A100自有目录准备固定Qwen3.5-9B通用基座与隔离runtime。只下载/CPU兼容性检查，未开始新GPU训练或推理；后续最小接口smoke须重新核GPU与主线程排期。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. 整患者47×2已完成，A800 GPU2已重新查询无进程并释放。外存`patient-set-fold0-bf16-v1/`的`training-plan.json`、`checkpoint-actual-hashes.json`、`inference-plan.json`、`evaluation-plan.json`、`evaluation-summary.json`是完整入口；base作业`proc_8eb8c4d2-940d-4f37-a376-7238b3625fb5`、adapter作业`proc_a5346f44-e7d1-4b09-bfee-841b6a30ec11`均结束，本地session96434也已退出，不要重启。
2. fold1抽取全链也已完成，.100 GPU1由研究子代理重新查询无进程并释放。外存`sft-extract-fold1-bf16-v1/fold1-final-report.json`、`linked-original/summary.json`绑定7臂预测、自己的train48重训/原88窗/唯一23子窗恢复/HPO56/冻结Sap来源。研究子代理当前仅CPU做整患者quote/ID一致性诊断，不管理运行中GPU。
3. 新模型准备：10.5.3.12 mount`476f81cf-09e7-445b-a7f0-d0e12b12e50f`，自有cwd`case-retrieval/qwen35-preparation`，唯一模型目标`models/Qwen3.5-9B`；下载作业`proc_b557721e-c2e8-4cb1-89d6-b432e3211eba`于15:45:37 UTC启动。固定HF revision`c202236235762e1c871ad0ccb60c8ee5ba337b9a`；完整来源在`qwen35-preparation/model-source-manifest.json`与仓库小摘要，下载结束必须检查`download-complete.json`全SHA，不能用partial。
4. A100系统glibc2.17，默认Python无torch；两套旧peft环境导入libcurand失败。已找到只读`/dataset/miniforge3/envs/vlm-det/bin/python`（Python3.10.21/torch2.6.0+cu124）CPU import成功，CUDA未初始化。私有`runtime/`安装TF5.20.2/PEFT0.21.2/Accelerate1.15.0/hfhub1.31/tokenizers.23.1，作业`proc_947a0f6b-0a9d-416b-989f-997ba971e357`；只写本目录，不修改共享/系统环境，安装完成还需CPU实际import。
5. 新A100 GPU3 UUID`GPU-f5417237-913f-06e6-77bc-00d0ce84b68b`由root预留，尚未使用，开跑前再核。A800/.100已释放，.3 GPU0由本体持有；任何历史running字段只是启动快照。原Qwen3-8B/runtime/两个SFTcheckpoint继续保留，不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `01b2e72`（含整患者完整47配对前冻结；此前正常merge共享main@811b8cc，方案/评测源码SHA不变），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
