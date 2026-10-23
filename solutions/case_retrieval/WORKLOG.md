# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **固定dev最佳**：CPU四折选出的 k21/.35 为 **.625073738**（首日 .613517597）；独立重训/逐字节预测复现通过。后续GPU研究仅在train64内部fold0/fold1，尚未重评dev。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：整患者开放集合SFT完整47×2配对严重负：base grounded .375189、epoch002 grounded .396343，同实体规则 .711354；adapter 44合法中26空数组，3触顶，停止这套配置。原文quote存在不代表自由HPO ID正确。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：Qwen3.5-9B原58 base完整Score **.633859674**（同实体8B base .623068497，rules .711353522），58严格JSON/0触顶、397/415候选合法，18按原LR回退。正式fresh204行×2epoch已17:33:24 UTC启动A800，单作业硬预算90分钟；固定同58配对、原候选/标注/decoder不改，原dev不动。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. **正在运行**：A800 `69b991ee-183f-4add-af1a-579af92479c9` 的GPU2 `GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc`，fresh固定204×2epoch作业 `proc_c35c5440-2f32-4842-b4aa-8e444173673b` 于17:33:24 UTC启动。真实初检盘4,277,125,120 bytes，预计两epoch checkpoint约1.04GB；命令由`timeout 5400`约束，60秒TERM宽限。外存 `qwen35-preparation/relation-training-preparation/formal-plan.json` / `formal-job.json` / `formal-monitor.json` 为恢复入口；本地monitor session89754，不要重启重复训练。源码9d2e0ee/commit46bdcc4，真正CPU与独审通过；fresh基座不继承smoke/cost adapter。
2. **下一步**：等正式epoch002完成及实际adapter/state SHA回收，再同58配对（batch1/maxinput14000/maxnew1600/greedy/thinking=false），用冻结原merge/v2/格式检查。唯一预定终点epoch002；52updates、3warmup、b1acc8、lr1e-4、r16/alpha32/drop.05、seed20261023。base58原作业已结束、raw SHA42223211…、pred9165275c…在外存 `qwen35-preparation/relation-quality-plan/`。collector曾误断原输入顺序；冻结runner按token长度排序，已核完整身份映射，未改raw/重新推理，此错误见`base58-recovery.json`。
3. **实际GPU证据**：短smoke `proc_5720de34-27af-45a5-aee7-e33ee6432ecf` 32.34秒、finite/nonzero梯度、adapter保存/磁盘重载全tensor相同；forced im_end只证明停止引擎，不是自然生成质量。最长输入 `proc_378d31c1-4b25-4195-bd62-c6e7de1f8451` 6040→212tokens，7.767秒生成，8/8候选合法，未评分。最长训练单步 `proc_85141604-a694-455d-9fd8-1374b70f3648` 5787tokens、loss .380170、三家族梯度/B变化通过，36.95GB allocated/42.17GB reserved；单步10.810秒，token线性估2轮40.68分钟、408×最长保守规划73.51分钟，均非实测总训练时长。证据见 `qwen35-gpu-interface-and-cost.json`。
4. **模型和环境**：A800只读旧Python `ontology-round2/qwen-venv/bin/python`，本路线私有 `case-retrieval/qwen35-preparation/a800-migration/runtime` 提供TF5.20.2/PEFT0.21.2，不能用旧TF4.51。模型 `/dev/shm/patientphex-qwen35-1045-c20223623576` 为已全SHA核验15文件的易失缓存；持久源在A100 `476f81cf-09e7-445b-a7f0-d0e12b12e50f:case-retrieval/qwen35-preparation/models/Qwen3.5-9B`。HF c202236…/MS aa2aaf…、实际文件SHA在来源manifest；LICENSE差异已独立处理。迁移源HTTP已关闭，A100 GPU等待/自动启动全部结束，禁止重启重复任务。
5. **失败保留**：A800短smoke v1 `proc_d81cb8a8-82fe-461c-a570-854e6e06e763` 仅在loading-info set写JSON处失败、未forward；最小序列化修复b52337c后v2通过。原v1原目录保留，不能写成模型训练失败。旧A100 prepared plan从未跑GPU。
6. **已结案**：整患者开放HPO集合47×2负结果在 `patient-set-fold0-bf16-v1/evaluation-summary.json`，固定抽取跨fold1七臂在 `sft-extract-fold1-bf16-v1/fold1-final-report.json`，均独立CPU重放通过。只保留参考，不继续无收益网格；原Qwen3-8B/环境/checkpoints不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `46bdcc4`（9B正式训练入口与实际CPU审查；此前144325e正常merge main@512cf8d共享规则），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
