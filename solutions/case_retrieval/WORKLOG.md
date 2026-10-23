# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **固定dev最佳**：CPU四折选出的 k21/.35 为 **.625073738**（首日 .613517597）；独立重训/逐字节预测复现通过。后续GPU研究均在train64内部fold0，尚未重评dev。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **当前任务**：优先整患者开放集合SFT的带原文证据版本。研究代理准备新schema/合法训练目标/strict merge及实际token审计；审核后A800最长样本smoke，再从原base正式2epoch，不从smoke或旧adapter续训。尚未启动此项GPU训练。
- **后续**：抽取fold1仅CPU准备（305训练窗、88验证输入窗），等待新任务运行与root资源安排；只允许相同2epoch/既有阈值做一个固定内折复验，不扫参数、不提前评dev。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. 整患者新任务：A800 `10.5.4.7` mount `69b991ee-183f-4add-af1a-579af92479c9`，GPU2 `GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc`。自有cwd `case-retrieval/round2-patient-set/` 已上传冻结trainer/runner，未有此任务GPU进程；卡暂可由sequence共指配对短批使用，启动smoke前与其/root交接并重新核空。
2. A800解释器只读 `/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen-venv/bin/python`；base只读同目录`pretrained-qwen3-8b`；PEFT只读`PYTHONPATH=/home/zhangyuheng/projects/CHIP2026_PatientPheX/sequence-round2/peft-runtime`。CPU核验torch2.7.1cu126/transformers4.51.3/PEFT.15.2通过；不修改共享环境。
3. 已完成抽取SFT：.100 mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，远端`case-retrieval/round2/sft-extract-fold0-bf16-v1/`，保留epoch001/002及optimizer状态；原推断`inference/`与唯一恢复`inference/retry/`均已结束。实际作业/配置/全SHA在外存同名目录。当前.100 GPU1由sequence共指使用，本路线没有运行作业。
4. 完整HPO作业`proc_855aacdd-70a5-45bb-b204-80b6d205a02f`已完成并释放A800 GPU2，外存`hpo-inference-plan.json`明确completed；响应/manifest/检查点全部回收。CPU重装命令在`sft-extract-fold0-bf16-v1/linked-original/commands.json`与`sap-filter-control/repaired101/merge-evaluation-commands.json`。
5. .3 GPU0现在由本体真实PhenoTagger训练使用，不能启动原预想的fold1抽取。全部GPU都须重新确认资源归属和空闲进程；任何旧启动凭据中的running字段只是当时快照。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近已核验远端检查点 `dfbf92b`（含`470a47c`完整控制与新方法研究），本台账允许滞后自身提交；实际新增变更需按关键阶段继续commit/push。大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。
