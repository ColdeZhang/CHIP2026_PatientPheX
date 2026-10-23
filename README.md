# CHIP2026 PatientPheX：从头研发

本仓库现以干净的 `main` 为后续实验起点。前任的代码、数据、本体、词典、预测和接手审计已完整移入 [历史归档](archive/README.md)。首日五条独立路线已完成，五份最终预测均获平台接收；五条实现和完整分支历史现已合并主分支，统一数据、评测和提交记录在主分支维护，见 [首日进展](docs/campaigns/day1-20261023.md)。

## 接手入口

1. 每次新任务先读 [HANDOFF.MD](HANDOFF.MD)，避免重复踩坑。
2. 再读 [WORKLOG.md](WORKLOG.md)，核对当前状态、实际工作树及下一步。
3. 按 [AGENTS.md](AGENTS.md) 执行；有效实验及时更新 [实验记录](docs/experiments/README.md)，并提交、推送和核验。

## 比赛目标与约束

比赛要求从生物医学全文中识别表型的位置、映射到 HPO，并判断各个指定患者的表型归属。

- 四项评分：实体 mention F1、文档概念 F1、患者关联 micro F1 和 macro F1，等权平均。
- 模型不超过 10B；使用 HPO v2026-06-23 的 HP:0000118 分支；禁止外部人工标注表型数据集。
- 截至 2026-10-23 为赛后评测阶段。完整规则、字段、赛程、榜单和官方原文已保存在 [比赛资料](docs/competition/README.md)。

## 当前目录

| 入口 | 用途 |
|---|---|
| `AGENTS.md`、`HANDOFF.MD`、`WORKLOG.md` | 执行规则、长期经验、当前台账 |
| [docs/competition/](docs/competition/README.md) | 官方资料和来源快照 |
| [docs/experiments/](docs/experiments/README.md) | 新实验索引与模板；旧诊断只留归档链接 |
| [docs/workflow/](docs/workflow/README.md) | 工作方法的依据与适配理由 |
| [archive/](archive/README.md) | 冻结的历史证据，不是新项目的代码或数据入口 |

共享评测与提交校验位于 `scripts/`；模型代码已集中到 `solutions/`；本轮本体规则、病例检索和 CRF 在三个新分支/worktree 中分别优化。本地分数为固定留出集近似诊断，不能当作官方 B 榜成绩。

## 后续起点

先确认真实目标、资源约束和成功标准，再建立可信的数据、评测与验证链路：核验官方数据来源及指定 HPO 版本，确定按文献分组的验证划分，处理评分定义和提交格式的边界情况，最后以最小可运行方案建立基线。

不默认复用归档中的词典、代码、划分或预测；如确需取用某项资源，独立记录来源和验证。历史结论不决定新技术路线。

新实验从最新 `origin/main` 开始，记录实际 commit、数据/配置版本和结果；大产物按 AGENTS 约定保存到仓库外。首日五份预测已完成提交；以 [提交凭据](docs/submissions/day1-20261023.json) 中的平台接收记录为准，五份官方分数已按后续指令回收，当前本体规则最高（0.603338），详见 [官方分析](docs/campaigns/day1-20261023-official-analysis.md)。

当前工作：[第二轮总记录](docs/campaigns/round2-20261023.md)。三个方案各自保存台账、实验、笔记和证据；主线程只汇总关键状态。本轮目标为每路线本地 Score ≥ 0.8，用户通知前不上传官方评测。
