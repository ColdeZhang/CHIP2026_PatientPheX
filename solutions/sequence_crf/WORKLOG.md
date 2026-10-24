# CRF 路线工作台账

分支 `round2/sequence-crf`，worktree `/workspace/patientphex-worktrees/round2-sequence-crf`；负责人只修改 `solutions/sequence_crf/`。新任务先读当前 `main:HANDOFF.MD`、`main:AGENTS.md` 与主线程总账，再读本页及对应实验文档。完整旧台账已移至[历史快照](WORKLOG-HISTORY-20261024-0335.md)，其中旧“运行中”字样不表示当前状态。

## 目标与已知最佳

- 用户验收仍是原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text、Score ≥ .8 且格式/复现通过；内层F0/F1的诊断不能替代该目标。当前不上传官方，不自动迁移原dev。
- 每个独立模型 ≤10B，允许多模型；不要重新按全链参数合计禁用集成（main HANDOFF H47）。
- 本路线已知的跨方案原dev候选 **.746081965**：固定逆序关系并集，1072实体不变，原368关系保留、新391关系；+.007588212、15TP8FP，已独审PASS，仍未.8。[结果与恢复](experiments/reverse-json-order-dev.md)。更晚的跨路线最佳以主线程总账为准。
- 本路线独立原政策 **.711324992**（独审PASS），旧CRF/Sap、GLiNER及HGB实体控制详情均在历史与各实验文档中。[当前资源清单](experiments/best-pipeline-resource-inventory.md)、[已存HGB推断入口](experiments/load-existing-selector.md)。

## 当前唯一GPU作业：patient-first F0 原58推断

**两轮52calls已完整训练并持久化；固定原58推断已启动，尚无F0患者优先质量结果。** 严格train48/204条、原F0纯输入58；fresh公开9B、r16、同优化器/seed、连续2epoch/52calls/3warmup。唯一变化是PID→CID集合监督；原binary目标可双逆还原，但token级损失权重随因子分解变化，不能称数值目标相同。

| 项目 | 当前事实 |
|---|---|
| 当前作业 | `proc_22400ec4-e939-41a7-988e-bef7c87ddc23`，driver PID93337；训练proc_ced868e2已rc0 |
| 设备 | A100 GPU2，UUID `GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1` |
| 推断开始/截止 | 2026-10-24 04:08:22.106526 UTC；approx TERM **04:18:22.107**，KILL **04:18:52.107**，以driver-started为准 |
| 真实训练终点 | 04:04:29 rc0，native2279.145秒/wrapper2280.456秒；两轮52calls，epoch2 train loss .1009212；峰allocated37.416GB/reserved40.802GB |
| 保存 | A100项目持久盘；每epoch adapter+optimizer/scheduler+全部RNG，file/dir及新父链fsync、原子rename、重开SHA、独立durability收据 |
| 固定源/计划 | trainer `cc707253`，train-plan `4b70dbce`；完整入口独审 `bcaf14d3` PASS |

新最长5769 token单步已真实PASS，13.936秒step、81.688秒wrapper；峰allocated36.894GB/reserved41.748GB。**仅单步证据，不保证accum8/常驻optimizer/多形状全程稳定**。正式预算3900+TERM30包含保存，首OOM/数值/截止失败停，不改loss/精度/allocator/长度、不自动重启。

两个epoch全部5成员各519,866,582B，已file/dir/祖先链fsync及重新开SHA；final adapter51f8c7a4、state9932ff4d。16件小终点226,862B已回收逐SHA核验（collection b9693ce9）。04:06正常监控短暂LUNA_OFFLINE，≥60秒后原只读恢复；未重启。训练释放不代表本camp移交卡，现执行固定原58推断（另600+30）；primary和预设oldSFT48∪new保持947实体/415CID/1166对，两完整预测共同freeze后各唯一v2，旧baseline .6954439475814587不重评。保留条件是delta(micro+macro)>0且macro不降；不得新增8B/其他规则组合或据分数改政策。

- [完整方法、失败恢复和证据链](experiments/patient-first-F0.md)；[入口终审](evidence/round2/patient-first-F0-formal-entry-review.json)、[真实job](evidence/round2/patient-first-F0-formal-train-job.json)。
- 外部P=`/workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation`；P内 `resource.json`、`train-job.json`、`train-launch-intent.json` 和 `RECOVERY.md` 是恢复入口。A100自有目录 `sequence-round2/patient-first-F0-formal-v1/`，不要重调train start。
- 源入口 `patient_first_f0/formal/formal_operations.py`：先 `collect --phase train --directory P`；只有rc0/52calls/两完整durability收据后，fresh卡启动唯一infer；完整collect infer后依次 `finish.py assemble`、`secondary_union.py assemble-common`、`secondary_union.py evaluate-both`。缺件/格式失败不得评分。
- MCP UNKNOWN/OFFLINE只表示传输未知；保留原intent，至少60秒后一次原目录只读库存，不能当作模型失败或重复启动。无其他本路线GPU后继；A800已交检索。

## 已闭环的最近关键决策

| 控制 | 结论与下一步 |
|---|---|
| patient-first F1 | 单独.645531861低旧.660942003；预设并集.664421522，+.003479519，micro微降/macro升，新增15TP32FP。小幅互补促成本次固定F0复验，未达.8。[完整记录](experiments/patient-first-closed-set.md) |
| 旧8B∪旧9B F0 | 唯一CPU控制.695443948→.688565023，micro/macro双负，新增54=16TP38FP；旧8B1处LR回退披露。独审c12e3ae5 PASS，负关闭，不F1/dev，不改正在运行的患者F0。[记录](experiments/old-sft-union-f0.md) |
| 全局pair-conditioned BERT | 双fresh8ep：uniform.542016、pair.557825，均低同池SFT/rules；完整负关闭。[记录](experiments/pair-conditioned-document.md) |
| Thinking人工边界 | 单条格式/成本PASS，仅可用性，无质量外推。完整80固定2304/512政策因约3小时成本已明确暂缓，不降cap或自行启动。[固定政策](experiments/thinking-budget-80-plan.md) |

其他已完成的实体、共指、检索、GLiNER、公开模型及正负控制见[历史快照](WORKLOG-HISTORY-20261024-0335.md)和[证据索引](evidence/round2/README.md)，不重跑已审合同。

## 记录与备份

- 原实验索引：[E-20261023-009](experiments/E-20261023-009.md)、[研究笔记](NOTES.md)。实验材料根为 `/workspace/patientphex-runs/round2-20261023/`；权重、全pred/raw、完整日志留仓库外。
- 截至本页整理前，已push并lsremote确认 `8946555a80c6c5711449b8460a4fabf9ad0942e8`，包含首轮持久收据；本次完整训练终点即刻备份。不要为记录台账自身SHA递归提交。
- 下一里程碑是原58完整推断及唯一双质量终点；逐关键阶段更新本页、实验记录、commit/push。未达原dev .8的实现不合并main，不宣称比赛目标完成。
