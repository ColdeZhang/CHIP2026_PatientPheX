# CRF 路线工作台账

分支 `round2/sequence-crf`，worktree `/workspace/patientphex-worktrees/round2-sequence-crf`；负责人只修改 `solutions/sequence_crf/`。新任务先读当前 `main:HANDOFF.MD`、`main:AGENTS.md` 与主线程总账，再读本页及对应实验文档。完整旧台账已移至[历史快照](WORKLOG-HISTORY-20261024-0335.md)，其中旧“运行中”字样不表示当前状态。

## 目标与已知最佳

- 用户验收仍是原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text、Score ≥ .8 且格式/复现通过；内层F0/F1的诊断不能替代该目标。当前不上传官方，不自动迁移原dev。
- 每个独立模型 ≤10B，允许多模型；不要重新按全链参数合计禁用集成（main HANDOFF H47）。
- 本路线已知的跨方案原dev候选 **.746081965**：固定逆序关系并集，1072实体不变，原368关系保留、新391关系；+.007588212、15TP8FP，已独审PASS，仍未.8。[结果与恢复](experiments/reverse-json-order-dev.md)。更晚的跨路线最佳以主线程总账为准。
- 本路线独立原政策 **.711324992**（独审PASS），旧CRF/Sap、GLiNER及HGB实体控制详情均在历史与各实验文档中。[当前资源清单](experiments/best-pipeline-resource-inventory.md)、[已存HGB推断入口](experiments/load-existing-selector.md)。

## 当前任务：patient-first F0 格式失败后的固定恢复准备

**两轮52calls完整持久训练成功；原strict58在第24行格式失败停止，尚无新F0分数。** 原推断proc_22400ec4于04:10:32.994 rc1/130.785秒退出，23条合法raw已存；04:10:33.116 GPU2实际0MiB/无进程。第24输入3563609:relations:8（3073token、9患者/8CID），顶层patient-key集合不符合strict。旧runner先parse再write，丢失该条raw/IDs；不能猜具体缺键或重新生成后冒称原件。

主线程已授权独立后继：精确复用23条缓存，其余35（含丢失失败条）新生成，同durable epoch2/greedy/输入/1600cap，独立600+TERM30。先逐行原子持久raw/IDs/终止信息，再有限解析；合法PID子集的缺键为弃答，未知/重复key或非法CID等整行弃答，不把补空列表伪称模型答案。完整58各有已存响应/失败后，primary/old∪new共同freeze再各唯一v2，旧指标复用，原strict失败保持独立。不重训/自动改变数值或增加其他组合。新源/实际CPU已通过，恢复job proc_d19bf1ac-96f4-49a4-b65c-56aec88322c2/PID100444于04:18:08.457145Z已启动，approx TERM04:28:08.457/KILL04:28:38.457（driver为准）；不重复start。

- 训练rc0/native2279.145秒，两epoch各519,866,582B完整state，真实file/dir/祖先链fsync/重开SHA；final adapter51f8c7a4/state9932ff4d。原16件collection b9693ce9；新推断7件回收精确。
- [完整方法和原失败](experiments/patient-first-F0.md)、[严格失败诊断](evidence/round2/patient-first-F0-strict-failure-diagnostic.json)；训练入口bcaf14d3保持，恢复入口另立。
- 外部P=`/workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation`；原`training-collection.json`、`inference-collection.json`、`strict-failure/diagnostic.json`和两个terminal目录不可覆盖。旧finish/commonfreeze/evaluate均未运行，不能对23条部分评分。新目录`abstain-recovery/`，MCP UNKNOWN只查既有intent，不重复未知start。
- F0固定基线old48 .6954439475814587；保留规则仍delta(micro+macro)>0且macro不降。原dev验收不变，未授权官方上传或自行迁dev。

## 已闭环的最近关键决策

| 控制 | 结论与下一步 |
|---|---|
| patient-first F1 | 单独.645531861低旧.660942003；预设并集.664421522，+.003479519，micro微降/macro升，新增15TP32FP。小幅互补促成本次固定F0复验，未达.8。[完整记录](experiments/patient-first-closed-set.md) |
| 固定5维LR F1→F0 | 1次fit/19iter/6 scalars，.697656525低old∪reverse .704687440与rules .711353522，macro也低，按事前强基线门槛负关闭；F1标签实读用于meta训练，非pure或OOF元验证。loader首次失败在gold前，独立恢复唯一v2，无再fit。[结果](research_readonly/relation-stacking-five-feature-control/README.md) |
| 旧8B∪旧9B F0 | 唯一CPU控制.695443948→.688565023，micro/macro双负，新增54=16TP38FP；旧8B1处LR回退披露。独审c12e3ae5 PASS，负关闭，不F1/dev，不改正在运行的患者F0。[记录](experiments/old-sft-union-f0.md) |
| 全局pair-conditioned BERT | 双fresh8ep：uniform.542016、pair.557825，均低同池SFT/rules；完整负关闭。[记录](experiments/pair-conditioned-document.md) |
| Thinking人工边界 | 单条格式/成本PASS，仅可用性，无质量外推。完整80固定2304/512政策因约3小时成本已明确暂缓，不降cap或自行启动。[固定政策](experiments/thinking-budget-80-plan.md) |

其他已完成的实体、共指、检索、GLiNER、公开模型及正负控制见[历史快照](WORKLOG-HISTORY-20261024-0335.md)和[证据索引](evidence/round2/README.md)，不重跑已审合同。

## 记录与备份

- 原实验索引：[E-20261023-009](experiments/E-20261023-009.md)、[研究笔记](NOTES.md)。实验材料根为 `/workspace/patientphex-runs/round2-20261023/`；权重、全pred/raw、完整日志留仓库外。
- 截至本页整理前，已push并lsremote确认 `8946555a80c6c5711449b8460a4fabf9ad0942e8`，包含首轮持久收据；本次完整训练终点即刻备份。不要为记录台账自身SHA递归提交。
- 下一里程碑是原23+新35完整推断及唯一双质量终点；逐关键阶段更新本页、实验记录、commit/push。未达原dev .8的实现不合并main，不宣称比赛目标完成。
