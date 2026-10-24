# CRF 路线工作台账

分支 `round2/sequence-crf`，worktree `/workspace/patientphex-worktrees/round2-sequence-crf`；负责人只修改 `solutions/sequence_crf/`。新任务先读当前 `main:HANDOFF.MD`、`main:AGENTS.md` 与主线程总账，再读本页及对应实验文档。完整旧台账已移至[历史快照](WORKLOG-HISTORY-20261024-0335.md)，其中旧“运行中”字样不表示当前状态。

## 目标与已知最佳

- 用户验收仍是原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text、Score ≥ .8 且格式/复现通过；内层F0/F1的诊断不能替代该目标。当前不上传官方，不自动迁移原dev。
- 每个独立模型 ≤10B，允许多模型；不要重新按全链参数合计禁用集成（main HANDOFF H47）。
- 本路线已知的跨方案原dev候选 **.746081965**：固定逆序关系并集，1072实体不变，原368关系保留、新391关系；+.007588212、15TP8FP，已独审PASS，仍未.8。[结果与恢复](experiments/reverse-json-order-dev.md)。更晚的跨路线最佳以主线程总账为准。
- 本路线独立原政策 **.711324992**（独审PASS），旧CRF/Sap、GLiNER及HGB实体控制详情均在历史与各实验文档中。[当前资源清单](experiments/best-pipeline-resource-inventory.md)、[已存HGB推断入口](experiments/load-existing-selector.md)。

## 当前任务：selected-CE fresh full64 正式训练中

**当前唯一远端job** `proc_7f51657c-95c7-4118-9d9d-68377242d294`，A100 GPU2 UUID2c22，driver141616/child141633；05:13:13.317Z启动4500+TERM30，近似TERM06:28:13/KILL06:28:43，以remote driver-started为准。环境重连后05:18:35实际核两PID仍存在、GPU used28305MiB/util68%；05:17:47最后打印epoch1 step5、loss .302332392、峰allocated23.892GB。还没有完整epoch或新质量，不从观察连接状态推断终态。

原本地watcher108070已不存在。新只读watcher独立namespace接管，入口F/watch_reconnect_0518.py、每180s只读原状态、原KILL+300后止，不start/修改远端；F=`/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-memory-recovery/formal`。新会话先检查该watcher实际PID，若断开仅恢复监控，不能重复launch。实际恢复证据见[小收据](evidence/patient-first-full64-memory-recovery/reconnect-0518-status.json)，完整操作见[恢复入口](patient_first_full64/memory_recovery/OPERATIONS.md)。

**已核依据**：旧proc5b00首epoch backward OOM、无完整状态，独审f7668be7；不能exact续训。新恢复只将有监督位置先选后FP32 CE，完整BF16 head/每rowmean/有效group/精度不变。目标Torch288例通过；新probe12次前后向/1次完整group更新，已有496 Adam状态与累积梯度下最长6166通过，147.295秒rc0、峰allocated25.031GB/reserved29.977GB，05:11:10实际释放并弃权重。统一独审b46983ef通过，正式fresh已获root一次统一授权。探针通过不保证完整训练稳定，原失败记录不改写。

- root已授权完整链：每epoch持久fullstate → 真实epoch2三SHA绑定 → 原dev纯64一次900+30 → raw和a179并集完整freeze → 唯一v2；不评patient-alone，不重算旧候选，不删旧1072实体/391关系/35public-new。Score严格升记候选best；macro降低如实披露；.8加独审才达标。
- [full64运行/恢复记录](experiments/patient-first-full64.md)。最长6166实际一步81.873秒PASS/卡已释放；正式CPU10.454秒PASS。未知上传经库存确认未落，3小片补齐同源后再启动。源码与job已推送；当前未新devgold/未质量/未官方。

**F0恢复58完整结束，原strict失败保留；primary负、oldSFT48∪patient增量通过事前判据。** 新primary .684342231，旧baseline .695443948；固定secondary **.705527893（+.010083945）**，关系micro+.020643194、macro+.019692587，新增26=16TP10FP且保留全部旧关系。947实体/非关联不变。仍低同池rules .711353522；原dev .8目标未完成，不自动迁dev。

- 原两轮52calls训练成功/完整state已持久；原strict推断第24行格式失败、无分数，23raw保存。独立恢复23精确缓存＋35新生成，于04:22:08 rc0/240.178秒完整终止，GPU2实际0MiB/无进程，该恢复协议无后继；当前新full64 probe另列。
- 完整58中57complete/1partial/0整row弃答；新3563609:relations:8只缺O7P，弃8候选对，53tokens/真实EOS248046。这是新生成实际内容，不能冒称找回旧失败raw。162小件逐SHA、全部ID独立decode/codec一致；两fullpred共冻be17f7b4后才两次v2，旧metric不重评。
- [完整记录](experiments/patient-first-F0.md)、[结果](evidence/round2/patient-first-F0-recovery-results.json)、[终点清单](evidence/round2/patient-first-F0-recovery-terminal-manifest.json)。外部P=`/workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation`，B=P/abstain-recovery。source/完整raw/state/原失败均冻结，禁止重启已完作业或重评。统一终点独审22325a29 PASS；未重训练、parser测试或评测。
- root随后授权的事后提出、事前固定CPU控制已通过：F1/F0 literal union(oldSFT,reverse,patient)，两折预测共冻后各唯一v2；门槛两折Score均严格高于各自old∪reverse且macro均不降。无扫描/第四臂，总.479秒。F1 .663841821（+.001369374，micro降/macro升）、F0 .709556743（+.004869303），仍低F0 rules；独审3444fe0a PASS。[结案](experiments/relation-triple-union.md)。

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
- 截至本页整理前，已push并lsremote确认 `beb362fa55ddb11dd1ee4e12b2eb78833ec5564d`，含原失败、新工程完整终点、fresh正式job及后继单一dev入口。不要为记录台账自身SHA递归提交。
- 原full64已OOM停止，新selected-CE fresh正式训练正在运行；下一里程碑为两完整epoch持久终点，再按持续授权执行原dev64/唯一a179并集。缺少真实epoch2时推断入口仍不可执行。逐关键阶段更新本页、实验记录、commit/push。未达原dev .8的实现不合并main，不宣称比赛目标完成。

- full64 新CPU实际通过：265条/860405 token，最长6166/122监督；已冻结单步240+30新源与6输入，native265实际8.432秒通过，唯一最长probe81.873秒rc0、6166完整一步PASS，峰allocated38.184GB/reserved39.083GB，结束free2.666GB；04:34:44 GPU2实际核空，root据探针已统一授权正式4500+30与原dev64另900+30，实际formalCPU10.454秒通过，正式唯一proc_5b00069a/PID120406于04:42:33.455Z启动，04:51:51 OOM/rc1退出；原4500+30未耗尽。见 [full64记录](experiments/patient-first-full64.md)。

- 新正式fresh已05:13:13Z启动 `proc_7f51657c` / driver141616，4500+30，近似TERM06:28:13/KILL06:28:43（remote driver为准）；已实际部署22件SHA/compile/fsync，持久盘free57.076GB；保持265×2/68calls/4warmup、4500+30及原dev64另900+30，原a179单臂不变。新probe通过但不保证完整两轮；[entry](evidence/patient-first-full64-memory-recovery/formal-entry-readiness.json)已获root一次统一授权，**新fresh正式运行中**；旧OOM与新probe状态独立。恢复入口见[操作说明](patient_first_full64/memory_recovery/OPERATIONS.md)。
