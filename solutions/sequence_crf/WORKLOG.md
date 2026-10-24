# CRF 路线工作台账

分支 `round2/sequence-crf`，worktree `/workspace/patientphex-worktrees/round2-sequence-crf`。负责人只修改 `solutions/sequence_crf/`；新任务先读当前 main 的 AGENTS/HANDOFF.MD/WORKLOG，再核本页与真实文件。完整历史见 [08:14快照](WORKLOG-HISTORY-20261024-0814.md)和[03:35快照](WORKLOG-HISTORY-20261024-0335.md)，其中“运行中”只代表当时状态。

## 当前任务：患者均衡混合损失 α=.5，CPU 合同

- **目标/边界**：在固定总 Score 目标下测试跨患者归一权重，与已负关闭的 value×2 不同。root已授权CPU原件完整证实后唯一240+TERM30工程probe，顺序在H5正式实际释放后fresh核GPU2；8微批非零Adam→3累积→最长5556，状态弃掉、不评分。正式训练/推断预算仍待probe成本，不新fit或质量。当前原dev best仍 `.754361398199`，未达 .8。
- **固定机制**：原 train32 patient-first 124行/32文献/87患者、原两轮顺序；保留一半原 token CE，新增一半患者及患者内 chunk 等权 list-value CE。每epoch P87/S16；row内新系数含真实group size，保留外层 `/len(group)`，包括尾4。原格式/key/EOS/换行由旧半项保留；不按 prepared 正例数加权，prepared空不等于真实gold空。
- **训练侧准备已完整通过**：唯一2.716751秒，manifest `c713111b…`；原全部124行IDs/labels/rendered/audit exact，400614 tokens/4694监督/max5556。精确 Fraction 证明两轮每患者新增份额1/87、混合1/174，跨两个患者value token为0，106处空列表保留。此系数不等于实际梯度、Adam更新或质量因果。[协议/源](patient_balanced_loss/README.md)、[实际清单](patient_balanced_loss/preparation-actual.json)。
- **数值合同完整通过并独审**：原唯一A100 Torch2.6.0+cu124、60+TERM5/CPU隐藏，worker5.207856/driver5.690608秒。8原件全SHA/fsync、317观察全PASS，source141c/plan2145；完整124系数桥接1/87→1/174、FP32/BF16、ignored NaN/Inf及g4/g8均通过。独审8765aeb8/10376B已PASS；FP32 loss最大绝对误差4.768e−7/grad3.388e−9，BF16梯度对已舍入参考max0，不声称所有loss位精确。[单份终态](patient_balanced_loss/CPU-terminal-summary.json)。原start UNKNOWN与收件UNKNOWN/LUNA只影响传输，保留原收据；无重执行。
- **新工程入口**：12行仅一次native编码0.693748秒，44992tokens/530监督，原IDs/labels/rendered及整rowmetadata exact，payload4cfdb1f1；顺序[62,108,84,87,63,93,106,78,65,85,80,6]。GPU源6f0188d4/hook97a0c5ce/driverafa8c149及CPU输入静态核无阻断。H5 proc431edce8已08:21:27运行，原600窗至08:31:27；新probe包94176B/b0bd5acf与14成员已CPU部署SHA/compile/fsync/重开通过，无UNKNOWN，无GPU启动。H5原08:24:13释放后，root现优先其新H5格式guard机械恢复至原08:31:27窗；probe没有start-intent/job，旧release快照仅历史，不用作当前启动gate。等恢复新实际release后fresh核唯一probe；[部署与恢复](patient_balanced_loss/probe-deployment.json)。
- **运行位置**：外部 `R/research/patient-balanced-mixed-loss-alpha05/{preparation,cpu-numerical-v1}`；R=`/workspace/patientphex-runs/round2-20261023`。远端 mount476f / project `sequence-round2/patient-balanced-alpha05/cpu-contract-v1/actual`。模型未载入；原数值任务已终态；新GPU2排在H5后，尚未启动。
- **下一步**：完整CPU已收回并统一独审；现源计划push → 私有包CPU部署 → H5实际释放/fresh卡核 → 原240+TERM30唯一probe → 原结果全SHA收回。后继 [trainer接线设计](patient_balanced_loss/FUTURE-INTEGRATION.md)已落实为probe专用hook；尚无正式训练入口。root已给上述唯一probe预算，须CPU全件到齐、H5实际释放、fresh卡核和源计划push后执行。候选保留总Score为准、macro完整披露，不恢复旧macro硬门。

## 当前最佳及官方授权

原dev当前跨方案已接受最佳 **9016863c / Score .7543613981990362**：1061实体/36患者/416关系；3049支持gate完整基底加765−a179精确6条患者关系。men `.792801107522`、doc `.828677839851`、micro `.738780207135`、macro `.657186438289`。相对3049总分仅+.000144428917，macro−.001238412671，如实保留；未达 .8，实验实现未合并main。[唯一合成终点](evidence/support-patient-synthesis/terminal-summary.json)、[源码/恢复](support_patient_synthesis/README.md)。统一审76ada874已PASS。

用户一次性授权已经完成：root **07:54:13 UTC**只上传901真实B100预测 `50ef5daa…/894806B`，平台accepted、额度5→4，文件名 `1792828438709_round2_20261024_best901_fixed64_50ef5daa212f.jsonl`。不查询官方分数，其余上传暂停。gate `R/official-highest-901-B-20261024/submission/postaccept-release.json` /919362af。此授权不能再次使用。

每个独立模型≤10B，允许多模型，不按整条流水线参数相加否定（main H47）。验收保持原 train64/dev16、HPO2026-06-23、公共v2默认all/text，Score≥.8且格式/复现通过。F0/F1辅助控制不能替代原dev目标。

## 已完成的官方 B 工作（不得重跑）

| 终点 | 实际结果与恢复入口 |
|---|---|
| 共同train64规则/Sap64/CRF，Task2追加GLiNER48 | B100/244 full-LR `77322710…`：6101实体/3415关系，含原LR parser fallback；保留原5871实体对象与3324关系、原排序策略。原devc1c2完整byteexact。只做既有纯计数资产恢复，无新fit、无train80缓存。[旧池终态](official_901_B/old_pool_cpu/terminal-summary.json) |
| patient f787纯B | 原job proc_7c9baef1/driver229438，07:17:15→07:43:36，rc0/1580.513s；420fresh全complete，0partial/abstain/omitted/cap，GPU2实际释放。原ZIP897210B/1df686ce及1273成员逐SHA/fsync核过；无新NN或重decode。[终态](evidence/official-901-B/patient-terminal-summary.json) |
| 纯patient导出与等价证明 | caf48原合同一次.856s导出1769纯模型正例，无baseline/fallback；8881一次.205s证明420实际map/PID exact、positive⊆7147legal、与public-new raw/canonical空间交0。全部100文献/244患者闭合集固定，不能把原dev6条当B规则。refs在同一终态 |
| 官方最终联合审查 | `R/validation/official-highest-901-B-review/review.json` /cd7a6040/8983B，PASS覆盖最终50ef5daa及patient原raw→native→codec→positive全链，无重NN/评分 |
| 持久恢复 | 原raw ZIP同字节保留A100/.100；另56件CPU终点小包750944B/46c42f85，逐SHA/fsync/reopen已过；独审只另加pointer不重包。[补包终态](official_901_B/patient/endpoint-supplement-terminal.json)、[传输故障与同字节恢复](official_901_B/patient/BRIDGE_RECOVERY.md) |

当前901需要同一Qwen3.5基座的**两份不同full64 LoRA**。旧weight1a703b在.100，原state/checkpoint仍A800但原weight已删，不能称A800目录自包含。新patient f787完整state在A100，173200462B四件推断副本已06:31:24实际复制.100并两端SHA/fsync通过；[资源locator](evidence/support-patient-synthesis/two-adapter-resources.json)。901最小补包120468B/65a1已由root实际恢复；原3049/0e2c依赖沿既有链，不重复权重。

## 最近控制与已关闭方向

| 控制 | 结论 |
|---|---|
| N/R/P固定≥2票，两fold纯缓存 | **两折总分均负，关闭**：F1 .663841820660→.663437054811（删29TP70FP；micro升/macro降）；F0 .709556743085→.699390541129（删15TP26FP；micro/macro降）。原双pred共冻9650，官方accepted后仅各唯一v2。F1metric成功后gold诊断误用预测重复断言，保留原failure/metric，恢复只用冻结evaluator集合语义并完成唯一F0；不重评F1。总.632206s，统一审7c445dc9 PASS。[完整终态](research_readonly/majority_vote_control/terminal-summary.json)。不扫票数、不迁原dev |
| full64 selectedCE patient训练及原dev唯一a179并集 | 两轮265×2/68calls，rc0/2991.911s，两epoch完整durable；原dev64唯一311.034s，全部fresh/complete。a179 .746081964986→.746261042824，6新增=3TP3FP，macro降低；不是当时全局best。独审ae594212，203原件全保。[完整终态](evidence/patient-first-full64-memory-recovery/final-terminal-summary.json)、[恢复入口](patient_first_full64/memory_recovery/OPERATIONS.md) |
| 原full64 OOM | 旧proc5b00首epoch反向OOM无完整state，保留并关闭。新selectedCE保持完整BF16head，仅监督位置gather后FP32，CPU/GPU压力probe已独审；不能重复原失败训练或更换精度/预算偷偷续训 |
| patient-first F1 / F0 | F1单独.645531861低旧.660942003；预设并集.664421522。F0主臂负，old∪patient .705527893略升；原strict失败恢复完整58，57complete/1partial，丢8候选如实。随后N/R/P literal union两折Score均升，现作为majority基线。[F1](experiments/patient-first-closed-set.md)、[F0](experiments/patient-first-F0.md)、[三方并集](experiments/relation-triple-union.md) |
| 固定5维LR F1→F0 | .697656525低old∪reverse .704687440及rules .711353522，负关闭；F1标签用于meta fit，不能称pure/OOF。[记录](research_readonly/relation-stacking-five-feature-control/README.md) |
| 旧8B∪旧9B F0 / pair-conditioned BERT / Thinking80 | 分别完整负关闭、完整负关闭、成本暂缓；不因新会话重开，详见08:14历史快照和各实验文档 |

## 记录与接手

- [实验索引 E-20261023-009](experiments/E-20261023-009.md)、[研究笔记](NOTES.md)、[原证据索引](evidence/round2/README.md)。大日志、raw、完整pred、权重均在R或已核持久locator，不进Git。
- 最近实际推送核验：`dec93f4691ffeea985eeb8b6e1704a80ad65c8d5`，含新probe源/12编码事前计划/quality固定设计；`dd7f782`是原CPU执行前源冻结，`3a81ef4`保留首次UNKNOWN启动收据。后续结果按实际完成更新，不为台账自身hash循环提交。
- 所有既有train/infer/score终态均不可重启或重评。当前待执行的是上述已授权唯一GPU工程probe；不能重跑已完成CPU合同。新GPU或质量阶段由root统一预算与端点。
