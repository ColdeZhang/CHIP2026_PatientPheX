# CRF 路线工作台账

分支 `round2/sequence-crf`，worktree `/workspace/patientphex-worktrees/round2-sequence-crf`。负责人只修改 `solutions/sequence_crf/`；新任务先读当前 main 的 AGENTS/HANDOFF.MD/WORKLOG，再核本页与真实文件。完整历史见 [08:14快照](WORKLOG-HISTORY-20261024-0814.md)和[03:35快照](WORKLOG-HISTORY-20261024-0335.md)，其中“运行中”只代表当时状态。

## 当前任务：患者均衡混合损失 α=.5，正式两轮训练

- **目标/边界**：固定总Score目标，单变量跨患者归一混合loss，与已负value×2不同。CPU与真实工程已完整通过；root授权唯一fresh两轮2400+TERM30，最终epoch2完整durable后另80fresh/600+TERM30；主臂N∪R∪newP替换旧P、诊断newP-alone双pred共冻，各唯一v2。旧指标复用，macro完整披露，无新官方上传。当前原dev best仍 `.754361398199`，未达.8。
- **固定机制**：原 train32 patient-first 124行/32文献/87患者、原两轮顺序；保留一半原 token CE，新增一半患者及患者内 chunk 等权 list-value CE。每epoch P87/S16；row内新系数含真实group size，保留外层 `/len(group)`，包括尾4。原格式/key/EOS/换行由旧半项保留；不按 prepared 正例数加权，prepared空不等于真实gold空。
- **训练侧准备已完整通过**：唯一2.716751秒，manifest `c713111b…`；原全部124行IDs/labels/rendered/audit exact，400614 tokens/4694监督/max5556。精确 Fraction 证明两轮每患者新增份额1/87、混合1/174，跨两个患者value token为0，106处空列表保留。此系数不等于实际梯度、Adam更新或质量因果。[协议/源](patient_balanced_loss/README.md)、[实际清单](patient_balanced_loss/preparation-actual.json)。
- **数值合同完整通过并独审**：原唯一A100 Torch2.6.0+cu124、60+TERM5/CPU隐藏，worker5.207856/driver5.690608秒。8原件全SHA/fsync、317观察全PASS，source141c/plan2145；完整124系数桥接1/87→1/174、FP32/BF16、ignored NaN/Inf及g4/g8均通过。独审8765aeb8/10376B已PASS；FP32 loss最大绝对误差4.768e−7/grad3.388e−9，BF16梯度对已舍入参考max0，不声称所有loss位精确。[单份终态](patient_balanced_loss/CPU-terminal-summary.json)。原start UNKNOWN与收件UNKNOWN/LUNA只影响传输，保留原收据；无重执行。
- **首个GPU工程probe已失败收口**：原proc_470248fd/PID4698/child4722，08:31:12→08:32:24，rc1/72.035446秒，未超时/OOM，GPU2实际0MiB/no_process。row01/原index62/2704tokens的完整head已走到loss hook，随后参数guard断言；mixed_loss core未调用，0backward/0Adam/无质量。11原件完整SHA/fsync收回，collection7199bb34；[原失败摘要](patient_balanced_loss/probe-failure-summary.json)。保留原source/plan/全部失败，禁止重启原目录。
- **独立机械恢复完整通过**：proc_0f960ddf/PID13376，08:43:45→08:46:24，rc0/159.843927秒，12反向+1真实非零Adam，最长5556在已有Adam+3累积梯度下通过；496梯度/state finite、三族B实变/base冻结。12真实call均只有两diagnostic=None，scope全恢复。峰allocated25.699GB/reserved40.077GB，后者接近卡额如实披露；GPU2实际0/40339释放。77原件/ZIP971478f6全SHA/fsync，collection8986ae69；观测LUNA只延迟收件，未重启。原72.035秒失败仍保留。[单份完整终态](patient_balanced_loss/interface_recovery/terminal-summary.json)。
- **probe输入/源码**：12行唯一编码0.693748秒、44992tokens/530监督，原IDs/labels/rendered/整metadata exact，payload4cfdb1f1；source6f0188d4/hook97a0c5ce/driverafa8c149已独审静态无阻断。[部署原件](patient_balanced_loss/probe-deployment.json)。原H5初次释放后等其机械恢复完成才start，旧快照保留但未作当前gate。
- **正式两轮完整成功**：proc_afaf0b83-b325-446e-9769-5955d8c8136d / driver19546，08:52:35.025475Z实际driver启动，2400+TERM30（TERM09:32:35.025472/KILL09:33:05.025472）。源fc24e27先push，trainer3eab/同7b99hook、driver43c730、plan e3c80b43；19成员已完整SHA/fsync/reopen。固定124×2/32calls/warmup2/尾4，fresh模型/newLoRA，探针状态丢弃；09:17:01实际rc0/1466.505575秒、124×2/32calls；两epoch含保存699.541837/693.651216秒。69原件全SHA/fsync收回，collection22117c77、formal_reported_success=true；两ACK/四大件实际重开SHA核过，完整Adam/RNG与两adapter留原持久目录，峰allocated25.698GB。09:17:04 GPU2实际0/40339释放。[作业记录](patient_balanced_loss/formal-deployment-start.json)、[操作与后继入口](patient_balanced_loss/OPERATIONS.md)。工程统一终审7a010be3已PASS，仅证明工程/源码，未声明正式或质量成功。
- **运行位置**：本地 `R/research/patient-balanced-mixed-loss-alpha05/formal-v1`；R=`/workspace/patientphex-runs/round2-20261023`。远端mount476f/project `sequence-round2/patient-balanced-alpha05/formal-v1`，输出formal2ep；不要重启/另namespace重复fit。
- **下一步**：原formal已完整终态，禁止重启。最终80的三实际收据已唯一绑定：planffb898c3/resultd9ed01c0，source仅三常量变化AST exact；source/plan推送后部署原600+30唯一fresh80。推断须80全fresh且raw-first/原strict EOS与完整PID-CID政策，双端点先冻再唯一评估；失败/超时保留原件，不自动改loss/allocator/预算。当前源prepare与观测并行，不把工程PASS当质量改进。


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
- 最近关键推送核验：`57bd62761c461f6095d00d19b9840d7e2a417230`（第一轮完成与端到端恢复入口）；`fc24e27`是正式训练执行前源/计划，`25a6e074`是v2薄接口PASS与恢复plan。09:11:55原job实际已epoch2/global_step20，PID19546/19564仍在。后续结果按实际完成更新，不为台账自身hash循环提交。
- 所有既有train/infer/score终态均不可重启或重评。v2工程probe已完整终态，正式两轮唯一proc_afaf0b83已完整结束；不能重跑已完成CPU合同。新GPU或质量阶段由root统一预算与端点。
