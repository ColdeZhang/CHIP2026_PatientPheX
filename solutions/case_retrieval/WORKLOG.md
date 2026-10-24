# 病例检索路线：当前工作索引

更新：2026-10-24 UTC。分支 `round2/case-retrieval`，worktree `/workspace/patientphex-worktrees/round2-case-retrieval`。本文件只保留接手必需的信息；完整阶段过程在对应方案目录。

## 先读与真实目标

- 执行新任务先读主仓 `AGENTS.md`、`HANDOFF.MD`，再读本索引及当前方案计划；主仓 `/workspace/CHIP2026_PatientPheX` 负责总指挥记录。禁止凭旧running字样或旧空卡收据重复启动；先查known job、绝对deadline和完整终点。
- 用户目标仍是原train64/dev16、冻结公共v2 all/text Score≥.8。最近root确认原dev共享方案为.746081965，尚未达标；主线实时结果以主仓台账为准。当前路线不得把F1开发折的分数当原dev/独立泛化/官方成绩。官方上传仍须用户另行通知。
- 参数限制按已核官方原文是**每个独立模型≤10B**，不是全链参数相加≤10B；完整模型、adapter和资源须如实披露。Med原件4.300079472B＋LoRA29.802496M=4.329881968B。

## 当前状态：Med短输入完整负结案

- 固定strict32旧124条×2epoch，原CID→PID任务、原生assistant CE/r16/238语言投影，32calls、248曝光；完整训练＋两次全状态持久导出/回读657.090秒<1200。两epoch loop合计316.568秒。
- 原80输入完整推断，native226.016秒、campaign257.325秒<600；80/587CID全部有效，0未决、0触顶，1189实体及全部非关联原字节保持。GPU2于03:38:06.634Z实际空释放，旧训练/export/inference作业均结束，勿重启watch40635。
- **旧基线 .660942003 → Med单独 .603701549；预先唯一并集 .644663430。** 关系TP/FP/FN分别310/73/210、253/147/267、326/176/194。并集新增119关系仅16TP/103FP，两端micro与macro均降，不达预定留用条件；不迁F0/原dev，不补阈值/组合/额外epoch。
- 两个fullpred共同freeze `d7a52409` 在首次gold前，各唯一调用原v2，旧指标复用。见[固定结果](medgemma_relation_control/RESULT-short-input-frozen.md)、[结构化结果](medgemma_relation_control/quality-result.json)、[质量摘要](medgemma_relation_control/quality-F1/summary.json)。这是固定配方质量负，不是格式故障。
- 统一工程审 `1dc8c37b`、正式入口审 `1d15ecd8`；完整终点审 `fcc84401` 已通过，见[终审](medgemma_relation_control/review/formal-terminal-review.json)。raw与双fullpred另已.100归档b66fd723/33项619551B，逐成员回读/fsync通过。实际全链清单为[terminal-manifest](medgemma_relation_control/terminal-manifest.json)。当前状态文件已转closed，并保留before-terminal原件，原deadline不改。

## 当前后继：闭集全文正式运行

- 不变的CID→PID任务只追加全部允许原文。CPU13.458秒核124/80、原答案/CID/PID/候选/原消息与患者坐标精确保持；训练965436tokens/ep、最长12824、原监督7150不变，约旧382000的2.527319倍。F1输入645000tokens、最长13805。
- 51条训练超过现有8192；旧Med trainer原样保留，未默改或截断。单独最长诊断显式容量14000，原native更新/cache数值段5562B逐字节复用；源码/CPU已审 `4a6544d2`。最长诊断用显式14000、freshbase、状态丢弃，已完成且无质量；当前正式控制另用新源/namespace。
- 全文最长probe已完整关闭：proc_b941c71a…，native30.681秒/更新8.469秒，allocated58.364GB/reserved60.677GB，真实2token cache通过；03:45:17.126Z GPU2实际释放。原03:47:36Z+TERM30未延。见[工程终点](medgemma_relation_control/fullcontext-preparation/probe-a800/REPORT.md)，无该probe质量，不重复启动。
- 仅按旧epoch loop乘token比约400.255秒/ep、800.510秒/2ep，是粗估；不含load、每次实际约118秒的导出、推断，注意力和显存可非线性。实际多形状峰与总耗时待本次记录。
- 见[全文CPU报告](medgemma_relation_control/fullcontext-preparation/REPORT.md)、[CPU小清单](medgemma_relation_control/fullcontext-preparation/backup-filelist.json)、[最长源计划](medgemma_relation_control/fullcontext-preparation/probe-source/source-plan.json)。历史做过开放HPO/quote全文SFT；尚未测的是同CID闭集目标的train＋infer全文SFT，不能泛化成所有全文SFT未测。

## 恢复与持久资产定位

- `R=/workspace/patientphex-runs/round2-20261023`；当前实验外部根 `R/research/medgemma-relation-control`。大模型、raw/fullpred、长stdout均在仓库外，Git只放源、冻结计划、关键收据和摘要。
- A800 mount `69b991ee-183f-4add-af1a-579af92479c9`，GPU2 UUID `GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc`；自有SHM `/dev/shm/patientphex-medgemma15-1045-20261024`。GPU占用必须fresh核，SHM不是持久备份；项目盘仅约165MB，只存小源/日志。
- .100 mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，项目根 `/home/zhangyuheng/projects/CHIP2026_PatientPheX`。固定19件8.64GB模型在 `research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30`，MS revision `9bca96ca3e3049f6a084d8c876078cedb3069f18`。
- 两份Med全状态各358430727B，持久目录 `research/medgemma-relation-control/formal2ep/checkpoints/epoch-001`、`epoch-002`；final adapter `9b46f756`、Adam/RNG `000a8043`、checkpoint `68c59c94`，包含_run源码与_cpu依据。回读ACK见[epoch2](medgemma_relation_control/formal-preparation/formal/epoch-002/export-ack.json)。
- native运行时沿用TF5.20.2/PEFT.21.2/Torch2.7.1，未安装新框架；精确python/env/源hash看冻结run-plan，不能凭模型卡版本替换。新任务恢复原自己完整epoch必须保持输出/模型路径及fingerprint；resume数值等价未另作实测。
- 首次Med复制因监控503后finally停源失败，原2.092GB part保留；唯一900恢复实际722.052秒完成。修复规则：监控UNKNOWN保留known IDs，不自动停健康配套server/重启；远端原绝对timeout仍有效。

## 最近关键信息与入口

- [顺序训练增强](order_warmstart_stage/RESULT.md)：旧.660942003，新O/O .658421563、R/O .645528610，固定两臂负；四全状态已持久/审d253f625。训练增强负不否定此前固定reverse推断union原dev正结果。
- [共指B](public_coreference_readiness/prompt-control-b/REPORT.md) .660756991负；公开212训练 .674046941低于对应旧.682128804，value2 .660769446亦未改善。各原始输入/终点见evidence/round2与历史索引，不重复跑已闭路线。
- [thinking4096](research/thinking-sampling-followup.md) 未闭think/无strict final是格式终点；后续budget-forcing归sequence，旧行为摘要已补token位置限定。FLA raw-logit合同失败见[fused结果](fla_kernel_readiness/GPU_RESULT.md)，不等于任务质量负，正式runtime未切换。

## 记录与下一步

- 本索引只更新阶段结论、活动job/预算、资产入口和下一步；实验细节写各方案目录。关键方法/结果/失败与恢复及时commit、push并核ls-remote；正常逐步训练日志不入Git。只改本路线目录，主总台账/HANDOFF由root统一维护。
- 短Med负与全文工程审2cc62bee已结案。root新授权全文fresh124×2ep/32calls：共享2700+30含两load/两全状态ACK，原full80另900+30；唯一primary fullMed/secondary oldSFT32∪fullMed，双freeze后各唯一v2。正式新源/端点20+16项CPU合同及9件实际部署已完成，独审[41e51269](medgemma_relation_control/review/fullcontext-formal-entry-review.json)已PASS，统一入口[6c87fee0](medgemma_relation_control/fullcontext-preparation/formal-preparation/entry-manifest.json)、plan60eb799f；首epoch510.187秒/16calls已完成；27件358631877B首state已持久ACK并全SHA回读，47.764秒。04:06:23.594Z第二epoch以同state恢复proc_443612cd/PID60537，watch31021；训练总绝对截止04:40:33.082Z+TERM30，不重新execute。状态见[恢复入口](medgemma_relation_control/fullcontext-preparation/formal-preparation/formal/execution-state.json)。H48追加窄CPU17.716秒，Med/Qwen各80文本Auto/native IDs与解码同，组件/音频special差异如实记[报告](medgemma_relation_control/tokenizer-family-audit/REPORT.md)，无模型/gold。首OOM/非有限/截止停，不自动扩预算或F0/dev/官方。任何方法即使高于短Med.603702，也需与强旧基线.660942比较。
- 本次压缩前完整台账原字节已归档至[历史正文](history/WORKLOG-through-20261024-0339.md)，SHA `ddb89cb9545f1653b1304c6e97948cf14ad4603cce97b61956b8a09ca22bdb55`；它保留当时状态，不是当前启动依据。
