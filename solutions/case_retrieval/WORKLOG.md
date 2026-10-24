# 病例检索路线：当前工作索引

更新：2026-10-24 UTC。分支 `round2/case-retrieval`；worktree `/workspace/patientphex-worktrees/round2-case-retrieval`。先读主仓 `/workspace/CHIP2026_PatientPheX` 的 AGENTS/HANDOFF/总台账，再读本索引。用户目标仍是原dev公共v2 Score≥.8，当前不得把反复观察的内部F1分当原dev、独立泛化或官方成绩；官方上传须用户另行通知。

## 当前：全文Med双端点完整负结案，无运行GPU

- 强对照旧SFT32 **.660942003**；全文Med **.592418621**，唯一旧SFT32∪全文Med **.627966889**。关系micro/macro均低于旧对照，不达事前“关系micro+macro严格增且macro不降”条件。并集新增193关系中28TP/165FP；格式80/587CID全有效、0未决，1189实体及非关联原始bytes不变。这是该固定配方质量负，不是格式/覆盖故障。
- 实际测试的是同CID→PID闭集任务在train+infer同时追加允许全文，fresh公共Med基座；原strict32的124条、答案/CID/PID/候选不变，7150监督tokens/ep、238语言投影r16，32calls/248曝光、原nativeCE/AdamW/BF16/SDPA。显式训练cap14000，不继承短Med adapter。训练965436tokens/ep、最长12824；F1原80输入645000tokens、最长13805，均不截断。
- 两epoch loop510.187+507.378=1017.565秒；两load/训练/两完整持久导出及回读合计1264.674秒<2700。两份27文件全状态各358631877B；final训练进程allocated52.003GB、reserved75.244GB。先前800.510秒是token线性估算，未改写为实测。
- fixed80 native282.138秒、campaign307.881秒<900。两完整pred共冻`28f5d4be`先于firstgold，各唯一原v2，旧指标复用。raw`c4fdf9e8`、主pred`f0f83c8c`、union`3063f98b`、summary`cf69328e`。见[完整结果](medgemma_relation_control/fullcontext-preparation/RESULT-fullcontext-frozen.md)、[质量摘要](medgemma_relation_control/fullcontext-preparation/quality-F1/summary.json)、[全终点清单](medgemma_relation_control/fullcontext-preparation/terminal-manifest.json)。
- 原controller/watch31021已结束；epoch1 proc_2d5099e9、epoch2 proc_443612cd、fixed80 proc_fc7f61d1均终态。GPU2 f516于**04:21:32.942369Z**实际空81156MiB，release`9443432d`，已交root/ontology后续自行fresh核。不得因旧空卡收据抢占。训练截止04:40:33.082Z+30、推断04:31:39.654Z+30未延；两个current state已closed并保留before-terminal/原训练中原字节。
- 入口独审[41e51269](medgemma_relation_control/review/fullcontext-formal-entry-review.json)通过；raw/fullpred已额外持久归档`07b2ca5a`，33项638317B，逐成员回读/fsync通过，见[回执](medgemma_relation_control/fullcontext-preparation/result-backup/actual-persistent.json)；完整终审[f7e5df86](medgemma_relation_control/review/fullcontext-formal-terminal-review.json)已PASS负结案。**不迁F0/原dev/官方，不补阈值、组合或额外epoch。** 更换其他基座/schema的全文SFT不在此否定范围。

## 对照与关键解释

- [短输入Med固定负结果](medgemma_relation_control/RESULT-short-input-frozen.md)：同strict32两ep，Score .603701549、旧并集.644663430，终审[fcc84401](medgemma_relation_control/review/formal-terminal-review.json)通过。全文主/并集再低.011282928/.016696542，旧短结果仅缓存比较，没有重评。短raw与双pred已.100归档b66fd723、33项619551B、逐成员回读/fsync。
- 本次全文CPU13.458秒，51训练行超过旧8192；[最长真实probe](medgemma_relation_control/fullcontext-preparation/probe-a800/REPORT.md)单步8.469秒、allocated58.364GB，状态丢弃。CPU审4a6544d2、probe审2cc62bee复用；不把probe或短行资源当正式多形状保证。
- H48新具体风险引出的[一次CPU对照](medgemma_relation_control/tokenizer-family-audit/REPORT.md)17.716秒：Med/Qwen各原80纯文本Auto与发布JSON的IDs及双decode相同。组件序列化、Qwen额外3个未用音频special差异保留，不泛称全词表/所有输入等价。
- [H49窄核](medgemma_relation_control/fullcontext-preparation/H49-raw-first-note.md)：本runner不parse，先逐条decoded raw写盘/fsync，完整80回收冻结后才独立parse。未热改/重启；不宣称保存了全部generated IDs或所有I/O故障都无损。

## 资产、恢复与记录

- `R=/workspace/patientphex-runs/round2-20261023`；本阶段外部根`R/research/medgemma-relation-control/fullcontext-preparation`。大模型、raw/fullpred、正常stdout在仓库外；Git只留源、计划、关键收据、指标与摘要。实际CLI、runtime、sourceSHA见[run-plan](medgemma_relation_control/fullcontext-preparation/formal-preparation/formal/run-plan.json)，不得重跑已结束execute。
- A800 mount`69b991ee-183f-4add-af1a-579af92479c9`/uid1045，Med私有SHM `/dev/shm/patientphex-medgemma15-1045-20261024`。SHM不是持久备份；04:21终点仍约237.455GB空，后续占用以fresh事实为准。项目盘仅约155MB，只放小源/日志，不放权重。
- .100 mount`5a9955da-5240-41cc-9e52-05d3d8da8a3c`，项目根`/home/zhangyuheng/projects/CHIP2026_PatientPheX`。Med19件8.64GB固定原件在`research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30`；MS revision`9bca96ca3e3049f6a084d8c876078cedb3069f18`，全部SHA/883 headers已核，勿重复下载。
- 全文两epoch持久目录`research/medgemma-relation-control/fullcontext-formal2ep/checkpoints/epoch-001,002`；final checkpoint`c4044dac`/adapter`47798260`/Adam-RNG`0bfa4203`，两份fingerprint相同。含9源码、完整优化器/scheduler/所有RNG及实际CPU依据，均导出ACK后独立全SHA回读。短Med旧目录`.../formal2ep/checkpoints/`独立保留，不能混用。
- 运行时TF5.20.2/PEFT.21.2/Torch2.7.1沿旧验证环境，无新安装/FLA/量化。单个Med完整bundle+LoRA共4,329,881,968参数；官方上限是**每个独立模型≤10B**，不是组合总和≤10B。视觉/投影模块冻结并禁止forward，计数仍包含它们。
- 监控503/UNKNOWN必须保留known IDs/原绝对deadline，不自动重启或finally杀健康HTTP源。修复模板见外部`../transfer-a800-recovery/run_once.py`0f64a522；只在fetch确认终态或原截止+30后清理自有server。

## 历史与下一步

- [顺序增强](order_warmstart_stage/RESULT.md)：旧.660942003→O/O .658421563、R/O .645528610，四全状态持久/审d253f625，固定负；不否定此前reverse测试时并集的原dev正结果。[共指B](public_coreference_readiness/prompt-control-b/REPORT.md)、公开212、value2等已闭负控制不重跑。
- [thinking4096](research/thinking-sampling-followup.md)是未闭think/无strict final的格式终点；后继budget forcing归sequence。FLA raw-logit合同失败见[fused结果](fla_kernel_readiness/GPU_RESULT.md)，不等于任务质量负。旧零样本全文、开放HPO全文SFT与本次闭集全文SFT分别记录，不能混称同一实验。
- 后继[全局兼容解码](global_compatible_decoder/REPORT.md)已按root新增授权完成固定负控制：先3.194秒无标签census，再3.488秒精确有理数枚举；4,633候选/66冲突组件最大3，唯一F3一组件删1加2，其他三fold byteexact。共冻后唯一5次新v2、旧指标复用；pooled .730993616→.730822309，F3 −.000731780，其余Δ0，全部非entity原始bytes/关系保留。事前pooled严格升且各fold不降不满足，keep=false，保原nested、不改效用/cut/迁dev；统一增量终审98962250已PASS负结案。无新GPU、fit或下载。
- root授权[四服务器一次资源快照](evidence/round2/gpu-survey-20261024-043924.json)：04:39:24 UTC未发现新可推荐≥40GiB空卡。A8000仍foreign PID56706/594MiB，1/3有进程，2仍R1F0预留；.100两卡在用（1为R1F1）。A100只返回UNKNOWN_OUTCOME，未重试；GPU2全量训练预留不变。.3 A30 GPU0仅4MiB、0util/no compute，为24GiB小任务候选，使用前仍须fresh核。仅查询UUID/显存/util/PID，未查命令行或启停任何模型；已同步root及两owner。
- [固定LightGBM控制](lightgbm_selector_control/REPORT.md)已获root新增授权，源/计划d8935e74及entry59aaa484冻结待唯一执行：同强B八池/特征/160轮/15叶/min35/.06/L2=5，.3/nested不动，四foldfit/predict总180秒，完整共冻后新5次v2、旧指标复用；keep pooled严格升且≥3/4fold不降。独立私有4.6.0 wheel cb19b5af安装1.619秒，synthetic原生/正列/保存回载1.163秒通过；未改共享依赖或下载模型。实际分箱/生长差异明确，不宣称数学等价；正式当前未开始，运行后以外部formal-control状态为准，不重启已有输出。
- 仅改本路线目录，主总台账/HANDOFF由root维护。关键阶段及时commit/push并核ls-remote，正常逐step日志不入Git；旧完整台账原字节见[历史](history/WORKLOG-through-20261024-0339.md)，SHA`ddb89cb9…`，不能作为当前启动依据。
