# CRF 路线工作台账

分支 `round2/sequence-crf`，worktree `/workspace/patientphex-worktrees/round2-sequence-crf`；负责人只修改 `solutions/sequence_crf/`。新任务先读当前 `main:HANDOFF.MD`、`main:AGENTS.md` 与主线程总账，再读本页及对应实验文档。完整旧台账已移至[历史快照](WORKLOG-HISTORY-20261024-0335.md)，其中旧“运行中”字样不表示当前状态。

## 当前优先：用户已授权唯一901方法官方B真实推断

root已转述用户新授权，只提交当前最高901/.754361398199方法；此前默认暂停上传已被这次明确授权覆盖。序列负责共同train64规则base/Sap64/CRF原文候选底座、Task2 GLiNER48追加，以及实际patient f787 B闭集推断；root最终校验并只上传一次，我不上传。禁止新fit/换checkpoint/加入未完成模型或把dev16/六条固定概念当测试预测。

[纯B输入清单](evidence/official-901-B/input-manifest.json)：100篇/244患者/3802passages/1,708,003字符，2,060,389B，SHA878fe248；与train64文献零重叠，未含entities/association。oldpool加载子代理`official_b_oldpool_cpu`只动official_901_B/old_pool_cpu，准备共同base/CRF/Sap供Task1与Task2，GLiNER48仅Task2。旧CLIs会重fit，必须薄加载已存coef/pickle/bank。patient final_infer baf06014只泛化新输入identity/count，保持f787、BF16/native/greedy14000/1600与raw-before-codec，不用devgold。

当前B执行状态（07:05 UTC）：共同Sap B100/90303已19.219秒rc0完整结束，GLiNER链接B100/7048已10.100秒rc0完整结束，均为一次fresh、无fit，.3 GPU0已释放。传输LUNA/UNKNOWN仅影响取回原包，私网原zip桥接.100成功，全253/211成员及每doc原raw SHA核过，自有HTTP已停；不重跑NN。完整full-LR B 77322710已交retrieval：100/244、6101实体/3415原LR回退关系，GLiNER新增230实体。原5871实体逐对象和3324关联保留（按原排序策略，不称prefix）；继承18个identifier=-1实体及13条text fallback未擅改。所有入口与实际收据见[旧池终态摘要](official_901_B/old_pool_cpu/terminal-summary.json)。

患者B纯CPU原生token审计唯一proc_f1a4339f已07:05:38以12.573秒rc0完整结束：420行、1,240,743 tokens、最长6055，全部发布原生IDs/双decode一致；实际prompt9f94与独立转置字节相同，tokenauditccafddec、输入合同eabbfa5f。[CPU终态](evidence/official-901-B/patient-native-CPU-terminal.json)、[原生输入清单](evidence/official-901-B/patient-native-CPU.json)。唯一完整NN计划1e4e67c2已9f93bf9 push/远端exact：原f787、420fresh、3600+30；按旧原dev实耗/输入及候选对比例估1756秒，留2.05倍上界，不采样B NN。原infer5b899保持，driver46abb/exportcaf48仅补绝对deadline和目录项fsync。

**原A100唯一B NN已07:17:15.810647Z真实启动**：`proc_7c9baef1-5019-462b-960a-d5278f96971b` / driver PID229438，420fresh，3600+TERM30，近似TERM08:17:15/KILL08:17:45，以远端`control/infer/driver-started.json`绝对时间为准。[作业](evidence/official-901-B/patient-infer-job.json)、[部署](evidence/official-901-B/patient-infer-deployment.json)、[全卡fresh库存](evidence/official-901-B/patient-infer-fresh-all-GPU.json)。GPU2为0MiB/无进程，其他3卡均有foreign进程；不共享。原运行库实际Torch2.6.0+cu124/TF5.20.2保持，无环境迁移。

07:08/07:11两次LUNA后，07:16库存确认infer-plan原未落，才同字节部署8324B/1e4e并fsync重开；此前未start，失败收据保留。root批准的.100备用只完成资产库存，派生入口尚未产生，原A100启动后已正式[取消于准备前](official_901_B/patient/cache_fallback/STATUS.md)，不能称有备用可执行源码。恢复只运行[只读监控](official_901_B/patient/monitor.py)：`python solutions/sequence_crf/official_901_B/patient/monitor.py --run-dir /workspace/patientphex-runs/round2-20261023/research/official-901-B/patient --client-dir /workspace/patientphex-tools`；本地flock单owner、≥180秒一次、只读已有control/raw，不重启remote。当前本地watch session70981；若会话断先核lock/PID/最新receipt，再接管监控。

[独立实际B合法空间证明](evidence/official-901-B/patient-space-proof-summary.json)：420原schema逆变换/逐行code_map一致，100文献原HPO后代alt图与public-new实际code_maps canonical+literal交0；16个合法text概念保持。实际patient prompt9f94fffc/5403066B已与native产物相同字节。此时仅证明输入空间，实际positive成员/输出级互斥待完整raw后验证，不能称B结果已完成。后继研究只读一页[机制盘点](research_readonly/post901-relation-proposal.md)，不阻塞官方，不新NN/fit/评分。

B最终合成还须按root规则核：patient合法HPO来自oldpool，public-new来自相对oldpool差集，在实际canonical/alt处理后逐doc空间及输出级严格互斥才允许等价全union；否则精确原counterfactual a179_B差集，不能猜或临时删patient。此证明与retrieval共同完成。

恢复补线：患者173,200,462B四件第二副本已06:31:24实际完成（22.709秒rc0），两端fsync/SHA及目标祖先fsync全过，ACK c38a609b，源原完整state不变。此前LUNA只影响部署；实际两次空库存后同字节压缩部署成功，唯一HTTP06:31启动且完成后仅停止自身。零GPU。新最小901补包已持久并由root06:42实际下载验证35 content+members.json，zip65a1d0b5/120468B，manifest7d479a63；[补包指针](evidence/support-patient-synthesis/persistent-backup.json)；[复制终态](evidence/support-patient-synthesis/adapter-copy-terminal.json)。

## 目标与已知最佳

- 用户验收仍是原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text、Score ≥ .8 且格式/复现通过；内层F0/F1的诊断不能替代该目标。用户现在已授权唯一901官方B真实推断/提交，由root执行上传；本次full64→原dev唯一端点已完整结束，其他新评分臂需明确决定。
- 每个独立模型 ≤10B，允许多模型；不要重新按全链参数合计禁用集成（main HANDOFF H47）。
- 本路线已知的跨方案原dev候选 **.746081965**：固定逆序关系并集，1072实体不变，原368关系保留、新391关系；+.007588212、15TP8FP，已独审PASS，仍未.8。[结果与恢复](experiments/reverse-json-order-dev.md)。更晚的跨路线最佳以主线程总账为准。
- 本路线独立原政策 **.711324992**（独审PASS），旧CRF/Sap、GLiNER及HGB实体控制详情均在历史与各实验文档中。[当前资源清单](experiments/best-pipeline-resource-inventory.md)、[已存HGB推断入口](experiments/load-existing-selector.md)。

## 当前任务结案：full64 完整成功，原dev增益很小，未达 .8

selected-CE fresh训练已06:03:05Z完成265×2/68calls，rc0/2991.911秒；两个epoch各5成员519,867,104B均fsync/祖先fsync与重开SHA通过。真实epoch2三SHA绑定后，唯一原dev64推断 `proc_05ce4894-3a78-49c7-8c9b-60191eac14d4` 已06:13:10Z **rc0/311.034秒**；64条全fresh，64 strict/codec complete、零partial/abstain，203小件逐SHA及原始ID独立decode通过。两作业均已结束，卡实际释放，不重启或重评分。

唯一a179∪patient新 **Score .746261042824**，相对原a179 .746081964986 **+.000179077837**；新增6关联=3TP3FP，micro+.001940340117、**macro−.001224028768**。1072实体不变、旧391关系和35public-new全保留、最终397关系。完整raw/pred先冻2ac4b470，再唯一v2；格式0错0警告。新pred765cd5a0/141732B。仍低root最新已独审全局支持gate3049d9fe **.754216969282**（完成端点时的前一0e2c为.748583439566），原dev .8未完成，未替换全局best、未合并/官方上传。[完整结果](evidence/patient-first-full64-memory-recovery/final-results.json)、[终点与原件索引](evidence/patient-first-full64-memory-recovery/final-terminal-summary.json)。终点增量独审ae594212 PASS（原203件/64EOS/三SHA/保留合同/唯一评分顺序），不重跑旧合同或新质量。

F=`/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-memory-recovery/formal`。实际training-collection b5164026，inference-collection f68df296；训练/推断原raw、两个fullstate、freeze/metrics/旧OOM均外存。已完的train/infer launch-intent及watcher-terminal是终态凭据；恢复只读既有小文件，不执行旧start/bind/assemble/evaluate命令。[恢复操作](patient_first_full64/memory_recovery/OPERATIONS.md)。

**已核依据**：旧proc5b00首epoch backward OOM、无完整状态，独审f7668be7；不能exact续训。新恢复只将有监督位置先选后FP32 CE，完整BF16 head/每rowmean/有效group/精度不变。目标Torch288例通过；新probe12次前后向/1次完整group更新，已有496 Adam状态与累积梯度下最长6166通过，147.295秒rc0、峰allocated25.031GB/reserved29.977GB，05:11:10实际释放并弃权重。统一独审b46983ef通过，正式fresh已获root一次统一授权。探针通过不保证完整训练稳定，原失败记录不改写。

- root已授权完整链：每epoch持久fullstate → 真实epoch2三SHA绑定 → 原dev纯64一次900+30 → raw和a179并集完整freeze → 唯一v2；不评patient-alone，不重算旧候选，不删旧1072实体/391关系/35public-new。Score严格升记候选best；macro降低如实披露；.8加独审才达标。
- [full64运行/恢复记录](experiments/patient-first-full64.md)。最长6166实际一步81.873秒PASS/卡已释放；正式CPU10.454秒PASS。未知上传经库存确认未落，3小片补齐同源后再启动。源码与job已推送；当时未新devgold；现原dev单一质量已完成，未官方。

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
- 截至本页整理前，已push并lsremote确认 `ca72c3cde197650428ed2ba4550aa84bd20f1f43`，含原失败、新工程完整终点、fresh正式job、环境重连恢复与后继单一dev入口。不要为记录台账自身SHA递归提交。
- 旧full64 OOM完整保留；新selected-CE fresh两轮与原dev64唯一a179并集已完整结束，终态增量独审已PASS；新唯一CPU合成另设源与计划，不改原终点。逐关键阶段更新本页、实验记录、commit/push。未达原dev .8的实现不合并main，不宣称比赛目标完成。

- full64 新CPU实际通过：265条/860405 token，最长6166/122监督；已冻结单步240+30新源与6输入，native265实际8.432秒通过，唯一最长probe81.873秒rc0、6166完整一步PASS，峰allocated38.184GB/reserved39.083GB，结束free2.666GB；04:34:44 GPU2实际核空，root据探针已统一授权正式4500+30与原dev64另900+30，实际formalCPU10.454秒通过，正式唯一proc_5b00069a/PID120406于04:42:33.455Z启动，04:51:51 OOM/rc1退出；原4500+30未耗尽。见 [full64记录](experiments/patient-first-full64.md)。

- 新正式fresh已05:13:13Z启动 `proc_7f51657c` / driver141616，4500+30，近似TERM06:28:13/KILL06:28:43（remote driver为准）；已实际部署22件SHA/compile/fsync，持久盘free57.076GB；保持265×2/68calls/4warmup、4500+30及原dev64另900+30，原a179单臂不变。新probe通过但不保证完整两轮；[entry](evidence/patient-first-full64-memory-recovery/formal-entry-readiness.json)已获root一次统一授权，**当时fresh启动，现已06:03:05完整结束**；旧OOM与新probe状态独立。恢复入口见[操作说明](patient_first_full64/memory_recovery/OPERATIONS.md)。

## 后继唯一CPU合成已接受：当前全局 .754361398199

root在两独立结果及终审已知后明确授权；源b9ea4f50/plan edee73ab由e731663提交推送并lsremote精确后，完整16pred于06:18:59冻结d3f058fd，06:19:04唯一v2完成。以3049完整基底只加765−a179精确6差集；实际与现有410/54public零交集，因此加入全部6=3TP3FP。1061实体/36患者/416关系，原410及54public全部保留；非关系原字节/旧phenotype前缀与元数据精确，新支持gate删除的2public未恢复。

Score **.754216969282→.754361398199（+.000144428917）**；men .792801107522/doc .828677839851不变，micro .736964078795→.738780207135（+.001816128340），macro .658424850960→.657186438289（−.001238412671）。按固定Score门和统一终审76ada874 PASS，root已接受为当前全局best；幅度很小且macro下降，尚未.8。pred9016863c/140578B，格式valid/0错0警告，assemble .02849秒、eval .17502秒；0NN/旧重评/扫描。终审已回收；最小补包已持久并实际恢复，新adapter第二副本也已完成，不重复模型/评分。已有原dev反复开发，不是盲验证。[终点](evidence/support-patient-synthesis/terminal-summary.json)、[结果](evidence/support-patient-synthesis/result.json)、[源码与恢复](support_patient_synthesis/README.md)。

恢复资源必须按同一Qwen3.5基座**两份不同full64 LoRA**记录，不能再称单adapter重复推断；[资源locator](evidence/support-patient-synthesis/two-adapter-resources.json)。旧LoRA的A800原weight已在此前释放，.100已有真实SHA副本；原state/checkpoint仍A800，不能称原目录完整自包含。新patient完整state在A100；root明确授权4文件173200462B另外备份.100，8MiB/s/180+10，仅CPU/磁盘。现已06:31:24复制完成，见上述实际ACK；先前未落部署/LUNA保留为历史，不重模型。

B CPU底座6件来源入口已完成，见[阶段摘要](official_901_B/old_pool_cpu/cpu-stage-summary.json)：旧dev55f完整字节回放；B CRF5229实体、rules5728，100/244；经root明确授权的纯计数恢复不新fit，Sap90303候选/65673 query、42575 labelbank精确。GLiNER最终排回原native .100GPU1，在R1释放后唯一100篇已06:41:29完成rc0/97.733秒；3885完整chunks/7048unique候选，GPU1已0MiB/无进程并交retrieval。9件原raw/候选/windows/日志已逐SHA/fsync收回，见[实际终态](evidence/official-901-B/gliner-terminal.json)、[收件清单](evidence/official-901-B/gliner-collection.json)。Sap query由子代理在.3 GPU0单owner依次执行；先原dev16缓存CPU合同，再B600+30，随后GLiNER片段的同模型链接。原模型/labelbank不重编码或fit。

patient B参数化入口只改输入identity/count和artifact标签，f787三actual SHA、greedy14000/1600、原raw-before-codec与native BF16完全保持；源码[增量清单](official_901_B/patient/source-delta.json)。normal-B闭集prompt3a8e已绑定完整oldpool7732；native420全ID/decoder核过，唯一计划1e4e已真实启动，详见顶部当前状态。

后继唯一N/R/P≥2票双fold控制源码/计划已c5f2455提交推送并lsremote exact；两完整预测共同冻结9650ab99，F1关系469→370、F0 317→276，其余字节保持，0.08785秒纯CPU。未读gold/评分，无keep结论；[待评分清单](research_readonly/majority_vote_control/pending-score.json)。根的保留门为两fold Score各自严格高于原tripleunion，macro只披露；必须等root真实官方accepted receipt释放后再各唯一v2/总30+1，不迁原dev。
