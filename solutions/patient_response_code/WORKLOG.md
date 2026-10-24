# 工作台账

当前：固定strict编码单模型F1 .686618943020、F0 .700893777351均超过各自强N；两折都选coded-alone。F0两micro下降而macro/总Score升，仍低全流程更强参考，不称globalbest。F1已独审/202包持久；F0187件全收、两固定质量完成，正做小备份/终审交接。无活跃GPU/监控；原结果不改。root另授权下一独立PT cyclic1仅CPU/source准备，待本次持久收尾后新建分支。

- 目标：新响应code表示能否超过同任务强N .676546354663。主coded-alone、诊断N∪coded，固定两全量先共冻后各1v2/旧0，最高端点超过N可候选；另报历史fullsupport .686338981，非原dev。
- 数据：短train124 cb7f→f73cb，全文held80 2a95→54c8，原user整段UTF8/CID-HPO/clinicalPID/mentions不变，旧renderer0调用。原样本顺序保持；只是新的system输出说明＋response表及target患者值。
- 实际CPU：18.607秒PASS，训练427560tokens/max5910，监督7170→8138（PT001四token）；held685372/max14570，新工程cap14848不裁切，gen1600。所有target逆映射精确，12负例拒绝；旧identity桥接非本机制。来源R/research/patient-response-code-readiness-20261024/ready.json。
- 探针：A100GPU2 UUID2c22a952；12行first8建立非零诊断Adam，然后3条累积（含max监督128 index94）＋最长5910 index6；诊断不复演formal warmup，丢弃状态。240+TERM30；复用实证indexed标准CE/loader，A100Torch2.6/runtime明确区别CPU A8002.7。
- 正式条件：fresh原2epochs32calls/warmup2/标准CE/r16/seed，含两完整SHM→.100 ACK统一3600+30；原逻辑state恢复不重置时钟，probe状态不用。最终全部80fresh，只有最终epoch2质量，不选epoch。
- 外部R/research/patient-response-code-v1/。原始数据/模型/完整输出外存，main与旧patient分支未改。source/plan先push后唯一start；UNKNOWN查原作业，不盲重启。

### 2026-10-24 11:26 UTC — formal / F1 source prepared

- Dedicated helper prepared only `formal_train/`: source manifest `e112cd2b…`, trainer `9b8addda…`, plan `43db4c0c…`; 59 stdlib checks passed in 0.026066 s. Original indexed epoch loop and complete Adam/scheduler/RNG restore AST reused. No model run by helper.
- Owner prepared two-epoch / 3600+30 controller, per-epoch private SHM→.100 full SHA/fsync/reopen ACK, detached exclusive-lock local watcher, complete80 fresh F1 decoder and two fixed quality arms. Formal activation remains conditional on this route's actual 240+30 pressure result and measured cost.
- First A100 read-only deployment inventory returned UNKNOWN (no write/claim/start); next inventory returned explicit LUNA_OFFLINE at 11:23:44 UTC. Preserved both receipts separately. Unified ≥185 s cooldown until 11:26:49.308 UTC. No GPU work has started; original method/source/clock authorization unchanged.

### 2026-10-24 11:27 UTC — unique actual probe started

A100 recovered; all 13 package members SHA/fsync/reopen passed. A subsequent start inventory UNKNOWN was before local intent/remote claim; same inventory recovered once, no repeated model start. Actual job proc_6e7979ca-c61e-4cb0-9807-ce042f7fbfe3 / PID 129094 at 2026-10-24T11:27:21.190874469Z. Driver original TERM 2026-10-24T11:31:21.388561+00:00, KILL 2026-10-24T11:31:51.388561+00:00. Full proof in evidence/probe-start.json; no formal/quality yet.

### 2026-10-24 11:32 UTC — actual H51 PASS, fixed formal bound

Original probe ended 11:29:55.022 UTC, rc0 /153.633742 s; 12 forward/backward +1 real group8 Adam, 496 states, all grads finite/frozen base empty, three LoRA families changed. Max5910 and128 supervision covered. Peak allocated24,774,604,800/reserved26,724,007,936 B; actual GPU2 released. Two late observations UNKNOWN did not change this original endpoint. Exact small raw SHA refs saved; full evidence collection may follow without repeating probe.

Measured pressure82.430726 s, other native overhead69.915781 s. Frozen cost formula max(row/token projection)×1.15 +twice overhead +600s combined export/control allowance =2698.935144 s, below shared3600. A800 historical520MB exports56–60s are a proxy, not A100 network measurement; budget remains a hard cap, not a guarantee. Bound fresh2epoch controller entry9183521c; no checkpoint from probe.

A100 CPU/HTTP read window briefly handed to sequence owner while this local binding/source is pushed. Same-card GPU remains reserved for this route; fresh gate precedes formal start. Reliable exact-piece deployment inventories original bytes after UNKNOWN; source/target identity and all hashes preserved, no automatic model retry.

### 2026-10-24 11:44 UTC — exact source bridge and one managed entry

A100 transfer recovered through authenticated private .100 HTTP with600s TTL; capability stored private and redacted from receipts, never Git. The original460078B deployment archive SHA735543ba is unchanged: A100 original4 pieces reused,11 missing filled/fsynced; no existing different bytes overwritten. Post-transfer inventory then returned explicit LUNA, unified next window11:45:02.911.

New mechanical bootstrap entryd13841bc preserves original entry9183521c and all failure receipts. It reuses the exact original archive verify/preflight code strings, acquires an exclusive remote claim, then invokes the unchanged original epoch_driver. .100 fetch sources are staged first. One original3600 clock begins at the single managed bootstrap activation, conservatively including its source/fresh checks; epoch2 and exports never reset it. A100 no formal campaign/start yet. Changed local inspect exposes any bootstrap failure rather than silently waiting for a model that never began.

### 2026-10-24 11:47 UTC — unique formal activation and explicit read-only monitor recovery

Managed bootstrap proc_4e8346b7-b644-425b-9edf-8b323fc54ed8 /PID141053 accepted11:45:06.757009Z. Original shared campaign11:45:04.713781→12:45:04.713781 UTC, unchanged3600+TERM30. First detached watch158482 exited on explicit read-only LUNA at11:45:34.691; process stateZ and failure receipt retained, not unexplained lifecycle loss. No remote restart. Independent new watcher only retries the unchanged `inspect` for explicitLUNA/UNKNOWN with185s backoff; mutation ambiguity still stops. First recovery not before11:48:39.691. Model progress will be reported only after actual recovery. Owned authenticated .100 source HTTP was explicitly stopped after verified archive fetch; no GPU touched by bridge cleanup.

### 2026-10-24 11:51 UTC — actual original training confirmed

Read-only recovery succeeded11:48:47.771 UTC: original epoch1 driverPID141673 started11:45:58.308 and passed source/data/native token-mask/fresh GPU/base load into formal_training. Same original campaign/deadline, no second model. Recovery watcherPID158832 holds the exclusive lock. Next11:50:50 read returned UNKNOWN_OUTCOME; preserved separately from LUNA and backed off185s to11:53:55.245739. No assumption of remote failure and no repeated model start. Whole source/state/export chain remains fixed; F1 awaits both actual complete durable ACKs.

### 2026-10-24 11:58 UTC — original epoch1 progress and terminal backup preparation

Original read-only inspect11:56:14 succeeded: epoch1 step10/loss .07682358, peak allocated24,166,990,336 B; no completed epoch/ACK yet. Small independent read-only review found no concrete post-training inference/quality interface blocker, without running parser/model/quality. Prepared one private terminal bundle entry and reused prior .100 CPU-only fsync/SHA storage helper under this route namespace; not executed yet. Capability/transport intents, model tensors and held gold are excluded; actual private clinical output remains outside Git.

### 2026-10-24 12:01 UTC — first epoch complete, export-publication recovery

First epoch actually ended11:59:58.919894 rc0/840.611566s/16updates; GPU2 released12:00:00.368811. Full checkpoint exists in private SHM; persistent ACK pending. At12:00:32.536 explicit LUNA occurred while publishing the export entry, before any server start_process call. Original monitor158832 exitedZ and all receipts remain. Original entry7cd0dfda phase deadline12:10:32.215268 and overall12:45:04.713781 remain fixed. Single A100 cooldown until12:03:37.536. Prepared a unique managed CPU publication recovery that checks/reuses identical entry bytes and execs unchanged io_driver; no repeated model or export server, no new phase clock.

### 2026-10-24 12:04 UTC — same-entry server accepted, pure readiness read unknown

Unique recovery CPU server proc_827ccc3e/PID153761 accepted12:03:41.141091Z under unchanged600s phase. Next start-fetch only read server-ready, returned UNKNOWN12:03:54.348 before any fetch intent/publication/start. This is distinct from earlier explicit LUNA. No server replay. Independent monitor161802/watch-recovery-3 waits until12:06:59.348 then resumes that original read and unchanged fetch/ACK path; original phase ends12:10:32.215268, campaign12:45:04.713781. Both failed local monitors and original error receipts remain.

### 2026-10-24 12:07 UTC — read-only recovery classification

Root clarified first pure-read UNKNOWN may retry once after30s;185s applies explicit LUNA, repeated UNKNOWN remains conservatively bounded. Idle watch161802 had no child operation and was explicitly stopped, no remote process touched. New162090/watch-recovery-4 uses separate frozen monitor source3379e477, covers pure inspect/server-ready/export-read failures but never repeats unknown publication/start/ACK mutation. Actual12:06:01.548 ready-read returned explicit LUNA; next12:09:06.548 leaves85.7s in original export phase. No fetch has started; original server may continue while control is offline. Any actual phase failure remains separate from valid first-epoch checkpoint.

### 2026-10-24 12:11 UTC — original phase durable, epoch2 uniquely accepted

Original .100 fetch proc_bb6ac3d4 began12:09:17.227461 and completed ACK12:10:19.359279 (62.004988s), before phase deadline12:10:32.215268. All13 members/520,024,979B full SHA/fsync/reopen passed; ACKadc62307/source-copyf5ca5efe match exactly. Later local observation12:10:34 is not ACK creation time. Original server stop complete; no extra I/O recovery window used. Epoch2 proc_eacc8793-19f3-461b-aa25-39efadbafbc6/PID158377 accepted12:10:35.344602073Z, original whole deadline unchanged and~2069s remaining. It resumes the exact first state/ACK; actual full restore proof will be collected at training terminal, not inferred from start acceptance.

### 2026-10-24 12:13 UTC — bounded inference transport prepared only

Prepared inference/deploy_via_bridge.py by reusing the already-successful authenticated .100 private archive bridge in a new namespace. It binds the final inference archive only after both training ACKs, retains capability private/redacted, and executes the original inference operations.py deployment verification literal verbatim after same-SHA transfer. Source plus4 generated remote-code literals compile; no new remote action/model/quality. Native inference, generation, decoder and paired quality source unchanged. This avoids repeating fragile many-call A100 upload; whole archive and every member remain verified before original unique launch.

### 2026-10-24 12:31 UTC — complete training campaign, exact final ACK recovered, F1 bound

Epoch2 actually ended12:24:11.904241 rc0/816.317195s and released12:24:13.347342. Final adapter2c777e88/statef12d1b7e; complete32updates. Second .100 ACK5c8cfe2c completed12:28:19.908820 (56.740060s/520,025,589B), all14 SHA/fsync/reopen. Source-copy publication and a conditional recovery returned UNKNOWN; subsequent pure read found identical ACK already present, adopted without content write. We cannot infer which unknown request published it. Original server stop positively confirmed, then campaign closed12:31:13.206066 (2768.492285s), within original12:45:04 deadline. No repeated NN/transfer and no extra export window.

F1 binder passed both complete ACKs/source copies/original native80 and final adapter: entrybeeaa580, all80fresh/input14848/gen1600 and fixed two-arm quality unchanged. Transport entry separately binds the identical archive and original deployment verifier; source and small binding evidence pushed before unique inference start.

### 2026-10-24 12:33 UTC — unique complete80 inference accepted

Bound archive d3b2073a/469,434B transferred through authenticated private bridge. Original deployment verifier passed11 members/15 pieces/SHA/compile/fsync on actual UID1045, planbd88ca5b. Original sourceBridge was stopped immediately after verified fetch. Unique inference proc_a84ae36b-dac5-4e1b-8898-0ed1f22e7228/PID173877 accepted12:33:05.102124409Z. All80 fresh,14848input/1600greedy and strict PT inverse/raw-first whole-row abstention;1200+TERM30. Actual driver/GPU/response progress is separate from acceptance and will be read once; no repeated start.

### 2026-10-24 12:36 UTC — actual inference and original engineering evidence

Actual driver12:33:05.317552/child173895 binds TERM12:53:05.317552+30 and sourcea0794470. Read12:35:01 confirmed9/80 canonical terminal rows, no inferred score. Original training15 metadata members collected zip06dfa58c/20,209B; exact logical optimizer/scheduler/RNG resume proof passed. Original H51 full53-file collection10aeab87 now all SHA verified; one failed pure inventory UNKNOWN preserved, no probe repeated. Large arrays/clinical rows remain private; safe summary refs in evidence/complete-engineering-evidence.json.

### 2026-10-24 12:42 UTC — full80 read-only offline window

Actual inference reached52/80 terminal records at12:40:29, no partial score. Read12:41:38.104880 returned explicit LUNA; same mount recovery not before12:44:43.104880. Original proc_a84ae36b and12:53:05.317552+30 unchanged, no repeated generation. Recovery pointer D/read-only-offline-124138.json; original raw outputs remain on source until final collection.

### 2026-10-24 12:48 UTC — all80 actual terminal, original collection recovered read-only

Unique full80 job actually ended12:44:29.153898 rc0/683.960027s, 80fresh/0cache,79strict and1whole-row abstention; GPU2 released12:44:29.263048. Peak allocated22,280,655,872B/reserved41,437,626,368B. Original collection manifest request returned UNKNOWN; no model repeated. Independent recovery reads and verifies the already-existing original253-member zip71db687e/93,245B, never repacks remote output. Partial reads reuse SHA-verified bytes. Two quality endpoints remain unexecuted pending complete collection.

### 2026-10-24 12:49 UTC — complete fixed quality, positive candidate

Original253-member collection71db687e complete, including one30s bounded pure-read recovery; no remote repack/model repeat. Both full predictions froze at12:49:10.675487 before exactly2new/0old v2. Primary coded-alone Score.6866189430196797 (+.010072588357vsN;+.000279961748vs historicalfullsupport), diagnostic N-union-coded.6822756102237818 (+.005729255561vsN). Both pass; selectcoded-alone. Primary relation322TP73FP198FN/micro.703825136612/macro.627711699678. One duplicate-response-code whole-row abstention retained; no repair. All1189entities/nonrelationbytes preserved. Final review entry externalB/final-review-entry.json; no further GPU/quality/F0/dev/official. Private small backup pending.

### 2026-10-24 12:52 UTC — F1 private backup closed; independent F0 authorized

F1 terminal.tar.gz1,694,368B/b8afef77,202members actual.100 allSHA/fsync/reopen at12:51:43.444715. Source/code, private train/held inputs, allraw predictions/metadata included; heldgold/capability/modelstate excluded. Both520MB completeepoch states separately durable. F1-final-handoff.json links immutable33293985reviewentry and actualbackup. Root separately authorized same2c777e88 train32 model on original fulltextF0 58, native1200+30, two fixed endpoints vs existing SFT48 N .696603974744; candidate ancestors may train onF1 but must excludeF0. It is an actual candidate comparison, not a single-variable encoding ablation or uniformtrain32 pipeline.

### 2026-10-24 12:53 UTC — complete F0 candidate ancestry and fixed CPU source

F0 original58/16docs/47patients/415concepts/1166pairs are exact descendants of CRF+Sap+LR using train48 d87bf189, all48 disjoint held16 ec38c135. SamePT transform keeps fulloriginaluser, schema/order/clinicalIDs and pureinput. Codedhead32 is subset of train48; comparatorN usesold48, so actual candidate comparison rather than isolated encoding effect. One initial local identity-set check correctly failed because original -1 entity maps to literal text via original association_keys; source/error preserved and exact legacy function reused, no data change/model/quality. NativeCPU120+5 on A100 existingruntime prepared, noGPU/model load.

### 2026-10-24 12:55 UTC — F0 CPU start unknown during explicit offline

A100 CPU5members/10pieces verified deployment plan1d136fad; .100 original shortHTTP stoppedtrue. Unique CPU120+5 start request returned explicitLUNA, nojob returned; originalintent/error retained. Readonly recovery follows recovery-offline-boundary.json before any conditional action, no repeated CPU/model start. F0 GPU has not been requested. Same inference generate/loading/decode calls and entire raw-first sampling-loop AST exactly match F1; unchanged PT decoder. Only F0metadata/cap and required longest-formal-first scheduling differ.

### 2026-10-24 12:58 UTC — original F0 CPU remains unobservable

The scheduled single read12:58:27.406877 again returned explicitLUNA, not model/CPU failure. Latest independent boundary13:01:32.406877; original startintent remains, no model or CPU restart. New recovery reader uses the maximum of preserved offline boundaries. F1 positive result and durable archive unaffected.

### 2026-10-24 13:03 UTC — actual A100 restored, sole CPU reentry accepted

13:01:59 pure-read UNKNOWN used one30s bounded retry. Actual13:02:43.585 inventory showed original F0 CPU namespace no driver/native/terminal and no cwd process. Exclusive unchanged driver/command/env reentry accepted13:03:08.618951:proc_25ba1090/PID194503,120+5 CPU only. Delayed original request cannot pass the same exclusive driver claim; original LUNA intent retained. No fallback CPU launched and no GPU started.

### 2026-10-24 13:03 UTC — actual complete58 CPU PASS and final binding

A100 CPU finished13:03:21.831548 rc0/13.125329s (native12.197954s), complete7-member zip08d51e09 recovered. Original505090tokens/max16245→coded516365/max16542 uniqueidx30, no truncation; all58 Auto/nativeIDs/template/decode exact, clinicalmap bijective. Native context262144≥18142. Explicit engineering inputguard rises14848→16896 (ceil512), greedy1600 unchanged. Bound sameadapter2c777e88, 58fresh/0cache, longest30 first as one formal row, then old ascending length order. Source/plan/quality immutable and pushed before GPU activation.

### 2026-10-24 13:06 UTC — original unique F0 GPU start confirmed after UNKNOWN

Archiveed66def0/364,005B fetchUNKNOWN resolved by read-only exact11member/12piece verification plusfsync; no retransmission. Original.100HTTPstoptrue. Unique start returnedUNKNOWN but actualdriver196256 began13:05:44.871579; child/PGID196277, TERM13:25:44.871579+30, plan8b6087cf. ManagedID unavailable, not fabricated. Originalinspect confirmedactualdriver0/58 duringloading; no secondstart. F1immutable independentreview187cfa0f passed allchain, pointer only added without rewriting202bundle.

### 2026-10-24 13:16 UTC — complete F0 strict58 and fixed positive quality

Original GPU finished13:14:20.722876 rc0/515.979612s and released13:14:20.836959.58strict/0abstention/58fresh/0cache.187member zip5de9e767/69,494B complete; last3958B UNKNOWN resumed once using prior two verified pieces. Two complete predictions froze13:16:41.258865→2new/0old v2 .583918s. Primary .700893777351 (+.004289802607vsN), union .700560731214 (+.003956756470); primary selected. Bothmicro decline, macro improve; both belowstrongB0/root2of3. Pureprimary9e6b5840 supplied to root independentfusion. F0-final-review-entry.json immutable; smallbackup pending. No further GPU here.

### 2026-10-24 13:19 UTC — F0 durable backup complete; new CPU-only scope

Original89-member864,521B archive4c009372 received .100 allSHA/fsync/reopen13:19:07.162117; F0-final-handoff.json binds immutable964d reviewentry and actualACK. F0 complete positivecandidate, fixedoriginaloutputs unchanged. Next authorized120+5CPU tasks: full64 shortnormal265+originaldev fulltext64 (samePT, freshfull64 futuremodel only), then independent cyclic1 responsemapping80. No newGPU/fit/quality; originaldevgold never read for preparation.

### 2026-10-24 13:29 UTC — full64 complete structural CPU readiness

Newnamespace full64: originalshort265 train64/173patients/1908CID/5328pairs; originaldev fulltext64/16docs/36patients/466CID/1006pairs. Entire originaluser UTF8/metadata/targetinverse exact; existing full-report64 reconstructed exactly using frozenappend, no oldbudgetrenderer. Training/candidate ancestors excludeoriginaldev16. Structuralba3acc00; packagecc6d17ef629,576B/7files. Only120+TERM5 nativeCPU authorized; freshfull64GPU/quality remain unapproved. Old C32/F0 unchanged.
