# 工作台账

当前：explore/patient-response-code（base main cd4086b）；H51已实测通过，首轮16updates已rc0完成，原导出服务已接受/待ready恢复。原campaign截止12:45:04.713781 UTC，恢复监控PID161802（watch-recovery-3，12:06:59.348后查询）；未开始第二轮/F1/质量。只改本路线。

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
