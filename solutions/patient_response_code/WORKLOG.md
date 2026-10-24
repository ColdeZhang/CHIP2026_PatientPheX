# 工作台账

当前：新分支explore/patient-response-code，base main cd4086b；正在冻结并推送新H51，尚未启动GPU。只改本路线。

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
