# Full64 响应编码组件：当前执行入口

当前在 `explore/patient-response-code`，只推进此独立 A800 namespace。旧 A100 正式方案从未创建 campaign 或发送训练启动，已关闭；其有效 H51 和部署 UNKNOWN 证据保留，不调用旧 bootstrap。

- 仓外根：`/workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-a800-v1`，下称 `B`。
- 正式控制器：`B/controller`。唯一 managed job `proc_81366e65-6b36-4332-8517-06fe8a340baa`，epoch1 driver `49955`；真正状态以原 `campaign.json`、`epoch-*-job.json`、`inspect-*.json` 和 ACK 为准。
- 原共享截止：`2026-10-24T15:23:52.618929Z`，TERM 后 30 秒；两轮及两次完整持久导出均包含在 5400 秒内。
- 独立本地 watcher `178713`，PPID 1 / SID 178713，实际独占锁已核。先读本地 `local-watch-events.jsonl` 与 PID 身份，勿重复并发远端轮询；任何 UNKNOWN 先恢复原意图，不能重训。
- 配方固定：short265 / train64，fresh 基座和 r16，2 epochs / 68 updates / warmup4，每轮尾1，标准 indexed CE。原完整329 native CPU合同复用，实际 A800 运行再次逐 input IDs/labels 核验。0 originaldev gold / 0 本线评分。

两轮与两个完整 ACK 到齐后，继续原已授权链：

```bash
python solutions/patient_response_code/full64_a800/controller/entry.py collect-training --controller B/controller
python solutions/patient_response_code/full64_a800/controller/entry.py bind-inference --controller B/controller --inference-directory B/originaldev64-inference-v1
python solutions/patient_response_code/full64_a800/inference/operations.py deploy --directory B/originaldev64-inference-v1
python solutions/patient_response_code/full64_a800/inference/operations.py start --directory B/originaldev64-inference-v1
```

示例中的 `B` 须替换为上方实际绝对路径。唯一64 fresh使用原完整输入，cap13824、gen1600、native greedy、严格整行弃答及 raw-first。推断独立1200+30，不能复用旧32权重缓存。随后仅观察同作业，完整后运行一次 `operations.py collect`，再运行 `inference/assemble_component.py --inference B/originaldev64-inference-v1`。

纯 C 输出保留旧955实体底座的非关系字节，仅向 root 提供关系；root 固定融合保留自己的1061实体。原16文档、36患者键及身份对象已与901逐值核对一致，不重新推理来更换实体壳。

`finalize.py --directory B` 是新准备的本地收尾入口，尚未执行：只在两完整ACK、全状态恢复收据、完整64终点及纯组件已存在时，生成单份 `final-review-entry.json` 和私有内容包。它没有服务器、模型、gold 或评测入口。内容包还须沿现有私有持久传输完成实际全SHA/fsync/reopen ACK；模型与optimizer本体已由两份训练ACK持久保存，不再次打包。

发现原运行/回收入口的机械异常时，保留原错误和源SHA，只修接口/传输，不更改数据、方法、原截止或重新生成响应。恢复监控使用独立 recovery 意图，先证明旧 watcher 已死且没有双owner。

完成本地封包后使用 `python solutions/patient_response_code/full64_a800/store_terminal.py --directory B/terminal-backup`，复用固定SHA的既有私有传输实现，仅切换独立full64目录。尚未执行；上传UNKNOWN时先核原同字节分块，不重复盲写。真正stored-receipt到齐才称持久备份完成。

两轮ACK后再运行一次 `python solutions/patient_response_code/full64_a800/verify_export_closure.py --controller B/controller`，只核两个原导出HTTP的精确进程身份是否消失，不发信号、不查GPU；避开原watch在途调用。若尚在则保留实际状态，不能凭managed stop回执称完整关闭。终态包绑定此原收据。
