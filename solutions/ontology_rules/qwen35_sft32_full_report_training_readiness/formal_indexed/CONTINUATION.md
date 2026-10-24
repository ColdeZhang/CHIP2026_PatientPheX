# 当前继续入口（2026-10-24 08:53 UTC）

root同意owner暂转idle；本地watch及原远端训练继续，不停止或重启。当前第二轮原作业 `proc_e0b1196d-5c9f-4f40-aae4-728f03f48af8` / PID9466 已进入 formal_training，第一轮完整持久ACK已核。第二轮ACK尚未到，无推断或质量。

- 本地watch PID130118、原exec session11114；源 `controller/watch_authorized_campaign.py`。08:53实际进程存在、failure不存在，最近08:52:06原inspect完成。它只管理原两轮/两ACK，完成后自行结束，不启动F1。
- 外存根 `R=/workspace/patientphex-runs/round2-20261023`；原控制目录 `Q=R/research/qwen35-fulltext32-indexed-formal-v1/controller`。root可只读 `local-watch-events.jsonl`、`local-watch-failure.json`（正常应不存在）、`epoch-002-export-ack.json`、`campaign-complete.json`。若UNKNOWN/失败，先核原intent/job/真实状态，不重start、不重新计算、不要重启watch（其exclusive intent仍保留）。
- 原绝对deadline1792837648.873858，8400+TERM30覆盖两轮与两次导出；不可改起点。第一轮里程碑见 `epoch1-milestone.json`，checkpoint/Adam/scheduler/RNG真实恢复，旧probe状态不用。
- 两完整持久ACK与campaign-complete齐后，followup唤醒本owner。先运行 `f1_inference/bind_after_training.py`（无参数）把实际adapter绑定到 `R/research/qwen35-fulltext32-indexed-F1-full80-v1`，保存并push实际计划小指针；`evaluation_source/prepare_disabled.py`绑定仍禁执行模板。再按原 `operations.py deploy/start/inspect/collect --directory <该目录>`，fresh资源核后唯一全80fresh1200+TERM30，无旧adapter缓存。
- 完整80终态后才用 `evaluation_source/bind_completed.py` 绑定 terminal 和 evaluation-template，接着原 evaluate_full80 一次。primary新alone、diagnostic旧短SFT32并集，共冻两完整pred→各1v2/旧0；强基线.676546354662594，Score严格升才留，macro报告。保留1189实体/所有非关系字节。没有自动F0/dev、额外GPU或官方提交授权。
