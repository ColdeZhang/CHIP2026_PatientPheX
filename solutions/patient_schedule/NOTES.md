# 笔记与防坑

- 开始任务读main HANDOFF；本控制适用H44目录与ACK分离、H48原生tokenizer身份、H49 raw先存、H50 evaluator依赖、H51实际梯度累积状态、H53共享盘容量。独立路线只维护本文件，根HANDOFF由root汇总。
- 旧两轮终点LR0，不是第三轮暂停点。fresh4必须重新初始化基座/LoRA/Adam，从开始使用4warmup/64calls；第2轮也不等于旧模型。
- 原native CE A800 reserved80.753GB接近整卡；CPU源检查不能证明新第3/4轮显存成功。投影优化本轮未采用，避免扩大方法差异。
- 仅继承Python可执行文件会漏TF5.20.2 overlay。controller固定完整PYTHONPATH、离线参数和唯一UUID，运行时再核真实版本/native模型文件，无FLA。
- 四份520MB放1.5GB项目盘会撑满。SHM只暂存，.100全文件SHA/fsync/reopen回执是下一轮的必要数据依赖，不能用本地checkpoint写完冒充持久化。
- `training_complete.elapsed_seconds`是第四个进程耗时；campaign从第一轮start至第四ACK总耗时另记，不能用最后阶段当总成本。
- 本轮质量保留严格原schema；若失败，raw IDs和文字已持久保存，但不能把缺失回答补成空集合。恢复是否改政策由新的明确记录决定。
- 公开两轮的native tokenizer/监督合同真实证据已复用，本轮没有重复分词或新的Torch/GPU恢复证明。CPU小状态只核逻辑库存/哈希/拒绝错误；真实完整恢复须以后原运行记录证实。
- 匹配强N∪P2首次CPU基线负，不等于四轮一定负；同时不允许只比旧weak triple小升就声称globalbest。最终两固定臂按同强N及匹配对照解读，macro不额外盖过总Score目标。

- 本地监控进程不是远端GPU作业：10:16原watch PID143721/exec73046已失，但远端epoch1正常至step10，原watch没有failure，不能猜成训练故障或指定退出原因。恢复先核原PID确实不在及无第二owner，再独立意图/日志，用新session、文件stdio、PPID1和锁接续旧状态；不重训、不重置clock。必须复核真实后继tick，单次launch返回PID不保证长期存活。
- .676546是固定normal-fulltext强对照，非所有历史F1完整组合最高；public-new LGB支持另有.686338981。胜本门仅称可融合候选，主/诊断仍保持原冻结两臂，不新增第三臂或改变成功条件。
