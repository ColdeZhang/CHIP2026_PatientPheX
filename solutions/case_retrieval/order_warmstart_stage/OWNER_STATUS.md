# 严格32文献共同旧权重起点：新两轮顺序对照

**已完整负结案，禁止重启旧controller或重复评分。** 旧基线Score .660942003；新O/O .658421563；新R/O .645528610。两臂没有收益，不替换旧adapter、不迁F0/原dev、不上传官方。见[完整结果、范围与资源](RESULT.md)和[结果索引](../evidence/round2/order-warmstart-paired-result.json)。

两臂共同oldSFT32 epoch2 adapter、新AdamW/RNG/LR，各124行×2ep、248曝光/32calls；O/O与R/O的临床/标签语义保持，仅新第一轮候选/CID排列不同。原32文献严格隔离不变。相对旧基线同时增加训练及重置新stage，不能把下降单独归因于训练时长；此次训练增强负结果不推翻旧测试时逆序并集。

共享3900+30秒，全部NN及4次checkpoint导出实际3669.545秒完成。双80/587合法CID、双fullpred共同freeze后各唯一一次v2，旧指标复用。全部4个epoch完整adapter＋优化器/RNG在.100持久保留，源/大小/SHA回读通过；没有partial评分或延长截止。

外部根：`/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry`。原O/O epoch1在`formal/`；ack文件名工程故障修复后的其余三checkpoint、完整响应和质量结果在`formal-ack-recovery/`。`formal/execution-state.json`已明确complete-by-recovery；初始与恢复前状态快照仍保留。原`run_paired.py`、修复`run_paired_ack_paths.py`和`recover_ack_name.py`均是历史可复现源码，**不能再次执行**。

最终GPU2实际释放02:25:04.530Z，receipt `856903f5`，已交sequence独立实验使用。任何后继GPU任务需要重新授权和实际占用核验，不能把此历史空卡收据当当前状态。本路线当前无GPU进程/监控；本地原恢复monitor48145与唯一endpoint13222均已exit0。MCP未提供模型进程远端exitcode，不混用本地rc。

已核入口：CPU7.006587秒/CUDAfalse；统一entry `71b9c9b2`、最终绝对预算/intent增量 `39ad3212`、ack恢复 `baa8be3b`。最终实际终点独审待回收；无需重跑旧CPU/GPU来等待审查。准备时期的源码、plan和小传输证据原样保留在本目录与既有Git历史。数据、权重、完整预测和日志继续留仓库外。
