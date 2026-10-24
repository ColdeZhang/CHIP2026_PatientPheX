# A800磁盘空间恢复：旧state迁移附录

07:01实际自有项目盘可用0B，完整只读库存未发现足够256MiB且已核副本的普通安装包。按root显式授权，将本路线已完成full64 epoch002的原`training_state.pt`及checkpoint/config原字节经私网复制到`.100`；保留原checkpoint metadata不改。这里只搬文件，不加载张量、不训练或评分。

服务端只读三个固定SHA文件、限制接收IP与独立临时URL前缀，单次60秒窗口；接收端每件fsync、重开全部SHA，父目录链fsync后落独立receipt。只有上述真实receipt经回读且原源文件身份仍匹配，才移除A800唯一指定state副本并记录实际free。原adapter权重已在`.100/full64-dev-transfer/adapter-epoch002`持久；这次附录明确补state locator，不宣称搬走了整个原目录，也不改旧冻结清单。

实际唯一复制 38.345 秒，3 件原始字节全部 fsync／独立重开 SHA；`.100` receipt `731ba732` 已实际回读。2026-10-24 07:10:59 重新核源 UID／mtime／size／SHA 后，仅移除 A800 原 `training_state.pt`（38b9877a，346,667,053 B），源目录 fsync，原 checkpoint/config 不变；可用空间 0 → 338,419,712 B。恢复定位为 `.100/research/storage-recovery-20261024/full64-epoch002-from-A800/training_state.pt`，配合同机已持久 `full64-dev-transfer/adapter-epoch002` 的 1a703b 权重。见 `result-pointer.json`；旧冻结清单不修改。官方 raw／预测仍留仓库外。


为覆盖两路推断期间共享盘再次上涨，按后续授权额外迁了本路线已完成 full64 epoch001、旧 relation48 epoch001/002 三份 state，各约346.7MB；9个 state/checkpoint/config 原件分别持久重开并回读 receipt（18f6d929／ea11b265／10481f9d）后，于07:15:36仅移除对应3份原 state，free327,258,112→1,367,248,896B。各新 `.100/research/storage-recovery-20261024/` locator 与完整SHA见 `additional-result-pointer.json`。这里仅移动state与metadata/config，原基座/adapter恢复依赖仍按不可变checkpoint定位，不冒称独立完整checkpoint包。额外迁移首attempt触发已有HANDOFF H33（A800 system Python不支持Path.is_relative_to），在serve绑定和接收mkdir/下载前失败，0copy/0delete；旧plan/失败保留，v2只换路径guard并按原每份60秒窗口实际成功。无需新增同义坑条目。
