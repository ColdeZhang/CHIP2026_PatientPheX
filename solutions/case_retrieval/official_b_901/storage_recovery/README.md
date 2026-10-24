# A800磁盘空间恢复：旧state迁移附录

07:01实际自有项目盘可用0B，完整只读库存未发现足够256MiB且已核副本的普通安装包。按root显式授权，将本路线已完成full64 epoch002的原`training_state.pt`及checkpoint/config原字节经私网复制到`.100`；保留原checkpoint metadata不改。这里只搬文件，不加载张量、不训练或评分。

服务端只读三个固定SHA文件、限制接收IP与独立临时URL前缀，单次60秒窗口；接收端每件fsync、重开全部SHA，父目录链fsync后落独立receipt。只有上述真实receipt经回读且原源文件身份仍匹配，才移除A800唯一指定state副本并记录实际free。原adapter权重已在`.100/full64-dev-transfer/adapter-epoch002`持久；这次附录明确补state locator，不宣称搬走了整个原目录，也不改旧冻结清单。

实际唯一复制 38.345 秒，3 件原始字节全部 fsync／独立重开 SHA；`.100` receipt `731ba732` 已实际回读。2026-10-24 07:10:59 重新核源 UID／mtime／size／SHA 后，仅移除 A800 原 `training_state.pt`（38b9877a，346,667,053 B），源目录 fsync，原 checkpoint/config 不变；可用空间 0 → 338,419,712 B。恢复定位为 `.100/research/storage-recovery-20261024/full64-epoch002-from-A800/training_state.pt`，配合同机已持久 `full64-dev-transfer/adapter-epoch002` 的 1a703b 权重。见 `result-pointer.json`；旧冻结清单不修改。官方 raw／预测仍留仓库外。
