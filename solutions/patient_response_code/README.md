# 独立患者响应编码

原normal短124训练＋全文80推断，保留所有原clinical PID、user证据/markers/CID，仅明确新增响应PT编码和可逆目标。F1已完整：coded-alone .686618943020，N∪coded .682275610224，两者胜强N；按事前规则选单模型候选。两轮训练/两完整ACK、80fresh和两固定质量均完成，202件小包已持久。root另授权同权重完整F0复验，现独立准备。详见[台账](WORKLOG.md)与[固定质量](quality-policy.json)。

接手先读 [ready.json](ready.json) 的真实作业与监控指针，再读外部 controller 的原 intent、events、ACK；不得按源码重启已运行的模型。当前完整状态保存在私有 SHM，并按每轮约 520 MB 迁移到 `.100` 持久目录；原绝对时钟不重置。

两份完整 ACK 后，使用 `inference/bind_after_training.py` 绑定最终适配器；部署可用 `inference/deploy_via_bridge.py` 的 `stage → serve → ready → fetch`（`--controller` 指向新 inference 目录），传输后仍执行原 `operations.py` 的逐成员校验。唯一启动、观察、收件沿原 inference operations；任何未知启动先恢复原状态。

全 80 终态收齐后执行 `quality/final.py`，两完整端点先共冻、各一次 v2；不会自动迁移 F0／原 dev 或上传官网。最后用 `controller/build_terminal_bundle.py` 与 `store_terminal.py` 备份私有输入、响应和证据；权重／优化器在独立持久检查点中，held gold 与桥接 capability 均不进入小包或 Git。
