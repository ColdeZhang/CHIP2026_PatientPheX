# full64 原dev固定单臂操作源码

本目录保存外存 `qwen35-preparation/full64-dev-transfer-preparation/` 的精确源码快照；执行时从该外存目录运行，运行数据、权重、raw和完整日志不进Git。已有proc/job或冻结产物必须先恢复核验，不能重新开作业。

训练完整epoch002/68steps及实际SHA到齐后，`prepare_and_transfer.py`只复制两adapter文件并关闭临时源。GPU由root明确调度释放后，`run_fixed_arm.py`默认只冻结计划、`--launch`才启动900秒+30秒宽限的单新臂。同A30、原64提示/955实体/decoder/parser均固定。

推断完整后，以既有CRF venv运行 `evaluate_fixed_arm.py freeze` 先装配和冻结预测，再 `evaluate` 仅评新臂一次。旧三个对照按真实SHA复用；无金标调参、无官方上传。源码的已冻结SHA和外存接口见证据索引；不得边运行边修改。
