# 同旧 SFT32 的顺序增强新阶段：源码与 CPU 合同

**建议与当前状态：采用共同已训练 oldSFT32 epoch2 适配器，另开两轮优化阶段；本地源码合同通过，尚未运行新模型或 GPU。** 这是旧2轮权重后继续学习的受控比较，不是 fresh4epoch，也不是加载旧 Adam/RNG 的精确连续训练。当前 MCP 离线，不尝试服务器操作。

## 来源可以复用，旧 resume 不可直接复用

旧 checkpoint `5cecc79c…` 与原 run manifest `5535d33a…` 的 fingerprint 完全相同，明确 strict train32、124样本、2epoch/32calls、smoke_only=false。其训练/排除文献集合与 `356109e9…` 准备记录一致；候选由两个16→16折生成，排除F1与outer0。

源适配器权重 `4f4815b6…`、配置 `96564441…` 与最近单条 thinking 运行的实际文件核验收据一致，15个base文件库存也全相同。本检查未重新读取服务器权重，仅核已有冻结收据；新入口运行时必须再次核实际适配器两文件SHA和完整配置。原准备中的 `fresh_base_required=true` / `old_adapter_continuation_forbidden=true` 原样保留，它们描述当时的 fresh 基线；本次新协议显式允许同一个 strict32 终点初始化，不引入旧48/full64模型。

实际调用原 `validate_checkpoint` 对旧epoch2会得到“不是未完成的两轮训练”错误。因此不能把旧epoch2塞进 `--resume`。新入口把 `--warm-start-checkpoint` 与自身阶段的 `--resume` 分开：前者只加载已核适配器，后者只允许本输出目录内最新、完整的新epoch1。

## 最小改动与保持部分

`train_order_warmstart.py` 从冻结旧入口 `08fc142c…` 派生，`derive_trainer.py` 可确定性重建，`trainer.diff` 保存完整差异。

- base/原helper/LoRA配置、causal loss及microbatch循环、原数据隔离、梯度裁剪、优化器和checkpoint原子保存保持。9个关键纯函数AST及原loss microbatch主体、原自身resume的Adam/scheduler/RNG恢复主体均精确相同。
- 初始新阶段使用 `PeftModel.from_pretrained(..., is_trainable=True)` 载入同旧适配器；禁止读取旧 `training_state.pt`。核实际r16/alpha32/dropout.05、FP32 LoRA及43,278,336可训练参数；base/head冻结和三类LoRA-B真实更新检查保留。旧B已非零，不要求零初始化。
- 载入模型及适配器之后重置相同Python/NumPy/Torch/CUDA seed，创建空AdamW及新32步scheduler。初始lr0、warmup2、31次非零lr调用，使用的末lr为1e-4/30，末scheduler降到0。
- 两臂只改变第一轮候选/目标共同可逆顺序：control O/O、augmentation R/O。第二轮都O；每轮124同原样本ID顺序，每8累积、尾4单独flush，共32calls/248曝光。总tokens805188/805190，监督tokens均14340；原全串分词与双逆证据沿用已冻CPU准备，不重新分词或读取F1标签。
- 相同起始seed不等于逐步dropout mask必然相同：逆序导致少数行token长度变化，可改变后继随机消耗。保留相同算法/seed/初始状态，不声称浮点或随机mask等价。

## epoch 导出边界

复用原生完整epoch退出和自身恢复，不增加GPU持有状态的等待框架。

1. 新输出只能 `--stop-after-epoch 1` 启动，完成epoch1、保存全Adam/RNG及adapter后退出。
2. owner将完整checkpoint以及 `_run/run_manifest.json`、源码、epoch1进度复制到 .100 持久目录，逐文件fsync/SHA/大小核验，生成导出ack。
3. 只能以同新输出的 `--resume epoch-001 --stop-after-epoch 2 --export-ack ...` 恢复。旧fingerprint、完整checkpoint所有文件和导出ack均必须通过。阶段操作参数resume/stop/ack路径不进入语义fingerprint；两臂的arm仍进入，不能串用检查点。
4. epoch2结束后由owner继续持久导出，成功后才进入下游推断。训练入口本身不启动推断或评测。

ack采用owner约定：`status=persistent_checkpoint_verified`、checkpoint SHA、epoch/step、source checkpoint绝对目录、.100 mount ID、persistent目录、`files`大小/SHA及run manifest SHA。`files`必须包含原checkpoint成员和checkpoint.json、`_run/run_manifest.json`；安全的 `_run/` 附加文件也按当前本地字节核验。不要把resume前会更新的GPU preflight作为恢复身份；当前owner导出清单只含manifest、源码、进度。

本地11个ack反例/正例均通过：错checkpoint/epoch/step/mount/source目录、缺state、错大小、错manifest、缺run manifest、不安全路径全部拒绝。另检查初始不暂停、无ack恢复、错误stop epoch均拒绝。ack只是对已完成持久导出的收据验证，真实远端持久性由owner的传输核验提供。

## 成本与未验证边界

新训练量约为原fresh4epoch双臂的一半，新增2轮每臂805k tokens，与旧SFT32两轮规模相同。每epoch退出后恢复导致两臂合计额外两次base加载，需计入owner总预算；本子任务未给实际GPU吞吐保证或决定总cap。

最后 `training_complete.elapsed_seconds` 明确标注 `current_invocation_only_not_full_stage`，`full_stage_elapsed_seconds=null`，不能把第二段进程耗时冒充完整新2轮。owner按两段开始/结束及epoch记录合计，包括加载/导出开销。

纯标准库CPU合同实际约0.034秒，未import torch/tokenizer、未加载模型、未访问网络、未跑训练或任何质量评分。服务器真实CPU入口/完整LoRA加载与最终监督质量尚待后续执行和独立复核；本报告不构成模型数值或质量PASS。
