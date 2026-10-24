# 现有训练循环的双轮参数接口

此目录是原 grounded `198d1de2` epoch loop 的机械接线，复用已实跑 indexed2ep 的完整状态恢复与 SHM→.100 持久 ACK 协议。两路使用同一源码，不建立第二套优化循环；具体输入、计划、输出与预算分别冻结。

- `grounded-full64-v1`：173行、64篇、2轮44更新、warmup4、尾组5。
- `canonical-name-train32-v1`：87行、32篇、2轮22更新、warmup2、尾组7；实际token统计由该路固定计划给出。

两路均为原r16、lr1e-4、seed20261023、batch1/accum8、BF16/SDPA、标准逐行平均indexed CE。内部forward、CE、除实际组大小及backward的AST与原198d精确一致。配置不会自动改变样本、截断、loss、精度或allocator。

新完整native输入证据和新形状压力证据缺一不可。`train.py`接受 `--plan`、原绝对deadline、`--stop-after-epoch 1`，首轮完成即退出；只有同namespace最新完整epoch001和精确持久ACK才接受 `--resume`。保存/恢复adapter、Adam、scheduler、Python/NumPy/Torch/单CUDA随机状态及参数顺序，恢复随机状态位于所有模型/优化器构造之后。完整逻辑状态digest可核；不声称新GPU续训逐位等价已经实测。

SHM的fsync不是持久存储。每轮通过原受限HTTP传到.100自有目录，全部文件最终文件名重新打开SHA、文件与目录fsync并产生ACK；源端收到ACK才续第二轮或允许推断。`export-ack-epoch-*`不混入epoch目录glob。原绝对campaign时钟包含两次完整传输，不能为恢复重置；正式预算由root另行授权并写不可变plan。

当前仅CPU准备：controller派生源可编译，实际原forward循环AST已对照；GPU/恢复/持久ACK的新actual尚不存在。`cpu_state_interface.py`仅用微小CPU参数验证真实Adam/scheduler/CPU随机流恢复；CUDA字段结构是显式占位，不能当GPU RNG回放证明。原数据与新CPU状态合同由各独立运行plan绑定。新正式计划未激活之前，不能把这些脚本视为已执行训练。

源/祖先身份见 `source-increment.json`，旧实际桥接库存见 `../full64_readiness/persistent-bridge-readiness.json`。两路owner只能修改自己的输入/计划；此目录由grounded owner维护，其他线路只读引用固定SHA。
