# 响应患者编码：固定短上下文 SFT32 训练核心

训练仍使用原 32 文献、124 条候选记录和标准 indexed causal CE。原 clinical user 整段 UTF-8、候选和临床患者 ID 保持；system 输出约定、末尾 PT→临床 PID 映射表、答案患者值编码改变。逆映射保持原答案语义及序列化，但 **CE 监督 token 已由 7170 变成 8138**，不能称目标 token 不变。

`train.py` 最小改编自已实跑的 ontology `formal_indexed/train.py`，来源和准确 SHA 见 `training-plan.json`。本次输入 f73cb24f，encoded 15832a38；每轮 427560 tokens，最长 5910，上限 6144 仅用于拒绝，不截断、不填充到上限。r16/alpha32/dropout .05、seed 20261023、AdamW 1e-4/wd .01、clip 1、2 轮/32 更新/warmup 2 保持；每轮 15 组×8 加尾组4，按真实组大小归一化。

新正式运行环境固定 A100 Torch 2.6.0+cu124、TF 5.20.2、PEFT .21.2、tokenizers .23.1，native SDPA/GDN。已完成 native CPU 计数来自 Torch 2.7.1；正式入口会逐行再次比对已冻结 input IDs、labels SHA，不能把旧 CPU 环境写成本次 GPU 环境。indexed head/selected CE 辅助源保持原字节；共享运行环境不改。

两轮共用 **3600 秒 + TERM 30 秒**外部截止，包含每轮完整 checkpoint 的私有 SHM→持久盘导出。第一轮必须 `--stop-after-epoch 1`；第二轮只有在完整持久 ACK 校验后才能 `--resume`，恢复 adapter、Adam、scheduler、Python/NumPy/Torch/CUDA RNG 和 step。沿用原逻辑值摘要核验，不声称跨进程 GPU 数值逐位等价。probe 状态全部弃用，正式训练从 fresh base 开始。精确 namespace、CLI 和导出文件清单见 plan。

本次 `cpu_contract.py` 实测 **59 项 PASS / 0.026066 秒**：新数据/来源/原划分排除合同、陈旧全文绑定拒绝、原 epoch 循环和完整恢复 AST、仅 ACK namespace 改动、两轮尾批以及源中的 ID/mask 核验。只读安全元数据和文件字节 SHA，不解析训练答案，不重分词/数学测试，不加载 Torch/model，不访问远端或评分。旧数值和真实恢复证据通过 plan 引用复用。

当前核心源已准备；实际 A100 H51 探针、控制器、持久 ACK、训练和质量终点由方案主线程维护。未收到本次探针真实 PASS 前不得开始正式训练。训练源码不含推断或评测入口；held 全文输入上限 14848 属于后续独立推断，不能使用本训练 6144 上限。
