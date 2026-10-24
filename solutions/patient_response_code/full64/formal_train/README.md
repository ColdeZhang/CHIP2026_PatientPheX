# Full64 响应编码：fresh 两轮训练核心

本命名空间只机械适配已实跑的 C32 indexed CE 训练器。原 train64 四折候选生成祖先均排除原 dev16；完整265条短输入的临床正文、PT target 和真实 native IDs/mask 保持。训练数据 `4fdc4c66…`、encoded `3d294abc…`，每轮918058 tokens，17399监督，最长6462，拒绝上限6656，不截断。编码前原目标15410 tokens仅用于来源披露，不冒称目标 token 未变。

原 r16/alpha32/dropout.05、seed20261023、AdamW1e-4/wd.01/clip1、batch1/累积8、BF16/native SDPA/GDN、checkpoint use_reentrant=False 保留。265行每轮34更新（33组8＋尾组1），两轮68更新；原 warmup_ratio=.05 推导 **ceil(.05×68)=4**。每组按实际组大小归一化，未改变 row-mean CE 或 indexed head。不得从 C32 adapter 或工程 probe 状态续训。

完整训练共享5400秒＋TERM30，包含两次完整 state/adapter 私有SHM→.100持久ACK。epoch1在34更新处停止；epoch2必须校验完整ACK并恢复 adapter、Adam两矩、scheduler、Python/NumPy/Torch/CUDA RNG、global step34，再完成68。旧训练器的逻辑值摘要核验保持；不声称跨GPU数值逐位等价。每轮导出最多600秒，仍受原整体截止限制。

实际 CPU `0ef29c12…` /27.836秒来自 .100 Torch2.7.1，训练目标 A100 Torch2.6.0；TF5.20.2/PEFT.21.2/tokenizers.23.1及模型/CE源固定。正式入口在模型加载前对265行逐实际 input IDs、labels SHA 复核。held64的13824上限属于独立纯推断，不能误用训练6656上限。

新本地合同 **78项 PASS /0.043218秒**：实际来源与native账、64/16文献排除及四fold祖先、完整265顺序/尾组、4 warmup、旧逐微批数学AST、完整恢复AST仅16→34、新状态门与旧C32状态拒绝、真实probe缺失/失败不能激活、共享截止与只读状态恢复。没有重跑 native/tokenizer、模型或评分，没有读取训练target body或dev gold。见 `CPU-contract-result.json`。

当前只准备源。真正可部署controller由 `../controller/entry.py prepare` 在新H51真实PASS且绑定成本≤5400后生成；没有预造训练授权。正式两ACK后，`../inference/` 仅产完整64 canonical关系，交root执行已预锁融合，零本地质量调用。
