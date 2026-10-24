# Full64 响应编码 A800 运行源

这里只迁移原 `../../full64/formal_train/` 的运行位置和 Torch 版本，不换训练配方。原 A100 正式任务没有 campaign/start；原 A100 探针仍保留为旧机证据，不能授权新 A800 训练。

完整265行/64文献/173患者、每轮918058 tokens/17399监督、最长6462/上限6656；原临床正文、PT目标和 native 编码来源不变。r16/alpha32/dropout.05、seed20261023、AdamW1e-4/wd.01/clip1、batch1/累积8、BF16/native SDPA/GDN 保留。两轮各34更新，33组8加尾组1，总68更新、warmup4。row-mean indexed CE 和整个逐微批训练循环均复用，没有再跑数学探针或 native CPU 编码。

A800 GPU2 UUID `GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc`，Torch2.7.1+cu126/TF5.20.2/PEFT.21.2/tokenizers.23.1。正式训练仍在加载模型前核全265行实际 IDs/labels SHA；同版本不声称跨主机数值等价。公共基座使用已核 `/dev/shm/patientphex-qwen35-1045-c20223623576`，全状态仅写本任务 `full64-a800-formal2ep` 私有 SHM。

5400秒+TERM30共享窗口包含两次完整 state/adapter 至 .100 的持久 ACK，各导出最多600秒且不超过原整体截止。第二轮只恢复自身已完整导出的第一轮34步、Adam、scheduler、Python/NumPy/Torch/CUDA RNG，不复用任何 probe 或 C32 adapter。持久目录采用独立 `patient-response-code/full64-a800-v1/checkpoints`。

`prepare_source_plan.py` 只冻结源计划；`CPU-contract-result.json` 是本机标准库源/调度/数据元信息合同，绝非新 A800 GPU PASS。`../controller/entry.py prepare` 仅在新 `21255251…` 探针的实际12F/B+1Adam终态和实测成本通过后才创建部署目录与授权。新 CPU 合同详见结果文件；继承旧数学/状态协议证据，整个 trainer 仅 runtime、SHM路径和 namespace 合同 SHA 变化。

本 helper 没有 remote/GPU/Git/评分调用。两轮真实 ACK 完成后，独立完整64条纯推断交主 owner/root 融合，不读 originaldev gold。
