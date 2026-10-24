# patient-first F0 固定稳健性复验

## 目标、约束和成功标准

F1 patient-first单独负，oldSFT32∪new的固定secondary仅+.003479519，micro略降、macro升。真实目标是判断该有限互补在另一个预先留出文献折F0能否成立；不以更多训练或多试组合代替成功。

按主线程2026-10-24授权，复用旧strict train48完整204行与原F0纯输入58，转置同一闭集矩阵，fresh基座/2epoch/原优化器和解码不变。只允许primary新方案和secondary旧SFT48∪new，实体与候选池原样；两份完整预测共同freeze后各唯一v2，旧baseline复用不重评。保留条件沿用delta(micro+macro)>0且macro不降。未经后续明确安排不迁原dev/官方、不加第三组合/阈值，不使用F0 targets构建训练/资源。

## 当前授权与状态

- 已读当前main规则与H46/H47：每个独立模型≤10B，允许模型组合；不能把流水线总和当门槛。
- 新外部目录：`research/patient-first-closed-set-F0/`。自有helper只写其`cpu-preparation/`，做来源locator、先冻结plan、新204/58转换/分词与训练矩阵还原；不改Git、不MCP/GPU、不读F0 gold。
- 下一步核旧48实际native/A100资源与新最长tokens。若新长度或实际显存证据需要，可安排一次fresh A100最长行更新180+30，诊断状态丢弃；不重复已验证旧合约。正式2轮预算依新实际tokens/成本与空间冻结，此时没有新GPUjob或正式训练启动。A800留Med，不抢卡；MCP未知状态只查原intent/partial、不盲重启。
- 204/58及旧来源目前是主线程给定范围，具体SHA/token计数须待实际核验，不能把计划数当结果。F1 frozen源/产物均保持，见[完整F1记录](patient-first-closed-set.md)。
