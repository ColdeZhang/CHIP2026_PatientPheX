# 原生公共 pair 推断的恢复源码

这些是 biomed helper 已实际使用的自有小型脚本逐字副本；`manifest.json` 固定原位置、大小及 SHA。保留它们是为了能恢复真实接口，不代表在当前目录直接运行即可。

- `pair-native/` 对应外部 `research/phenobert-pair-native/`；`candidate-plan/` 对应 `research/phenobert-pair-candidate-plan/`。恢复时按原目录结构放回，使用实际 plan/manifest 的路径和 SHA，不修改正在运行的原文件。
- 官方 87 个源码文件不复制入 Git；`fetch_source.py` 与 `upstream-source-index.json` 绑定 `EclipseCN/PhenoBERT@195d5df2ad9c6265e1de523f139694c7646eee29` 和各 Git blob/SHA，可重取。权重、词表、配置、数据及响应在外部持久目录，完整指针见方案计划。
- 原模型是固定官方完整 Python pickle；只有验证来源/SHA/实际类后原生加载。不要把当前 `torch.load(..., weights_only=False)` 用于任意用户 checkpoint。已完成独立原生加载审查 `5a8a7185`，不为本特征控制重新转换模型。
- 部署/收集脚本引用现有 MCP 客户端和特定环境路径；新环境先更新独立 plan，核文件、实际解释器和目标 GPU。这里没有秘钥、权重或临时签名 URL。
- 当前只准备候选三类特征。不要自行运行批量 GPU、HGB fit、评估或原 dev；正式运行与预算由 root 调度。
