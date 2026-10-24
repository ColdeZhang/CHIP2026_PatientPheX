# 文档支持的患者跨度关系

当前状态：固定源码/计划准备完成，尚未构造完整融合或读取验证标签。真实目标是在当前强四源 B524 上增加净正确患者关系；一次内部开发折检查不代表达到原 train64/dev16 的 0.8。

- 分支 `explore/span-document-support`，起点 main `20fa611b`；只改本目录。
- [实验](experiments/E-20261024-SPAN-DOC-SUPPORT-001.md)、[固定决定](decision.json)、[完整计划及来源](plan-F1.json)。
- 固定 `B524 ∪ document_supported(geometry190, Task1 8cb)`。B 的全部 16 文档、39 患者、1189 实体、524 关系及其他字段保护。
- H59 查重：文档支持门已有成熟实际实验；此次仅换成最新固定 geometry 的独立供给，并非新门。选源已知负结论和患者归属局限，明确为开发选择。
- 下一步：边界合同、源码/计划 push 并核远端 → 完整构造及共同冻结 → 有新增才唯一 1v2/old0。严格 Score 正才提请 root 决定 F0，不自行延伸。
- 无 NN、模型/tokenizer、GPU、fit、远端调用、原 dev/F0 gold 或官方动作；大产物在计划指定仓库外，可由父来源和本源码重建。
