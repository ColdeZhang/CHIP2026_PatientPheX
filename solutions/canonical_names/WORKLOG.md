# Canonical-name 集合工作台账

2026-10-24 UTC。分支 `explore/canonical-names`，起点 `b1d73cc9`。本路线只写本目录；共享协调由 root 负责。

- **目标**：全文患者条件 → HPO canonical-name 集合 → 精确 HPO 映射，补现有强 B 关系；实际目标仍为原 train64/dev16 v2 all/text ≥0.8。
- **阶段**：CPU 实施，无 GPU、fit、权重下载、held gold 或质量调用授权。先完整 87 train32／39 F1 纯输入模板与 native 监督检查，再实际 HF prefix processor CPU 契约和有限探针计划。
- **两基座准备**：优先现成 Qwen3.5-9B；按 root 资源安排另算现成 Qwen3-8B 完整模板与形状。换 8B 时明确 base 与目标同时变化，不称纯目标消融；训练前固定唯一选择。
- **固定终点**：强 B F1 Score .6863389812716092 的关系 ∪ 本头所有新 HPO 关系；1189 实体与 471 原关系保留。完整 freeze 后一次新 v2，严格总 Score 改善才候选；micro/macro 报分项，不额外加“双项同升”验收门。单头保存 raw，不评分。
- **复用**：readiness 已完成 HPO/train32 目标与原生 trie 检查；不重复这些检查。训练循环／stop-resume／SHM→.100 ACK 的统一接口由 grounded owner 提供，本线只绑定新输入、配置、实际 receipt；不写第三训练 loop。
- **已知限制**：HPO 子集目标 87 行／796 关系；5 个真空，14 患者有未表示的 NIL／分支外标签。完整原文不保证患者归属；标准名可推断过度，无逐字证据。
- **资源**：大产物 `/workspace/patientphex-runs/round2-20261023/research/canonical-names-v1/`。现有 CPU readiness 位于同根的 `canonical-name-generation-readiness-20261024/`。
- **下一步**：完整 native 模板/mask/cap/Pareto → 实际无模型 CPU 接口 → 给 root 有界 GPU probe/2ep 预算；GPU 只能在 root 后续明确排程后启动。

详见 [实验](experiments/E-20261024-CNAME-001.md)、[方法笔记](NOTES.md)。当前尚无本线模型质量结果。
