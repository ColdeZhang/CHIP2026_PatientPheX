# Canonical-name 集合工作台账

2026-10-24 UTC。分支 `explore/canonical-names`，起点 `b1d73cc9`。本路线只写本目录；共享协调由 root 负责。

- **目标**：全文患者条件 → HPO canonical-name 集合 → 精确 HPO 映射，补现有强 B 关系；实际目标仍为原 train64/dev16 v2 all/text ≥0.8。
- **阶段**：唯一 managed 链 `proc_b327dfd3-24dd-4775-881a-713165fa1bcb` 的实际探针已通过（83.776 秒），13 FB／1 Adam、504 梯度与状态有限，两 Pareto 与训练行约束生成接口均通过。释放探针状态后正式训练自动于 12:17:08.859 UTC 开始，child2737169，原 TERM12:57:08.859／KILL12:57:38.859。12:21:07 已到 epoch1/update7；仍无质量结果。
- **基座选择**：Qwen3-8B b968826 已在任何新 held 质量前固定；base 与目标同时变化，不称纯目标消融。9B只保留后备：本地完整模板通过，额外运行因 gzip 容器 SHA guard 退出；已有解压 payload 全等，未复跑，实际HF processor未到达。
- **固定终点**：强 B F1 Score .6863389812716092 的关系 ∪ 本头所有新 HPO 关系；1189 实体与 471 原关系保留。完整 freeze 后一次新 v2，严格总 Score 改善才候选；micro/macro 报分项，不额外加“双项同升”验收门。单头保存 raw，不评分。
- **复用**：readiness 已完成 HPO/train32 目标与原生 trie 检查；不重复这些检查。实际 8B 直接调用 document owner 的固定 `native.probe/train`，两轮完整状态原子落入 .100 持久目录并 ACK；9B 的 grounded 共享桥仅作后备，不参与本次训练。不写第三训练 loop。
- **已知限制**：HPO 子集目标 87 行／796 关系；5 个真空，14 患者有未表示的 NIL／分支外标签。完整原文不保证患者归属；标准名可推断过度，无逐字证据。
- **资源**：大产物 `/workspace/patientphex-runs/round2-20261023/research/canonical-names-v1/`。现有 CPU readiness 位于同根的 `canonical-name-generation-readiness-20261024/`。
- **下一步**：持续查看原作业proc_b327dfd3，唯一驱动自动顺序完成授权链；不可再次start。真实压力8FB→Adam→3累积→Pareto(8832,208)/(5216,269)，释放optimizer/grads后用训练行验证greedy512+prefix；状态丢弃。OOM/形状/预算失败即保存原失败与释放，不截断、不改精度、不自动重试。

详见 [实验](experiments/E-20261024-CNAME-001.md)、[方法笔记](NOTES.md)。当前尚无本线模型质量结果。

8B完整模板：434582总tokens/max8832，监督5406/max269；F1 39共198785/max9923，固定输入cap9984。原生loop来自document分支f69d88d0，源码41443a75。实际探针峰值 23,612,581,376B allocated／24,970,788,864B reserved，正式成本估计1124.585秒＜2400秒。不可把成本估计当正式完成；按原作业与绝对截止继续观察。
