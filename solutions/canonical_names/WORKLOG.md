# Canonical-name 集合工作台账

2026-10-24 UTC。分支 `explore/canonical-names`，起点 `b1d73cc9`。本路线只写本目录；共享协调由 root 负责。

- **目标**：全文患者条件 → HPO canonical-name 集合 → 精确 HPO 映射，补现有强 B 关系；实际目标仍为原 train64/dev16 v2 all/text ≥0.8。
- **阶段**：已完整负结案。原探针→两轮22更新→全39生成均成功；唯一F1强B全并集 Score **0.672945837346**，低于B **0.686338981272**（Δ−0.013393143926）。追加52条全部为本口径FP，TP/FN不变。集中终审入口 [final-review](evidence/final-review.json)。
- **基座选择**：Qwen3-8B b968826 已在任何新 held 质量前固定；base 与目标同时变化，不称纯目标消融。9B只保留后备：本地完整模板通过，额外运行因 gzip 容器 SHA guard 退出；已有解压 payload 全等，未复跑，实际HF processor未到达。
- **固定终点**：强 B F1 Score .6863389812716092 的关系 ∪ 本头所有新 HPO 关系；1189 实体与 471 原关系保留。完整 freeze 后一次新 v2，严格总 Score 改善才候选；micro/macro 报分项，不额外加“双项同升”验收门。单头保存 raw，不评分。
- **复用**：readiness 已完成 HPO/train32 目标与原生 trie 检查；不重复这些检查。实际 8B 直接调用 document owner 的固定 `native.probe/train`，两轮完整状态原子落入 .100 持久目录并 ACK；9B 的 grounded 共享桥仅作后备，不参与本次训练。不写第三训练 loop。
- **已知限制**：HPO 子集目标 87 行／796 关系；5 个真空，14 患者有未表示的 NIL／分支外标签。完整原文不保证患者归属；标准名可推断过度，无逐字证据。
- **资源**：大产物 `/workspace/patientphex-runs/round2-20261023/research/canonical-names-v1/`。现有 CPU readiness 位于同根的 `canonical-name-generation-readiness-20261024/`。
- **下一步**：本固定配方关闭，不继续GPU／生成／评分；root负责已经预锁的独立交集。GPU0于12:36:50.561766 UTC实际0MiB/noapps，已交给span owner。新会话先读集中入口和下方结论，不重启旧链。

详见 [实验](experiments/E-20261024-CNAME-001.md)、[方法笔记](NOTES.md)。当前结果为已知开发折F1，原dev和官方均未测试本线。

8B完整模板：434582总tokens/max8832，监督5406/max269；F1 39共198785/max9923，固定输入cap9984。原生loop来自document分支f69d88d0，源码41443a75。实际探针峰值 23,612,581,376B allocated／24,970,788,864B reserved，正式成本估计1124.585秒＜2400秒。不可把成本估计当正式完成；按原作业与绝对截止继续观察。
