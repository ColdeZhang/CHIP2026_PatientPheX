# 当前状态

固定 BOUNDARY3470 配方已完整负结案：strong-union **0.721846056324**，new-only **0.714176156405**，均低于强基线 **0.728697536469**。不迁F0/原dev/官方，不据此改阈值或挑子集。

- 3470行×2轮，868更新/两份完整状态ACK；最终adapter `beecfb84`。全1416推断1007.237秒；16:45:43.619895 UTC `.100 GPU1` 实际释放。
- 解码1290 HPO（含1 empty+missing）、123 NO、3 BOUNDARY、0 NIL/拒绝/cap；1289实体原子。并集新增124=24TP+100FP；new-only整体985TP/304FP，precision .764158。
- 较旧Top8并集净FP减少12、TP数不变，Score +.000820473；new-only +.000992478。共享SYSTEM/语法与539监督共同改变，不能作纯数据归因。
- CPU56成员、GPU2946小成员全SHA回收；4大状态保远端精确locator与实际回读证据。终态 [result.json](result.json)，外部稳定terminal `15896d47`，共同freeze `f79f1d83` 后恰好2v2/旧0。
- [实验过程](experiments/E-20261024-boundary.md)、[风险与经验](NOTES.md)、[固定计划](plan.json)。95件小备份360584 B/SHA7771597d已16:49:29完整远端验证及同SHA回读；最终增量独审等待集中收口。

大产物：`/workspace/patientphex-runs/round2-20261023/research/mention-boundary-selection-v1`。root独立预锁BOUNDARY-veto由其自行消费同一完整输出，本线不另跑该端点。

## 后继独立控制准备

root预锁BOUNDARY-veto F1小正（删3=0TP/3FP），不改本线两负。CPU迁移核已确认3470训练与两个负例候选祖先排F0，无需重训；现成F0强LGB1085与同GLiNER接受736合为1149跨度，缺403公共Top8，完整头推断代理834秒/保守1254秒。原dev1061缺408queries，作为后继备用；未GPU/质量。见[evidence/transfer-readiness.json](evidence/transfer-readiness.json)，等待root固定F0端点与资源安排。
