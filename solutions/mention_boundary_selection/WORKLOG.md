# 当前状态

固定 BOUNDARY3470 配方已完整负结案：strong-union **0.721846056324**，new-only **0.714176156405**，均低于强基线 **0.728697536469**。不迁F0/原dev/官方，不据此改阈值或挑子集。

- 3470行×2轮，868更新/两份完整状态ACK；最终adapter `beecfb84`。全1416推断1007.237秒；16:45:43.619895 UTC `.100 GPU1` 实际释放。
- 解码1290 HPO（含1 empty+missing）、123 NO、3 BOUNDARY、0 NIL/拒绝/cap；1289实体原子。并集新增124=24TP+100FP；new-only整体985TP/304FP，precision .764158。
- 较旧Top8并集净FP减少12、TP数不变，Score +.000820473；new-only +.000992478。共享SYSTEM/语法与539监督共同改变，不能作纯数据归因。
- CPU56成员、GPU2946小成员全SHA回收；4大状态保远端精确locator与实际回读证据。终态 [result.json](result.json)，外部稳定terminal `15896d47`，共同freeze `f79f1d83` 后恰好2v2/旧0。
- [实验过程](experiments/E-20261024-boundary.md)、[风险与经验](NOTES.md)、[固定计划](plan.json)。95件小备份360584 B/SHA7771597d已16:49:29完整远端验证及同SHA回读；最终增量独审faae8e67已PASS完整负结案（2946成员/868更新/tail6/双ACK/1416/双冻结评分/备份），见[evidence/final-review.json](evidence/final-review.json)。

大产物：`/workspace/patientphex-runs/round2-20261023/research/mention-boundary-selection-v1`。root独立预锁BOUNDARY-veto由其自行消费同一完整输出，本线不另跑该端点。

## 独立 F0 veto：完整负结案

root的F1独立veto小正（删3=0TP/3FP）经同head严格F0复验失败：**.740572140440 → .740492025895**，删4=1TP/3FP，men小升但doc下降，两关系不变。唯一1v2/旧0在完整freeze351f651c之后；不原dev/子集/改cut/官方。此结果不改变本线原主头两负或F1小正历史。

1085完整推断770.098秒，17:15:54.981426 UTC GPU1真实释放；2202小成员全SHA，adapter504逐值exact。20 Sap＋24 nativeCPU原件也全收。稳定终态864c1e24，见[f0_veto/result.json](f0_veto/result.json)、[实验](experiments/E-20261024-boundary-veto-F0.md)。窄源核5ea22755和base/held1085双射51cdf15f已完成，完整质量独审由root集中安排，小备份回执另补。

3470训练及负例候选祖先确实排F0；本次无需重训。不要把F1 proposer48含F0误说成头训练泄漏，也不要把本次全量质量负解释成来源或工程失败。
