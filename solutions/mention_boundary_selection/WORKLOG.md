# 当前状态

固定 BOUNDARY3470 配方已完整负结案：strong-union **0.721846056324**，new-only **0.714176156405**，均低于强基线 **0.728697536469**。不迁F0/原dev/官方，不据此改阈值或挑子集。

- 3470行×2轮，868更新/两份完整状态ACK；最终adapter `beecfb84`。全1416推断1007.237秒；16:45:43.619895 UTC `.100 GPU1` 实际释放。
- 解码1290 HPO（含1 empty+missing）、123 NO、3 BOUNDARY、0 NIL/拒绝/cap；1289实体原子。并集新增124=24TP+100FP；new-only整体985TP/304FP，precision .764158。
- 较旧Top8并集净FP减少12、TP数不变，Score +.000820473；new-only +.000992478。共享SYSTEM/语法与539监督共同改变，不能作纯数据归因。
- CPU56成员、GPU2946小成员全SHA回收；4大状态保远端精确locator与实际回读证据。终态 [result.json](result.json)，外部稳定terminal `15896d47`，共同freeze `f79f1d83` 后恰好2v2/旧0。
- [实验过程](experiments/E-20261024-boundary.md)、[风险与经验](NOTES.md)、[固定计划](plan.json)。95件小备份360584 B/SHA7771597d已16:49:29完整远端验证及同SHA回读；最终增量独审faae8e67已PASS完整负结案（2946成员/868更新/tail6/双ACK/1416/双冻结评分/备份），见[evidence/final-review.json](evidence/final-review.json)。

大产物：`/workspace/patientphex-runs/round2-20261023/research/mention-boundary-selection-v1`。root独立预锁BOUNDARY-veto由其自行消费同一完整输出，本线不另跑该端点。

## 后继独立 F0 控制

root独立BOUNDARY-veto F1小正（删3=0TP/3FP），不改本线主头两负。3470训练及两个负例候选祖先排F0，因此复用固定head，无需重训。按root最终协议只用强F0的1085原跨度，373必要Sap已经完整且实际释放，新1085 nativeCPU6.673秒通过（616583tokens/max698），所有24小成员已回收。

同头唯一F0推断1800+30已17:03:03.946 UTC启动（proc9320b6db），完整EOS/strictparse后只删BOUNDARY，唯一veto冻结→至多1v2，与强F0 .740572140440比较。详见[f0_veto/README.md](f0_veto/README.md)、[协议](f0_veto/decision.json)。未运行原dev或官网；原F1主头负终态不变。

窄增量源核5ea22755未发现确认bug；静态覆盖依赖已用纯输入/基线核实1085精确跨度双射（51cdf15f），不读模型输出/gold或改已锁公式，见[evidence/F0-source-delta.json](evidence/F0-source-delta.json)。
