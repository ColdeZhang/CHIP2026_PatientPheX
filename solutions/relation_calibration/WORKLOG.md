# 工作台账

- 当前：**固定 LR F0 结案关闭。Score .722874376758 高于B0 .711507580027，但低于固定2of3 .724457355332，未通过原双门；无扫描/重fit。**
- 源码与结果：独立分支 `explore/relation-calibration`，起点main a89f225。唯一fit前4b6e732、F0实际源/计划前eb40446b、joint计划前388494d6均先推送核验。
- 唯一元训练：F1全16篇39患者，三raw并集561（364正197负），11迭代、fit.003523秒、无warning；模型0fc19cbe，pickle cfe7c007重开exact。没有F1 v2。系数/截距/固定cut.5均未修改。
- F0来源：grounded173（4原文literal）、sample交集280、Qwen3362，完整union431；所有任务祖先排F0。grounded47含3cap/2strict JSON弃答原样保留；另两源全量strict。`common_f0.py`只作literal/纯compound与明确空弃答兼容，原F1 common/plan/model字节不变。
- 共同冻结：11:58:08 UTC c6d0cace。11:58:20两完整端点各唯一v2，旧0/总.3404秒，已冻结前没有本控制F0gold读取/哈希。
- LR结果：保全部315旧关系/947实体/47患者/16篇/非关联bytes；加25=12TP13FP，micro+.00812848/mac+.03733870。对固定2of3 −.001582978574，`retain=false`。
- 固定2of3：.724457355332（B0+.012949775305），加15=9TP6FP，micro+.01002345/mac+.04177565，`retain=true`。对应方案分支独立记录；原dev和当前best901未改。
- 参照：现成独立sampled F0 .721969518388；2of3高.002487836944，LR高.000904858370。仅列资源优先级，未重评或改变原验收门。
- 完整入口：`evidence/F0-result.json`；外部 `R/research/relation-calibration-20261024/` 及固定2of3外部目录。模型小JSON和8pattern已在Git，原始关系/标签/pickle/预测/日志均仓库外。
- 下一步：向root交全部refs与小包；本LR配方关闭，任何新方法需另立预写计划。不自行迁原dev或提交官方。
