# 多路线关系融合

主线程独立分支，base main9c85180。当前无新分数。原3路线强B融合各为F1正：grounded .695950447742，双采样 .693653303500，Qwen3 .689312034492；不是原dev .754361的新替代。

预先固定主臂B保护+三新增集合至少二源共识，诊断B保护+三源全并集，2完整预测jointfreeze后各1v2，旧0。必须严格胜最佳单源grounded才能留作联合候选，不按标签/患者选来源。先前只考虑二源union，第三路线在执行前完成后统一改为本两端点；此前二源方案没有source/plan/quality执行。

入口：`python solutions/parallel_relation_fusion/run.py --plan solutions/parallel_relation_fusion/plan.json assemble`，成功后同入口 `evaluate`。全量数据/预测外存，关系以外对象完整不变，无GPU/fit。
