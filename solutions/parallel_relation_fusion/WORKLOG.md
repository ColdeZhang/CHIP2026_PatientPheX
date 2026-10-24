# 多路线关系融合

主线程独立分支，base main9c85180。F1两端点完整，主臂保留候选。原3路线强B融合各为F1正：grounded .695950447742，双采样 .693653303500，Qwen3 .689312034492；不是原dev .754361的新替代。

预先固定主臂B保护+三新增集合至少二源共识，诊断B保护+三源全并集，2完整预测jointfreeze后各1v2，旧0。必须严格胜最佳单源grounded才能留作联合候选，不按标签/患者选来源。先前只考虑二源union，第三路线在执行前完成后统一改为本两端点；此前二源方案没有source/plan/quality执行。

入口：`python solutions/parallel_relation_fusion/run.py --plan solutions/parallel_relation_fusion/plan.json assemble`，成功后同入口 `evaluate`。全量数据/预测外存，关系以外对象完整不变，无GPU/fit。

11:25完整2端点已成功：主二源一致36新关系18TP/18FP，Score .701087332488，较B+.014748351216、较最佳单源grounded+.005136884746，micro .747809152872 /macro .641601241291同升；诊断全并集177新关系44TP/133FP，Score .684526718071负，关闭全并集配方。原471关系、1189实体、16文献39患者保留，格式0/0；jointfreeze7fe01124后2次新v2/旧0，CPU .301580秒装配/.074897秒评分，无NN/fit。

后继只验证固定二源一致：三来源同原train32权重、相同推断/规范化/seed策略，纯F0完整输入。需先核全祖先排除F0；保持B0 .7115075800266846全部关系/实体。F0 Score必须严格胜B0才能保留为跨折候选，同时并列各单源F0，不扫阈值或根据患者选模型。尚未执行F0融合或读取其新gold。

23成员source/input/prediction/metrics小包已实际上传.100并全成员读回SHA/fsync重开通过，2次read_file；receipt5b590694。完整heldgold/模型不入包，独审另存索引。

集中终审297d9eb3通过：完整集合/三源身份/共冻时序/2v2计数及分项核验，未重评分。Git仅留freeze聚合投影，完整213条端点新增明细留外部freeze与已恢复小包，避免重复展开日志。
