# 固定跨度内 top3 ID 与真实观察特征控制

目标：检验已缓存 top3 的潜在 ID 供给能否转为 Task1 质量；上界诊断中的 298 个 gold 查询只用于决定研究方向，不用于本实验候选筛选。主线程已授权：原 Bioformer HGB `.3` 作匹配基线，四个 outer 各沿三套 train32→16 候选来源，固定 HGB160/seed20261023/cut.3；只在原既有候选跨度上机械加入两模型 top1–3 current-valid ID、保留全部旧键与字段。

每模型仅新增真实观察 `p/rank/observed` 三字段，共六项。缺失用 observed=false 配合数值占位，不伪称未观察类别的 posterior；多列映射同 ID 时保存实际 max-column p/min-rank。原 public/Bioformer 六字段和混合 score 不改。新候选与新特征是组合改动，不能称纯 rank 单变量。

先 label-free 重建旧池，核四个 held16 的所有旧 candidate 对象完全等于既有分数文件，三 nested 输入并集48且与outer16互斥；不解析 gold，原 generator 谱系复用已审 manifest。统一白名单仅 pmc_id/offset/length，由全部旧训练/输入池求并集，不含诊断 gold。远端只提取已有 raw top3，不加载模型/NN。

训练启动门槛：每折新训练行数不超过旧3倍，预估四 fit/infer≤240秒；满足后冻结实际 plan/source，单共享240秒硬限执行，不另开审批链，超过任一门槛则仅回报规模，不硬跑。四旧/新完整预测共同冻结后唯一完整 v2 all/text pooled/各fold评测，不扫描、不originaldev、不GPU。

关系边界已由主线程固定：旧cut.3仅改变实体、association原来自.4；本次两臂都完整保留该旧文件所有nonentities，只替换新entities，不调用重新关联。因此此前未有的旧cut.3完整v2也在共同freeze后此次一次性计算；不能拿旧.4完整Score冒充cut.3。

[准备计划](../evidence/round2/public-phenotagger-top3-control-preparation.json)。状态：label-free准备完成，实际行数超过预定硬门槛，停在训练前。

## 实际预检与停止原因

旧池9.447秒重建，61,293唯一旧跨度用于统一无标签lookup；helper用16.855秒读取两模型固定raw，全部坐标/列/源SHA通过。新池准备22.647秒，旧键/字段逐值保留，未增加跨度。

| outerfold | 原训练候选 | 新训练候选 | 倍率 |
|---|---:|---:|---:|
| 0 | 46,349 | 141,662 | 3.05642 |
| 1 | 45,039 | 137,372 | 3.05007 |
| 2 | 45,561 | 139,115 | 3.05338 |
| 3 | 46,064 | 140,680 | 3.05401 |

四训练/输入池重复计数共244,794→747,178行；136,096个有新增ID的pool-span中，六新增观察特征涉及新ID的完全相同向量碰撞为0。这仅证明这些输入的数值可区分，不证明HGB可学或质量。

按旧真实端到端耗时×实际行数倍率×新增维度，附25%余量及15秒固定开销，四fit估计218.60秒。**时间估计通过，但四折都超预定3倍行数独立硬门，因此未启动fit、未生成新预测/分数。**没有事后裁候选、调整特征、提高门槛或触碰原dev。fit/evaluate/run入口仅保存为未执行准备；未冻结正式训练计划。

[预检结果与实际SHA](../evidence/round2/public-phenotagger-top3-precheck.json)、[小lookup源索引](../evidence/round2/public-phenotagger-top3-lookup-index.json)；全池/lookup留仓库外，复原命令为 `python prepare_new_pools.py --plan <pool-plan.json>`（输出须改为全新目录并另存新plan，不能覆盖本次）。这不是质量负结果，只是预算门槛停止。
