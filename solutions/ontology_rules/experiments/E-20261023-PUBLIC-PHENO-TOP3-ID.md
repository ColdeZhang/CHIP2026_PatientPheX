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

## 无标签资源预算修订 v2（训练前）

主线程在未fit、未生成新quality前确认3倍属于预设资源保护门，允许按实际最大3.05642倍将行数上限修订为3.1倍；共享fit+infer240秒硬限不变。原3倍停止记录及提交2efd764完整保留，原池字节不变，不裁候选、不更改模型/标签/cut/评测。新计划独立冻结；同span六字段区分只作为接口证据，不预言收益。

[预算修订](../evidence/round2/public-phenotagger-top3-budget-revision-v2.json)、[正式计划](../evidence/round2/public-phenotagger-top3-formal-plan-v2.json)。正式执行入口 `run_control.py --plan <formal-plan-v2.json>`；先四fold fit+infer共同冻结，随后 `evaluate_control.py --plan <formal-plan-v2.json>`。当前计划已准备，运行结果后补。

## 唯一正式终点（负）

计划与源61b1290已push/远端核验后，22:04:42.950Z开始，四fold真实端到端38.166/35.457/35.382/36.381秒，共145.395秒<240，均rc0。22:07:08Z四旧/新完整预测共同冻结`09a8bd4d…`，随后一次完整v2。原池增量独立重建PASS `fe12dd82…`：8池旧244,794行字段类型/值/顺序、新747,178行观察特征/ID逐项相等，不重新NN。

| 指标 | 原Bioformer cut.3 | top3 ID+观察特征 | 差值 |
|---|---:|---:|---:|
| menF1 | .763831183 | .757437568 | −.006393615 |
| docF1 | .796554009 | .794195251 | −.002358759 |
| micF1 | .687846109 | .687846109 | 0 |
| macF1 | .654390253 | .654390253 | 0 |
| Score | .725655389 | .723467295 | −.002188093 |

四折Score差值−.003303132/−.000453335/−.002885351/−.002571215，全部负；mention净−5TP/+69FP、doc+3TP/+18FP，所有非entities/关系计数完全相同。实体改动新增48TP/130FP、移除53TP/61FP；两臂0错0警。新增候选与观察特征是联合改动，不能将结果单独归因rank；上界中的155可供给不等于学习选择后可恢复。

**固定配方负，关闭当前配置，不扫描/扩大/原dev迁移。**这次才首次测到旧cut.3完整v2，因此其.725655389不能与此前旧.4 fullScore混为同基线。全池、概率、预测外存，Git仅小源/计划/摘要；新HGB负终点未持久化估计器，保留精确重训与评测入口/来源，不冒称已有序列化模型。终点独审PASS `8368ffa1…`，四新fit全部概率/预测逐字一致、两臂完整v2/格式精确复现；[完整实际结果/源SHA/恢复路径](../evidence/round2/public-phenotagger-top3-control-results.json)。

## 独立九点校准诊断（新授权，原固定.3负结论不变）

主线程另授权只复用新模型现存四fold概率，采用旧已预先使用的0.1–0.9九点与原greedy；目标pooled `(menF1+docF1)/2`，并列距.4近再更高。36个新完整预测先共同freeze再用train64标签计算，所有nonentities继续旧cut.3原样，CPU≤30秒，无新fit/特征/原dev。以旧已校准最高.780192596作同口径比较，不能将训练侧校准最大值叫泛化或Score/.8。原.3锚点预测必须逐字节相同。

旧校准入口9b252562原件保留，decode/entity_counts/summarize三函数AST完全一致；新独立入口只绑定新分数/锚点/预算。[冻结计划](../evidence/round2/public-phenotagger-top3-cut-plan.json)。结果后补；不会覆盖上述固定控制或自动迁原dev。

### 九点校准实际结果：仍负，关闭完整配方

7d8fc06已push后唯一诊断于22:09:15.641Z开始，程序2.872秒、wrapper2.979秒<30，rc0；新36个完整预测共同freeze后才计算两实体目标。新最优为cut.4，menF1=.760888889、docF1=.794144144，目标均值 **.777516517**；旧同九点校准最优cut.3为 **.780192596**，差 **−.002676080**。选定运行点的四fold目标差−.003313831/−.002390436/−.001536116/−.003654151，仍全部负。

新.3固定锚点预测逐字节复现；旧/新所有nonentities仍原样。仅调到已预定九点中更好的运行点没有恢复相对旧方案的质量；本诊断不能证明某个单独组件造成排序损失。**按预定条件关闭当前完整top3-ID+观察特征配方，不再加cut/fit/原dev。**完整曲线、实际36冻结SHA和恢复见[校准证据](../evidence/round2/public-phenotagger-top3-cut-results.json)。同64标签选出的最大值仅校准结果，不是泛化、完整比赛Score或达到.8；原固定.3负结果完整保留。新增校准小审待回收。
