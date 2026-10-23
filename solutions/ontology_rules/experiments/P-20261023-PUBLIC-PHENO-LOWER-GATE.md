# P-20261023-PUBLIC-PHENO-LOWER-GATE：两点CPU候选池可行性

目标：确认已缓存的原生神经top3/原5dp输出能否无需NN、无需新标签精确重放固定>.95，再按root预定的>.5多数posterior门槛形成候选池，量化候选数量和CPU成本。只看.95/.5两点，不阈值扫描/质量评价/原dev。其余native dict/abbr/当前HPO映射不变；仅top1、非None。

先完成原.95全64 canonical字节复现及来源flag逐对象一致，再计.5；全部296984 raw候选offset和词索引映射、4130词典原结果逐项核验。现有raw均完整保留原top3顺序和native_probability_5dp字符串，解码只用第一项和固定阈值，不需要全部17266列或重新forward。代码不导入TF/torch/representation、不打开权重；单CPU线程180秒+TERM10硬限。

重要接口风险：原combine_ml_dict按连通重叠组每HPO留最高分，低分跨度可能桥接两个旧组；raw阈值集合单调扩大不意味着最终hybrid canonical严格超集。因此同时统计交集/新增/丢失，不自行改为union或添加新规则。数据/计划见[固定计划](../evidence/round2/public-phenotagger-lower-gate-plan.json)。

sequence当前prepare_public_selector明确绑定.95 plan0243，HGB使用public_present/score/neural/dictionary/abbreviation等属性，未额外筛public_score。可复用其严格三fold旧train32资源和固定HGB160/.4框架，但新候选池需要新的缓存计划/真实来源flag、重建各fold池并重新拟合；不能直接换池沿用已训.95树或冒充0243。本项只报告接口和成本，下一质量控制待root决定，与正在做的.95三fold稳健性分开。

实际首个CPU启动在资源绑定阶段0.165秒退出：远端只部署NLTK解压成员，未保留下载ZIP，计划错误地引用ZIP路径。未进入tokenize/候选处理、无模型调用；原source/plan/state/stderr保留于外部attempt1-resource-path。将21个远端成员逐一与原固定ZIP内字节比较通过，仅修资源路径/输出目录，source与数学政策不变，继续原180秒CPU预算。

修正版真实运行21:25:39Z–21:26:14Z完成，rc0/stderr空，程序29.538秒、wrapper34.589秒、峰RSS744380KiB；前处理27.136秒，原.95解码1.060秒、新.5解码1.174秒。所有原296984候选词索引/字符坐标和4130词典结果通过；原.95全64 canonical SHA876317bd与原缓存字节相同，全部来源flag对象等值（文件顺序不同）。

| 原生门槛 | raw非None接受 | hybrid缩写前 | 缩写后 | 当前有效canonical |
|---|---:|---:|---:|---:|
| >.95 | 6124 | 5973 | 6065 | 6064 |
| >.5 | 10047 | 8076 | 8219 | 8218 |

两臂均有1项不在当前有效HPO分支而丢弃。新池为旧池的1.3552倍；6061旧键保留、2157新键、3旧键丢失，证实native hybrid非单调，未自行union。原softmax分数只是类别质量，不是校准后的实体正确率；此计数不是质量提升证据。

现有缓存足以无需NN精确重放，新池CPU成本很低。下一步如root决定质量控制，必须新manifest/adapter并在原三fold隔离链重新拟合HGB160/.4，不能沿用.95已训树；方法与sequence已完成的.95稳健性分开。本次没有拟合/评分或原dev输入访问。[结果与恢复指针](../evidence/round2/public-phenotagger-lower-gate-results.json)记录实际源码/资源/产物SHA。完整预测/全raw留外部，Git不收大文件。

新增源码/资源/实际小终点增量独审通过（14099d49），不重复解码或读gold。结案后root在未看新质量前另授权旧.95∪新.5池控制，保留3个丢失键、真实flags OR/score max，四fold质量归下一独立实验；本CPU记录不改成质量结果。
