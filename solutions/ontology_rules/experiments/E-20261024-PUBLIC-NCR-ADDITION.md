# E-20261024-PUBLIC-NCR-ADDITION：强BioBERT基底增加NCR

真实目标：新增公开NCR的候选/置信度能否改善现有强BioBERT实体选择。主对照为既有固定四fold Score .730993616，pre-B .727975656仅作归因；本实验已完成，聚合微正但未通过事前稳定性条件，当前配方不迁原dev。

官方原PDF §3.1（页4）限制每个individual模型≤10B，§4.2（页10）明确两个8B模型组合。独审证据 `/workspace/patientphex-runs/round2-20261023/validation/model-parameter-rule-correction/review.json` SHA5bd082d3；原按全链总和强制去B是过严项目假设，尚未进行任何fresh拟合，现已撤销。旧源/计划与实际参数库存保留，不能将此纠正写成某模型质量负结果。NCR全部资源、CPU/GPU验证和纯输入缓存继续有效，不重NN。

方法预先固定：保留强B八池（246804行）及六字段/候选/provenance，NCR原生1–7词、strict>.8和原重叠后按固定公共当前HPO规则映射；新增仅ncr_present/ncr_score，score为实际原float32当前exact-key最大值，保留raw来源位模式。共享候选原前缀字段值/类型/顺序保持，原B来源不得清除。

四fold保持原文献48/16、训练候选内层32→16来源，只有固定公共NCR输入cache可跨fold共享；新增下游模型不放宽H40来源隔离。fresh vectorizer/HGB160树/15叶、原完整参数/seed/early_stopping=False、固定.3及原nested，实际F/树state/模型SHA运行后保存。900秒+TERM30是整个正式窗口，包含4fit/infer与单一终点；数据/map/pool准备独立在此前完成。各fold完整held预测先共同冻结，再固定v2 all/text及格式核验；不生成训练posterior，不扫cut，不自动迁原dev或官方。

比较预定：报告强B与NCR的pooled Score/menF1/docF1及逐fold差，pre-B仅辅助；保留方向需强B实体两F1均值严格改善且至少3/4fold非降。原dev≥.8仍是最终项目目标，不用内部控制冒充达标。

状态：完整NCR缓存CPU最终核验通过（3553e219）；02:52全部192972B rawspan已回收并核efdf3beb。仅实际1298原HPO ID用同公共map_label投影（1278same/16alt/4unresolved），5341span中45未映射、其余5296canonical无折叠；不声称本地完整head核验。实际八池11.973秒完成：246804→253204行（+6400），原B前缀与六字段逐值/类型/顺序全部保留。统一缓存/输入证据 [manifest](../public_neuralcr_readiness/strong-biobert-plus-ncr-preparation/cache-input-terminal-manifest.json) SHA492aaa11；随后唯一四fold已完成，见下方终点。完整NPY本地partial保留、远端全量SHA/5341绑定有效，不追加MCP压力。原失败/恢复见 [缓存记录](../public_neuralcr_readiness/full-train64-cache-attempt2/README.md)。

正式窗口已启动：2026-10-24T02:58:03.882394Z，controller PID93438，03:13:03.882394Z主截止+TERM30；plan2f665745，全部实际源/输入已冻结并推送11b1e63。恢复读取外部 `strong-biobert-plus-ncr-preparation/formal-control/execution-state.json`，不得因会话中断重复启动。


## 固定四fold终点

正式总窗 **60.879秒/rc0**，四fit子进程57.177秒，唯一decode/评测3.678秒（程序3.497秒）；所有四份完整预测先冻结 `30662420`，再执行固定公共v2 all/text。没有新NN、cut扫描、原dev或官方操作，格式0错/0警，全部关系和nonentities原样。

| 指标 | 强BioBERT | 加NCR | 差值 |
| --- | ---: | ---: | ---: |
| Score | .730993616 | .731229952 | +.000236335 |
| menF1 | .771948494 | .772431452 | +.000482958 |
| docF1 | .809789609 | .810251992 | +.000462383 |
| micF1 | .687846109 | .687846109 | 0 |
| macF1 | .654390253 | .654390253 | 0 |

| Fold | 原Score | 新Score | ΔScore | 实际F |
| --- | ---: | ---: | ---: | ---: |
| 0 | .739700184 | .738458373 | −.001241811 | 104 |
| 1 | .728056418 | .728727984 | +.000671566 | 103 |
| 2 | .753373106 | .752780060 | −.000593047 | 105 |
| 3 | .706780475 | .708873675 | +.002093200 | 105 |

聚合mention为3620TP/925FP/1208FN（比强B −7TP/−17FP/+7FN），doc1881TP/311FP/570FN（−5TP/−10FP/+5FN）。这是少报带来的微小F1收益，不能据聚合增益声称新增NCR候选提高召回。仅2/4fold实体目标非降，低于事前要求3/4，`prespecified_success=false`。比pre-B的+.003254296仅为辅助归因，不能代替与强B比较。当前配方保留结果和可复用公共cache，但不升级默认、不自动迁原dev或改阈值。

四个真实fresh模型均160树、4640节点/2400叶，所有模型/held概率/预测SHA外部保留。实际F104/103/105/105，完整数值存储槽分别62247/62247/62251/62256（含结构，不能称神经可学习参数）；模型参数上限按每个individual模型核，未删除B。完整终点 [manifest](../public_neuralcr_readiness/strong-biobert-plus-ncr-preparation/terminal-manifest.json) SHA5c45554f；结果8ec12cfb；统一完整独审PASS（de325a2c），统计保留条件仍false；[终审](../public_neuralcr_readiness/strong-biobert-plus-ncr-preparation/independent-review.json)。

正式plan的`status/template_policy`保留模板阶段描述，运行前已明确`execution_ready=true`及根线程持续授权，源实际检查后启动唯一window；运行中未修改冻结plan，实际边界由window与完整终点证明。后续新计划应同步清理模板描述，避免记录语义混杂。

增删归因边界：现有冻结诊断只记录nested解码的输入侧碰撞/追加，metrics只含逐文与聚合计数，**未保存相对强B的新增/删除各自TP/FP分解**。按本次只复用证据的范围，不新增gold诊断/预测/调参；−7TP/−17FP只是净变化，不能从中唯一反推出四格。聚合小正伴随净FP减少、召回略降；关闭的是本次固定配方，不能推广为NCR所有用法无效。证据 [可用性](../public_neuralcr_readiness/strong-biobert-plus-ncr-preparation/existing-diagnostic-availability.json)。
