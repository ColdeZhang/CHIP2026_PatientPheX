# 三个既有患者跨度控制：F1总量诊断

结论：在当前已知F1开发结果中，geometry 的主要可见问题偏向患者归属。47个新增FP有31个概念实际属于其他gold患者；39个有exact正gold跨度，37个还存在exact跨度且概念相同的支持实例。只有2个literal跨度概念不符，均涉及gold NIL；exact linked-HPO错ID为0。因此不能仅凭本次负分认为规范化是主要瓶颈，也不能把没有gold患者关系的概念一概称为非表型。

| 固定原配方 | B471外新增 | TP | FP | 概念属其他gold患者 | 文中无任何gold患者关系 | 后者有gold实体概念 / 无 | 强Task1文档支持 | 与B507 / B524交叠 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8ep |47|6|41|17|24|11 / 13|41|0 / 0|
| 24ep |87|10|77|55|22|5 / 17|74|6 / 8|
| geometry24 |55|8|47|31|16|7 / 9|46|3 / 4|

| 原配方FP证据 | 任一exact正gold span | 任一exact span literal概念不符 | exact linked-HPO错误 | 任一exact且概念相同替代 | 多个支持跨度 |
|---|---:|---:|---:|---:|---:|
| 8ep |27|2|0|25|10|
| 24ep |61|4|3|57|12|
| geometry24 |39|2|0|37|6|

口径：按原v2 associations的literal/合法compound-HPO拆分，其余字符串保留；NO实体排除，NIL实体概念用exact text。TP/FP仅相对原B471外新增，已断言等于原3份质量记录的总数；没有新Score/v2、预测、模型、患者选择或过滤。强8cb的1313实体按文档概念支持全部新增，不看gold；B507/524交叠也无gold依赖，不能当作患者归属证据。

跨度来自每条已保存selected-spans.json.gz的selected_patient_ids、identifiers、offset/length，原3线全部新增均有现成映射，无重建NN。同一关系可能来自多个跨度；any mismatch与正确替代可重叠，不能把一个坏实例归因成整条关系的唯一错误来源。gold NIL exact边界在Task1可获wildcard匹配，但在关系literal概念口径并不等价，故linked-HPO单列。没有gold实体/关系亦不能证明原文不存在真实现象或标注完备。

这是root在三条既有结果均已知后授权的有界开发诊断，所有源/预测保持不变。源、输入精确SHA与授权范围见intent.json；可复现入口为同分支diagnosis/aggregate.py，输出到新的独立目录。后续任何支持门或组合需另立完整预写控制；本报告未生成该类新臂。
