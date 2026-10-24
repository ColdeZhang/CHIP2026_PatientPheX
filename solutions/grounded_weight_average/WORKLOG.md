# 工作台账

分支 `explore/grounded-weight-average`，起点 main `85611ab1e245cb60b7b450d71b0caf01512b46c7`，只写本目录。仓外产物 `R/research/grounded-weight-average-readiness-20261024/`。

## 当前任务 GWA-20261024-001

- 真实目标：检验现有同轨迹两epoch LoRA有效增量平均是否能成为更可靠的grounded关系视图，补充当前四来源强对照。
- 本阶段仅准备：历史查重、两真实checkpoint/配置/祖先、精确数学和加载schema、成本与唯一未执行终点。无GPU、fit、gold、质量或大权重下载。
- 固定机制：同base的r16/alpha32两adapter按有效增量等权平均，rank32/alpha32精确拼接；不平均A/B因子、不SVD降秩，不挑epoch/alpha。
- 未执行主端点：完整F1 `B4 ∪ (G_meanweights ∩ G_originalgreedy)`，literal-v2关系原子；B4与所有非关系对象保护。当前B4 .701325377212172。只有严格胜B4才由root决定后续；本阶段不运行或评分。
- 已核供给：两epoch11/22 config字节同a9e753c4，adapter b513def3/d52e2e6c均173181072B，原A100持久locator+双ACK+collector433659a2；此次不下载/重哈希远端大文件。
- 已完成：32项NumPy数学/严格schema合同PASS；actual PEFT加载/权重header/实际合成仍未执行。短报告与唯一未执行协议见[evidence/readiness.json](evidence/readiness.json)。
- 参数：拟r32 text+adapter9,040,359,936；即全bundle保守9,739,661,040，≤10B。原39推断1103.2秒仅成本proxy，建议root决定有界完整控制。
- 供给：原G282关系中62在当前B4外，只是潜在新增上界，未读gold或计算新分数。
- 已提交并实际核验远端 `origin/explore/grounded-weight-average@d23a7fb06a85ce3d9c23dab929d6008cb72a1d88`，源码/短报告/未执行plan均已备份。下一步仅由root决定实际CPU合成/PEFT微型加载及GPU，不自动开跑。
