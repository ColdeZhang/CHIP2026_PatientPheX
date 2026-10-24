# 工作台账

分支 `explore/grounded-weight-average`，起点 main `85611ab1e245cb60b7b450d71b0caf01512b46c7`，只写本目录。仓外产物 `R/research/grounded-weight-average-readiness-20261024/`。

## 当前状态 GWA-20261024-001

实际CPU合成与tiny PEFT已通过；A100 GPU2唯一完整39条推断进行中，proc06596b00/PID239977/child239995，开始14:08:54Z，TERM14:38:54.278290Z，KILL14:39:24.278290Z。见[evidence/inference-start.json](evidence/inference-start.json)；远端原 `grounded-weight-average-v1/F1/infer-v1/wrapper-complete.json` 为唯一终态。恢复入口 `inference/remote.py observe --directory R/research/grounded-weight-average-v1/F1/infer-v1`，绝不重start。

当前根授权范围：同一固定平均、39 fresh greedy、必要固定Sap180+30、唯一 `B4∪(G_meanweights∩G_originalgreedy)` 完整freeze后1v2；若byteexact则0v2。原candidate-plan保留为历史，新执行见[execution-plan.json](execution-plan.json)。无新训练、无gold读取或质量结果；后处理源码和差异合同已push。仓外实际产物 `R/research/grounded-weight-average-v1/F1/`，真实CPU16件包3116a023完整回收；新adapter346,294,736B/ed184ecc留原持久服务器。

## 准备阶段记录（授权前历史）

- 真实目标：检验现有同轨迹两epoch LoRA有效增量平均是否能成为更可靠的grounded关系视图，补充当前四来源强对照。
- 本阶段仅准备：历史查重、两真实checkpoint/配置/祖先、精确数学和加载schema、成本与唯一未执行终点。无GPU、fit、gold、质量或大权重下载。
- 固定机制：同base的r16/alpha32两adapter按有效增量等权平均，rank32/alpha32精确拼接；不平均A/B因子、不SVD降秩，不挑epoch/alpha。
- 未执行主端点：完整F1 `B4 ∪ (G_meanweights ∩ G_originalgreedy)`，literal-v2关系原子；B4与所有非关系对象保护。当前B4 .701325377212172。只有严格胜B4才由root决定后续；本阶段不运行或评分。
- 已核供给：两epoch11/22 config字节同a9e753c4，adapter b513def3/d52e2e6c均173181072B，原A100持久locator+双ACK+collector433659a2；此次不下载/重哈希远端大文件。
- 已完成：32项NumPy数学/严格schema合同PASS；actual PEFT加载/权重header/实际合成仍未执行。短报告与唯一未执行协议见[evidence/readiness.json](evidence/readiness.json)。
- 参数：拟r32 text+adapter9,040,359,936；即全bundle保守9,739,661,040，≤10B。原39推断1103.2秒仅成本proxy，建议root决定有界完整控制。
- 供给：原G282关系中62在当前B4外，只是潜在新增上界，未读gold或计算新分数。
- 已提交并实际核验远端 `origin/explore/grounded-weight-average@d23a7fb06a85ce3d9c23dab929d6008cb72a1d88`，源码/短报告/未执行plan均已备份。下一步仅由root决定实际CPU合成/PEFT微型加载及GPU，不自动开跑。

## 固定控制实施中

Root 已授权真实双 adapter 校验、CPU 精确拼接与 PEFT 0.21.2 tiny save/load；原候选协议保持不改，新执行范围见 `execution-plan.json`。CPU 唯一 120+15 秒、CUDA 隐藏；通过后 .12 GPU2 完整 39 greedy，1800+30 秒。真实参数门 9,040,359,936，唯一质量终点仍 B4∪(G_meanweights∩G_originalgreedy)。无新训练或系数扫描，未执行任何新质量。

CPU 实际 9.114 秒成功（proc89b08812，2026-10-24 14:04:34Z），两父 adapter/496 keys/F32/全部 config 已实核。新 248 模块、86,556,672 adapter 参数逐半张量保持原值，重开 exact；248 FP64 投影合同通过。真实 PEFT 0.21.2 的12类 tiny cat/save/load 通过，加载张量和投影输出 exact，CPU 无大基座/GPU/训练。完整凭据 `evidence/CPU-synthesis.json`。推断只替换 adapter 来源/rank/参数门，原39输入/nativeIDs/解码/固定规范化保持；启动前冻结独立新 plan。

14:08:54.063619Z 原 A100 恢复后唯一实际推断启动：proc06596b00/PID239977、child239995；fresh门通过，TERM14:38:54.278290Z/KILL14:39:24.278290Z。原185秒退避和失败均保留，未重CPU或换卡。源码57ea4c74/plan580974f1；回执见 `evidence/inference-start.json`。原 output wrapper 为唯一终态，未知响应只读恢复，不再次start。

Root另准CPU准备既有 `.100 GPU0` Sap包装：原.3已给DPO，不迁大生成模型；只有本39完整且sequence78+Sap实际释放后，可fresh门唯一180+30。保持原public41099 bank/六件Sap/strict train32 aliases/b59 policy/query与driver字节，披露Torch2.7.0/TF4.50→2.7.1/4.51.3非逐位等价。入口 `normalization100/`；尚无新query/plan/GPU意图。此前 `normalization/` 的.3未执行准备保留，无两卡重复。
