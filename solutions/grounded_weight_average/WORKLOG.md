# 工作台账

分支 `explore/grounded-weight-average`，起点 main `85611ab1e245cb60b7b450d71b0caf01512b46c7`，只写本目录。仓外产物 `R/research/grounded-weight-average-readiness-20261024/`。

## 当前状态 GWA-20261024-001：双折门失败，固定控制关闭

同一轨迹两轮有效 LoRA 增量平均（r32/alpha32）加固定 CPU FP32 Sap 查询的联合控制已完整结束。唯一公式 `B4 ∪ (G_mean ∩ G_original)`：

| 内部研发折 | B4 | 固定新端点 | Score差 | 新增关系 | micro差 / macro差 |
|---|---:|---:|---:|---|---|
| F1 | .701325377212 | .703883137773 | +.002557760560 | 19=7TP/12FP | −.000218063 / +.010449105 |
| F0 | .728195628911 | .717985608916 | −.010210019996 | 19=2TP/17FP | −.013424249 / −.027415831 |

两折各完整freeze后仅1v2/旧0，Task1和全部非关系字节不变；这些内部折已用于研发，不称盲测。严格双折门未过，**不迁originaldev、不调alpha/epoch/阈值/precision**。该结论只针对这份固定联合配方，不等于所有权重平均无效，也不能将收益或下降只归因权重平均。

F1 39原输入1409.885秒、35EOS/4cap；174 CPU查询7.034秒。独审3a53a536已通过，480685B小包实际服务器写入与回读核验db46d892，入口 `evidence/terminal-pointer.json`。F0 47原输入1220.816秒、42EOS/5cap，15:15:01Z真实释放GPU2；同89 CPU查询7.037秒；完整16doc/47patient/947实体/B343保护。F0收件104必要文件、94raw/ID字节等价重建；离线后只读CPU分块取回原responses（64dbbb31），未重新生成。

结果 `evidence/F0-result.json`；F0 freeze26c1d92a、预测9c3aa01b已13dbab90先推送再唯一评分。F0 332412B增量包已.100实际写入/fsync/重开并3次读取恢复全部97成员（74cd3088），完整统一入口见 `evidence/F0-terminal-pointer.json`，已交主线程与validation集中增量审；本线无运行作业，仓外 `R/research/grounded-weight-average-v1/{F1,F0}`。原有GPU规范化提议和未使用尾部授权保持未执行；无其他质量臂或官方操作。

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

原39于14:32:24.163154Z成功，1409.885秒；GPU2同次0MiB/noapps真实释放，未触发尾部。35EOS/4cap，四cap整行弃答。目录collector两次UNKNOWN原收据保留，改按原complete41c4和原responses6f5直接读回；78单行按原序列化重建并逐原SHA证明字节相同，明确非逐文件下载、非远端ZIP冻结。88必要文件collection3d846e05完整，源码74f93df9保留收件范围差异，完整字段/身份/原raw验证沿原合同。

原严格解析得到402锚点、308不同词面、180无效条目被拒、4行cap弃答；174必要Sap查询，精确query plan在 `normalization_cpu/held-query-plan.json`，CUDA隐藏、FP32/16线程/300+30，预写后唯一执行。无新gold/质量/拟合；原4cap不重采样。

- F0 固定复验已授权：同 r32 合成权重、47 完整输入、同 CPU FP32 查询策略；289420 tokens / max9935，成本预估2685.36s < 3000+30。原生ID CPU合同先行；不读F0 gold、不新训。双父train32逐document排F0见 evidence/F0-ancestry.json。

F0 原生47 CPU 8.513秒全通过，实际 IDs/rendered/decode 与旧缓存一致（90be3840）；新推断14:54:40.106115Z proc3d9e8637/driver271042/child271060，TERM15:44:40.327297Z/KILL15:45:10.327297Z，同A100/GPU2。计划4edd175c、源码470b81bf已先推。只读watcher185906按180秒观察原intent；未重新训练或改候选。下游静态检查修正F0引用键删除写错（无下游执行/质量），补有区分力键合同v2；不改变正在运行源。

F0收件受.12连接阻断：read_file在15:16、15:20实际LUNA_OFFLINE；按root许可185秒后15:23:59尝试原responses≤32768B纯CPU run_command回传，仍明确LUNA_OFFLINE（非UNKNOWN）。无片段/无模型重启/无远端写，继续只读退避。原complete由成功终态观察按cd41d29a逐字节恢复，42EOS/5cap及GPU释放均已有证据；不把文件读取错误当推断失败。恢复入口 `f0/read_original_responses.py` 保存各原片段，可在连接恢复后接续，再走 `direct_collect.py`。
