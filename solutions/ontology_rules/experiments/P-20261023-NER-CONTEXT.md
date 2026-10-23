# 相邻上下文 NER：训练内 CPU 闸门

2026-10-23；研究基础见[增强提案 B](P-20261023-NER-AUGMENTATION.md)。当前完成训练32输入统计和专用 collator／低层解码实际 CPU 检查；统计已独审，新增接口独审PASS（42fb9b45…）。未执行新模型或获得下一 GPU 授权。

目标是检验“原短目标窗缺少相邻语境”这一可证伪机制，最终仍看完整 v2 Score。上下文空间大不能证明它修复漏检；FLERT 的通用 NER 结果也不是本赛临床收益证据。不同于已经结案的同 HPO 替换，本控制保留原窗、词面、标注和更新次数，只增加编码器可见的同文献输入。

## 已测统计

真实 native tokenizer，32篇原1150窗，CPU10.13秒。1150窗从不含 entities/association 的全文精确重建；每个原目标词的 token ID 组、全局 offset 和已表示监督保持不变。

| 项目 | 结果 |
|---|---:|
| 可加上下文 | 1121／1150，97.48% |
| 无完整词可加 | 29，2.52%；其中24已满384 |
| 非零新增 subword | 均81.21；中位77；p95／最大128 |
| 原／新增后平均输入 token | 106.34／185.50 |
| 跨 section／跨 passage type 的窗 | 449（39.04%）／850（73.91%） |
| 原正例／NO／有效跨度 | 2236／87／1,125,538，均不变 |
| 近目标边缘12词的正例 | 354，其中350所在窗可加上下文 |
| 同词面既正又NO | 34种，涉及123目标窗，其中113可加上下文 |

新增上下文包含1702次已知完整正跨度、37次部分正跨度和55次NO重叠，说明“附加后未标注”不等于负例。上述计数仅用于训练内诊断，不参与上下文选择，也不代表预计修复量。固定原种子／batch4／10ep共2880更新，padding token 总量代理×1.3158，平方长度 attention 代理×1.5521；不是实测耗时或显存。

## 固定实现与实际 CPU 检查

仅使用同篇输入全文。优先目标窗外的同 passage 词，到 passage 边界才取相邻 passage；各侧最多64实际 subword，含一个已有 `[SEP]`，总实际输入≤384。剩余预算左右平分、不转移未用配额，不削减目标、不按标签挑选或裁上下文、不新增 token。跨 section 保留并单列风险。

`ner_context.py` 的编码器输入为前文＋目标＋后文，context attention=1、words_mask=0；原目标首 subword 映射1..N，RNN／跨度头只见原目标。原处理器产生的 labels、NO、span_idx、span_mask、text_lengths 原样复用，保持原 masked BCE 分母。关闭上下文也走新编码路径，以便检查与原 collator 真正等价。

推理用同一个 label-free 窗口构造与适配器，不把拼接文本交给高层 predict_entities 后过滤。低层 decoder 的候选仅来自目标槽，越界／padding 在 greedy 前屏蔽；输出通过原 word_offsets 映射，不把虚拟拼接文本当连续原文。

`check_ner_context.py` 已在 .3 现有环境完成全部1150窗关闭上下文时各 tensor 等价、开启后原监督与候选不变、无标签推理 tensor 一致、真实384预算与 attention/wordmap 检查；实际库的词抽取和跨度头合成 embedding 检查得到相同 `[4,295,12,1]` logits 形状，非法高分候选、全局切片、masked BCE 值与梯度检查均通过。**它不加载 checkpoint，不等于预训练 transformer／RNN 的实际前向通过。** 若任一闸门失败，先修接口，不进入 GPU。

数据仅使用冻结 nested train32 和原准备1150窗；模型配置／tokenizer 复用已核验的 BioMed 本地副本，当前无新增外部标签或 HPO 词面。本阶段不据此选择训练基座；之后的模型基座、对照、成本 probe、固定轮次和预算仍由主线程决定。当前统计已独审通过，实际适配器 CPU 运行19:53:48–19:54:43 UTC，检查54.061秒／wrapper55.154秒、rc0、CUDA未初始化；新增接口独审PASS（42fb9b45…）。实际 Torch2.7.0+cu128／Transformers4.50.0／GLiNER0.2.22／tokenizers0.21.4；原本地统计tokenizers0.22.1，各实际输入总量和覆盖数一致。原参考处理器仅提示 tokenizer 没有默认 max_length 而不截断；新适配器显式不截断，全部实际输入仍逐行≤384。

统计、实际来源 SHA、短计划与后续证据入口见[小索引](../evidence/round2/ner-context-feasibility.json)；全1150行统计和报告留仓库外，不保存新训练文本或大日志。

## 条件性训练与推理入口准备

主线程随后授权继续不依赖质量结果的 CPU 源码准备。新增 `train_ner_context.py`、`infer_ner_context.py` 与 `ner_context_runtime.py`，原序列入口保持不动。当前候选明确为 fresh BioMed 原生基座；最终是否执行和同源对照须等序列10轮结果，由主线程选择。准备若关闭，只表示没有选择运行，不能算该方法失败。

静态 AST 已核：masked BCE／计数函数、AdamW、warmup／scheduler 与原70c5入口精确一致，整个epoch循环除collator调用替换外也精确一致。仍保留1150窗、batch4、10ep／2880步、原训练probe和恢复状态；不从当前epoch10继续训练。推理固定单窗batch、.5、nested overlap规则、目标原文offset与max-score去重，要求实际新训练终点SHA，再冻结raw预测。

两个入口均有 `--cpu-prepare-only`，只读tokenizer/config和固定四窗，不构造模型或读取权重。实际fingerprint覆盖新入口／helper／adapter、GLiNER所有Python源、分词配置与依赖版本；未因准备安装新环境。下一步仅做这两个CPU入口实例检查和新增源审，不跑GPU、Sap或评分。来源与条件计划见[小索引](../evidence/round2/ner-context-entry-preparation.json)。
