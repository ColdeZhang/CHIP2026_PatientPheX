# 当前状态

独立完整终审已通过：[稳定报告指针](evidence/final-review.json)；2941 原件、734 更新、两 ACK、1416 完整输出、两负终点与 87 件备份闭合。原终态不改。

2026-10-24 UTC：固定F1配方负结案；不扩F0或原dev，不改阈值。独立 explore/mention-candidate-selection，实际GPU源 f47a8bcb43dc9fd65d4bd0a81aee44ab8af8687b；唯一proc791完整probe→fresh两轮→1416推断结束。

| 完整端点 | Score | menF1 | docF1 | 实体数 |
|---|---:|---:|---:|---:|
| 强B固定基线 | 0.728697536469 | 0.774943735934 | 0.821052631579 | 1313 |
| B并集全部接受实体 | 0.721025583801 | 0.754461099215 | 0.810847457627 | 1449 |
| 新头全部实体替换 | 0.713183678719 | 0.741327300151 | 0.792613636364 | 1299 |

新增136原子=24TP/112FP；new-only整体983TP/316FP，precision 0.756736、recall 0.726534。这两种精度分母不同。两端点非entity对象保持，关系micF1/macF1全等原B。

全1416输出模式：HPO 1299（其中empty OTHER 0；预测missing总0），NIL 0，NO 117，reject 0，cap 0；接受1299原子。模式统计不是正确率，missing不是heldgold召回测量；OTHER不转NO或NIL。

完整两轮734更新、5862行参与、3033240输入tokens/67044监督，native 3116.626s/含load 3123.402s，两个504adapter/504Adam1512状态+scheduler/RNG ACK；只用epoch2，无eval/holdout/早停或择轮。完整infer 984.377s。probe13真实FB/1Adam与固定成本门通过后丢弃；正式从同公开基座fresh训练。

固定来源：原2353全部保留+578纯F3→F2/F2→F3零注释重叠annotation-negative，2931=2222HPO/44NIL/665NO；负标签有漏标风险。候选纯publicSap41099银行Top8当前合法唯一HPO，旧1284+新增1034词面逐同SHA复用，无gold/exact强插或训练别名。1416纯F1跨度=强B∪GLiNER已接受886对应跨度，±256上下文原样。head训练F2F3，F1 proposer训练48含F0，整个管线不是pure train32。

原生完整CPU51件已回收且独立静读PASS：4347 native/HF输入、2931全mask/EOS、513完整grammar/12293prefix、15实际HFprocessor/13拒绝；122missing训练行含117OTHER+5partial，6multi-positive、44NIL和665NO均保留。训练max684/17，held max621，无截断。

唯一终态：evidence/terminal.json，对应外存R/research/mention-candidate-selection-v1/terminal.json SHA4933cceab5e040c7bd84c7f0a74364e5eb5ba3d68130c6fb4fca9d5eebfd69d9；共冻095a0ea12c56718f450539cbfc013ed364179f1629d9f4c6a0d9f2f386d37b80→exact2v2/旧0，原dev/official0。完整原件gpu-v1-collected，入口gpu-v1-collection.json；大模型状态保留远端精确locator，owner完整张量重开，不声称独立下载或实际中断resume测试。

归因：相对原2353，public候选提示/多选目标和578负例共同改变；相对2931自由名称，候选任务和Torch主机版本也不同，不能称纯prompt逐位消融。旧自由名称/旧Biomed CE/旧9B veto负结论保留。新收益或失败只属于本固定配方；不因本F1改候选、模式或阈值。

下一步：完整终态已交root，同一集中增量审在进行；唯一87件345388B小包已15:02:47.303182完成server全成员SHA及完整回读，SHA6e564058，evidence/terminal-backup.json为独立receipt，原terminal4933ccea不变。root后继授权有界只读DeBERTa-v3-large查重/成熟span实现/公开资产与成本计划；不新GPU、不大下载、不改本线冻结终点。
