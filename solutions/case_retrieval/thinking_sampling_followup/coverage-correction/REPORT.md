# 冻结 4096 tokens 的覆盖位置

只核原生成文本的覆盖/结构，不判断任何候选答案是否正确。原 raw、IDs、tokenizer 未改；无模型、gold、重分词或重试。

| CID | 首次 literal 完整出现的生成 token | 首次明确列表决定行结束 token |
|---|---:|---:|
| C000 | 40 | 374 |
| C001 | 384 | 421 |
| C002 | 431 | 470 |
| C003 | 480 | 3529 |
| C004 | 1392 | 1469 |
| C005 | 1479 | 1554 |
| C006 | 1377 | 1642 |
| C007 | 44 | 1721 |

全部 CID literal 覆盖到 token **1479**。C000/C007 在开头范围表达中出现，literal 覆盖不是逐项作出决定。

初轮逐候选段落结束在 char7266，对应 token **1721**；此时只有7项明确 Result 列表。C003在初轮仍反复讨论；旧 behavior-diagnostic 所谓8项在refining之前已有明确决定过强，应更正。char7268 是编号3 refining小节起点，亦不能据此称是重复首次开始。

C003 首次清楚肯定映射的完整句（‘Is it safe to map C003 to P1? Yes…’）结束于 char9304/token **2214**。若接受这种自然语言映射作为决定，全8项到此覆盖；它仍不是列表/JSON。

C003 首次明确列表行（‘So C003 should be [\"P1\"].’）结束于 char14627/token **3529**，至此首次全8项均出现明确列表决定。首次连续、紧凑的全8项列表复核结束于 char15069/token **3704**。这仍是思考文本，后续继续复核，没有自然闭合或最终JSON。

计数定义：token编号从1开始；字符位置是Python Unicode字符的0-based半开边界，字节位置是UTF-8的0-based半开边界。用固定tokenizer vocab作ByteLevel逆码，每个实际generated ID对应一段字节，累计到完整事件末尾取最早token前缀；若事件末尾落在token中间，完整token可能包含少量后续字节。证据保存每个端点的实际ID、byte/char区间、越过事件末尾的字节数。全部4096 IDs逆码与原response逐字节一致。

可据此描述这一条原输出的行为，不能推出80条的新预算已足够、质量有效或应在某一个token强制结束；单条覆盖事实不代替固定80预算合同。完整来源SHA、可复现算法与三种不同覆盖定义见 evidence.json 和 analyze.py。
