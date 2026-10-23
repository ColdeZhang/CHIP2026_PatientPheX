# 关系连续置信度：只读机制判断（暂不运行）

现有Qwen集合生成结果**没有可直接校准的LLM概率**。runner没有请求生成scores/logits；204条训练OOF raw及既有配对raw只保存响应文本、token数、耗时、触顶等。树的四列是选中/空集/解析失败/选中人数；耗时、输出长度或人数不能冒充confidence。现有JSON token的序列似然也不等于各patient-concept边际概率，需另定义评分协议。

| 已做机制 | 真实输出和限制 |
| --- | --- |
| Qwen离散四事实→HGB | .653192→.649124；不能据此推断所有连续LLM特征均无效 |
| BiomedBERT患者关系CE | 已有patient-concept二分类softmax；train48/5114pairs/4ep，旧池.617746低于rules.656999。OOF候选分布、标记保留和9B backbone可以是新变量，但任务本身已做 |
| BiomedBERT max-MIL | doc-HPO是否属于任一给定患者；真实max-logit/sigmoid，.711354→.702753。只过滤整篇概念，不区分患者，目标不同 |
| span-ID CE概率入树 | 严格嵌套OOF84→85列，.714808→.701372；识别/链接候选真伪，不等同患者关系概率 |

成熟的[monoT5](https://arxiv.org/abs/2003.06713)以true/false目标词logits的二类softmax做连续排名，官方[固定源码](https://github.com/castorini/pygaggle/blob/ad09ec048888aaf441034d9f52525c3419f47fd2/pygaggle/rerank/transformer.py#L107)实际如此实现。可类比为固定9B基座对单patient-concept的yes/no下一token打分，保存两logits及差值；这会新增现有四facts没有的连续信息。但二词条件概率不自动等于校准后的关系概率，答案词/提示有偏置，[校准研究](https://arxiv.org/abs/2102.09690)也只提供方法依据，不证明本任务收益。现有JSON-SFT训练并非binary verbalizer训练，不能直接称其已具备同一评分头。

仅对train48的204条去标签消息做CPU token预算：冻结tokenizer和文本模板分支逐条重现原648409prefix tokens；4127个独立pair保留全部global anchors及该概念完整evidence，实际**4144716 tokens、最长2375，是6.39倍原prefix**。若每pair照抄原多概念batch，则至少14849294 tokens，尚未加问题。yes/no各单token（9405/2083）；本次没有计算logits、没有GPU、没有originaldev诊断或阈值扫描。单概念方案删去其他概念局部上下文，输入信息也变化；少解码不意味着总运行时间必然更短，真实GPU时间未知。

严格校准边界：冻结未用赛事标签训练的基座可对训练内OOF候选评分，并仅用训练侧选择阈值；若再训练二层分类器，其候选来源仍须随meta留出重新隔离。当前SFT48/64不能回推自身训练文献后声称SFT置信度OOF，需新的交叉训练/留出，成本显著增加。9B最后状态加判别头仍是患者CE目标，不能仅换backbone就称全新机制。

**保留为后续候选，当前不追加yes/no或判别头运行。** 先完成已授权full64唯一终点。详细token行、网页快照与来源SHA在仓库外，见[证据索引](../evidence/round2/continuous-relation-confidence.json)。
