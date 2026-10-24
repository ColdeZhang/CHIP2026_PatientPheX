# 方法边界与风险

原raw候选没有label字段；probability是另一16篇训练的词面Sap向量LogisticRegression表型存在分数，similarity是合并公共/该fold训练别名bank最近邻余弦。exact标记也包括train别名，不能称纯公共字典。旧slot/CE的0标签是exactspan+ID不匹配，不等于临床NO。

6224未匹配重叠中1886候选ID与所重叠正标相同、431精确词面在别处有正标（归一词面443），不能全部设NO。当前过滤保留crossing-only并排多注释/NO/NIL/outbranch/compound、词典exact、全train32归一同词面正标及同ID；剩809，固定上限选539。原合法嵌套/相同span多ID目标完整保留。仍有非穷尽注释风险，BOUNDARY是训练注释支持语义，不是人工确认临床错误。

F1跨度proposer48祖先含F0；新头/539 OOF来源排F0和F1，但整个管线不能称puretrain32或F0隔离。原dev/官网不执行。

## 完整负结果的适用边界

3470固定目标完整训完后，独立BOUNDARY模式仅出现3/1416；这只能说明当前解码行为，不能等同held边界召回或证明539训练样本无效。相对Top8并集净FP少12但TP数未增，仍24TP/100FP增量且未胜强基线。不要用相对旧负模型的改善替代强门，不据此修改cut/挑子集或补F0。HPO/NO/NIL/BOUNDARY四模式、旧目标保留及纯训练侧选样均有完整原件，可为后继机制提供证据；新颖性仅Top8多选任务中的独立边界监督，旧9B实际454重叠负训练历史保持。

## 同头 F0 迁移时的核对

要区分训练候选祖先与推断 proposer。F1 的候选生成器含 F0，不代表3470头训练含 F0；实际两个 train16 互换 OOF 来源只联合 F2/F3。本次独立 veto 只需要替换为排F0的强基线1085跨度，不需要重复训练，也不需要额外推断未被强基线使用的GLiNER跨度。主头两个负端点不迁F0，root独立正veto另立固定控制。

数量相等不足以证明覆盖：新F0准备额外核 baseline实体与held候选的 (pmc_id,offset,length,text) 精确集合双射，1085无遗漏/多余。这个纯输入证明51cdf15f补齐旧veto源码依赖的prepare覆盖，不读取模型输出或改已冻结规则。

固定F0 veto删4中3FP/1TP，menF1虽升但docF1下降、总Score略降；“删除多数是FP”不等于总指标提高。严格保留两折结果而非把F1小正外推原dev。
