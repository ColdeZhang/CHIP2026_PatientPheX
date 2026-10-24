# 工作台账

分支 explore/quote-set-completion，从 main 08a12023 创建，仅维护 solutions/quote_set_completion。当前仅成熟方法查重与 train-only CPU 供给/原生token准备，无GPU、fit、held输入或质量。

目标：检验给定已报告表型时的条件补全，是否比独立重新生成更能回收遗漏患者关系。当前原dev最佳 .769235669942；本线不读其gold、不宣称预计达标。先复用原train32的87患者弱quote目标，构造两个互补known/remaining视图；原真实关系标签不变，词面代表弱锚点而非人工患者证据。原双采样F1正/F0负已在旧分支关闭，不能重复其采样共识。

公开来源已实际HTTP200缓存Self-Refine、Levenshtein Transformer、BART及内在自纠错的反证论文；只借成熟条件去噪/插入目标，不宣称复现其模型或公开收益可外推本赛。CPU计划plan.json先固定，完整供给/长度核后再给具体成本与单一验证方案。训练与GPU尚未获授权，不占A800。
