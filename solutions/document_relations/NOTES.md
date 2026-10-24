# 方法与来源

- Zhou et al. AAAI 2021, [Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling](https://arxiv.org/abs/2010.11304)，官方代码 [wzhouad/ATLOP](https://github.com/wzhouad/ATLOP)。2026-10-24实际读取model.py、losses.py、README.md；根LICENSE路径404，故不复制其源码，按公开论文算子独立实现并保留引用。后续检查具体代码许可前不声明可任意再分发。
- 原ATLOP以实体提及logsumexp、各提及attention平均、头尾attention逐元素积得到局部上下文，拼接实体/上下文后作分块双线性分类。此次二元患者关联不需要多标签自适应排序，BCE仅作明示的二元适配。
- 原长文处理最多拼首尾窗口会遗漏中段，不直接套用；本实现覆盖所有连续token，用重叠窗口平均正文表示。每个物理提及选择包含完整跨度且上下文余量最大的窗口，避免重复窗口被当成新增原文提及。跨窗无共有attention时上下文向量为0，记录其规模，不把它冒称完全全局注意力。
- 当前新路线只复用固定发布的PubMed/Bioformer/BioBERT canonical候选；它们未在本任务标签微调，train32/F1同口径。候选空间外关联仍不可预测，最终必须披露这一限制。

- 后继查证：固定ATLOP prepro.py（SHA cc552fea…）在每个实体起止边界插入已有`*`，model.py取起始marker而非span均值。独立marker-v2据此做单一表示控制；仍不声称原论文多类ATLoss或完整长文模型已复现。全部原tokens精确保留，阈值/损失不变，质量须由完整控制决定。

- 30轮默认cut的直接并集/候选内替换已负，只关闭这组控制。AP改善不等于赛事分数改善。后续另获准做[E002](experiments/E-20261024-DOCRE-002.md)的F1文献meta-OOF筛查：每模型其他12篇标签训练、held4预测，基础来源排整个F1；它不是未见F1标签的盲验证，也不等于独立F0迁移。
- E002的固定低容量LR仍负：base-only复现完整base .68634，加入全提及分数降至.67407，删27TP/30FP、加1TP/10FP。当前候选/表示/固定线性决策未提供可部署增益，筛查已关闭。保存全部logit与OOF概率；如未来研究更丰富交互、额外证据或不同表示，应另建有明确新增机制和训练来源的实验，而非在这组F1结果上反复扫cut。
