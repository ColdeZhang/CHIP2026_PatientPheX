# 查重与目标边界

开始前已读main MAP（2026-10-24快照）和H59，精确source refs见evidence/history-check.json。现有DocRE、患者身份bridge、局部CE/occurrence-MIL/pair-query以及PCS，都在单pair CE/BCE或max-bag后独立BCE上训练；未找到同文同literal概念的多患者MLCCE实际终点。旧GP使用该公式做跨度检测，不等于已有本患者分组控制。旧prompt“patient competition”是输入/输出组织，不是此loss。

复用既存vendor.gp.multilabel_loss（Su等GlobalPointer，arXiv:2208.03054，bert4keras参考Apache2.0；本地实现）的pos/neg logsumexp＋zero。矩阵维度改为概念×患者，valid只置原可表示bags，概念key保原(hpo/nil,literal)tuple。包含masked-only概念的零项，明确全C平均。没有nearest或单患者假设，没有新标签或外部语料。

原独立BCE只是被父loss计算后立即丢弃的标量，不参与backward；原bag logits与max实例完全一致。新loss自然尺度及每个概念的权重与旧BCE不同，是此次唯一目标机制的一部分，不增另一自由权重。
