# 方法出处与边界

1. Lakshminarayanan、Pritzel、Blundell，NeurIPS 2017，[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)。实际读原文 §2.4 和 Algorithm1：各成员使用完整训练集，通过独立参数初始化与数据 shuffle 产生差异；论文并不要求 bootstrap。原方法主要平均预测分布，并含可选对抗训练。本方案共享冻结9B基座，仅独立 LoRA/随机训练轨迹、只增加一个成员，并用离散关系支持集合，不是论文的完整复现。
2. Fort、Hu、Lakshminarayanan，2019，[Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757)。实际读作者摘要：不同初始化可产生不同函数空间解；同一优化轨迹附近的点常更相近。它解释为何独立训练与同 adapter 推断双采样、同轨迹 epoch 平均不同，**不证明**这个小规模弱标任务一定受益。

原论文 PDF/摘要外存于本任务 research 目录；精确 URL、bytes/SHA 在 [readiness.json](evidence/readiness.json)。未把 PDF、全文摘录或运行日志放入 Git。

可证伪假设：新训练轨迹能补充当前 B4 漏掉、且由至少一个固定旧流支持的正确关系，收益超过新增 FP。只运行一个预先固定 seed；相同弱目标和冻结基座可能让错误高度相关。一次新 seed 的正/负结果不能估计跨 seed 方差，也不能否定整个 deep-ensemble 家族。

继承限制：Grounded 原目标的患者证据是弱监督代表 quote；四个原部分无效/不完整患者目标保持，不通过本试验重标。G 部分 train32 同时排 F0/F1；完整 B4/支持流包含各自合法候选祖先，不能称整条组合统一 train32。公开预训练是否包含这些文献未知。
