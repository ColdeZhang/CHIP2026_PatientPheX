# 方法笔记

- SCST：Rennie et al., arXiv1612.00563。greedy baseline停止梯度，sample概率必须与重放loss一致。公开复现固定ruotianluo/ImageCaptioning.pytorch@4c48a330，MIT；不是作者官方实现，不使用其外部数据/权重。
- MRT：Shen et al., arXiv1512.02433；GRPO：DeepSeekMath arXiv2402.03300。只作方法选择参考，不同时铺三条GPU路线。
- 集合reward必须经过真实固定概念标准化。32条训练记录已有字面不同、同概念的现成反例；不能用quote字符串匹配代替HPO集合。
- 全部818 v2 gold保留，旧G可构造814不等于训练reward只能保814。罕见/坏literal与临床否定不是一回事。
- 先预生成整轮再改变参数是off-policy；本方案每组现参数生成再更新，probe只在最后一次Adam之前保持全部四组父权重相同。
- 序列sum与逐样本length mean是不同目标。固定训练平均长度136.598只调统一尺度，CE mean另作明确锚定，不能混称原始纯RL。
- 复用DPO串行系数/replay与cache=False经验，不代表Qwen3.5策略梯度真实链已经验证；新风险在实际采样分布、EOS边界和完整normalization reward。
