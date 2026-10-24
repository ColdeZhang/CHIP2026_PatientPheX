# 研究笔记

- [Yao 等，Bayesian hierarchical stacking](https://arxiv.org/abs/2101.08954)：来源权重可依输入变化并部分收缩，适用于专家在不同输入区域表现不同。原作完整 Bayesian inference 与本项目拟议固定高斯先验 MAP 不同，报告明确披露。
- 作者公开实现固定 `03458d5e59a1f945e480a8546f59b56c99c43e14`。`well_switch/optim.stan` 使用 softmax simplex 权重、连续系数 normal(0,1)、加权 predictive density 的 log；`optim_mode.stan` 去掉超先验取 mode。未复制其代码到业务实现；准确 URL/SHA 保存在 evidence。
- 旧303列已有患者数、局部距离、否定、上下文及旧概率，不能把这些称作新信号。实际新供给是 S1/S2 分歧在 S∩之外的状态，且31条仍处于原 G/S∩/Q union。
- 不从 B1 delta 构造学习样本：B1 训练48含F0，会使未来 F0 端点不再独立。原完整 raw union 的旧隔离方案可复用。
- 没有打开 F1 标签验证31条正确率，故当前为可证伪假设。仅16个 meta-training 文献、来源相关、候选32→48变化均可能使条件化拟合不如稳定多数投票；单固定强对照可快速关闭。
