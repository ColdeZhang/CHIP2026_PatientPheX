# 实施笔记

已有GLiNER span BCE/focal、旧强池供给、GP对照都真实做过；新变量是GLiNER2.5公开预训练boundary/shared-pool/reranker头及原生多项loss的Task1监督适配，不能把模型新、候选多或工程通过称收益。完整机制比较及源见继承CPU准备报告。

原生trainer即使strict=True也会遇OOM后continue；本线在原生保存failed-batch后立即raise。原生loss内部nan_to_num会隐去非有限分项；透明观察wrapper先保存实际raw/finite，非finite停止，有限值继续同一原函数。原生训练计算、optimizer、loader与候选loss不替换。

公开loss稀疏采样：所有NO不是强制pair负例，实际仅候选/硬负采样/query mask进入pair loss；probe记录每次实际indices/mask和NO覆盖。Dense目标CPU合同不能代替native loss/FB/Adam。
