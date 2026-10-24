# 条件 full64 部署准备

只读准备，尚未授权原 dev 模型运行或评分。若 F0 独立或 root 固定融合过门，可复用原 normal full64 `1a703b24…`，同两个固定种子产生128条；不训练新模型。B901最终保护由root融合决定。

已核 normal64 原提示 `e09bcc41…` 是500字符局部上下文，原生长度最大5194，**不是全文提示**。本准备使用已审 append-report 构造器，在保持全部候选/PID/原文本逐字不变的前提下追加完整患者标记原文；新 `expanded64.jsonl` 在仓库外，64条/16文献/36患者/466概念/1006候选对。不可把旧短提示token统计当成新全文预算。

`prepare.py` 检查 full64 模型、四折训练候选、held候选/门控全部祖先与原dev16无交集，且完整文本去除追加后恢复原提示。`tokenize.py` 仅在隐藏CUDA的既有native运行库中核真实完整输入 IDs/正常化往返和现成1a703b适配器SHA；不加载NN。产物 `R/research/nonthinking-consensus-full64-readiness-20261024/`。完整CPU token形状及候选预算等待实际回执；不自动启动GPU。
