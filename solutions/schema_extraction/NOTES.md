# 方法笔记

本路线研究已预训练关系头和schema驱动边界模型的增量，区别旧GLiNER实体头与from-Biomed DocRE。完整预训练样本谱系未公开，不声称纯架构因果。

根线程来源见main docs/research/multitask-relation-readiness-20261024。GLiNER2上游2.1.0通过AutoExtractor按boundary加载，旧GLiNER2类仅span，不能混装。公开loader按strict state_dict加载。max_len为word-token相关接口，真实wordpiece/文本覆盖须通过实际processor确认，不能据4096配置承诺完整报告。

全候选与原分数外存；患者头必须映射原input明确mention坐标/身份，不能用训练或heldgold补病人。规范化固定strict/Sap政策，未知结果保持拒绝并计数。
