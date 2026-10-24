# 依据与边界

论文 Sounack et al. BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP, arXiv2506.10896，原厂repo lindvalllab/BioClinical-ModernBERT@fb40fb61。模型MLM权重MIT；论文下游Phenotype监督实验与发布MLM基座分开。官方preprocess只读CSV notes文本；未获得逐源原始预处理，不把病历文本等同人工表型标签，也不宣称掌握全上游样本。

模型原始PubMed/PMC预训练可能包含竞赛文献正文，无法逐文献证明排除；本项目监督及派生资源严格排F1/F0。不得据医学名称保证提升。

已有GP正span几乎全能被384窗表示，旧overflow未丢全篇token。新增是32训练整篇跨段注意；短窗FLERT真实负结果需并列。跨段重复同ID/词面1492→384窗可见468，只是上下文供给，不是错误归因或质量。

ByteLevel offset包含前导空白：只去除token边界外空白并显式映回原passage，不改变原文。1条训练词内标注不可表示要保留覆盖说明，不扩成伪标签。

SDPA原实现会建立两个L²mask，不能仅凭149M参数断言24GB必够；工程含真实驻留Adam与两极形状。不得打开全层hidden/attention或MLM词表logits。
