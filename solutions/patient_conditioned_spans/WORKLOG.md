# 患者条件跨度路线台账

当前：E-20261024-PCS-001，CPU 全量几何已完成，GPU 尚未启动。独立分支 `explore/patient-conditioned-spans`；只拥有本目录。

目标是改善患者归属与候选池外漏概念，检验整文共享编码、患者条件跨度与概念级 max-MIL；不重复已负的公共候选对 DocRE 或无患者条件 GP。

- 依据：[实验方案](experiments/E-20261024-PCS-001.md)、[注意事项](NOTES.md)、[证据索引](evidence/cpu-ready.json)。
- train32 F2+F3 / pure F1：87 / 39 患者，所有文献、锚点、全文窗口均保留。
- CPU-v2：813 个可监督正袋、1 个 masked 正袋；3 个不精确词片正跨度的覆盖候选不作为 NER 负例。完整数目与口径见实验。
- 下一步：现成 BiomedBERT 精确资产、远端生产代码 CPU autograd 和 HF token 合同；等待名称路线真实释放 .100 GPU0 后，唯一 120+30 秒整文双形状工程更新。
- 禁止依赖 held gold 调 cut；未训练、未推断、未评分。原 sampled 成功记录及所有旧路线保持原样。
