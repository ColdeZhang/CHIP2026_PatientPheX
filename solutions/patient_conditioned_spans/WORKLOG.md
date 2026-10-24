# 患者条件跨度路线台账

当前：E-20261024-PCS-001，完整CPU几何、实际生产autograd和原生HF分词合同已通过，GPU尚未启动。独立分支 `explore/patient-conditioned-spans`；只拥有本目录。

目标是改善患者归属与候选池外漏概念，检验整文共享编码、患者条件跨度与概念级 max-MIL；不重复已负的公共候选对 DocRE 或无患者条件 GP。

- 依据：[实验方案](experiments/E-20261024-PCS-001.md)、[注意事项](NOTES.md)、[证据索引](evidence/cpu-ready.json)。
- train32 F2+F3 / pure F1：87 / 39 患者，所有文献、锚点、全文窗口均保留。
- CPU-v2：813 个可监督正袋、1 个 masked 正袋；3 个不精确词片正跨度的覆盖候选不作为 NER 负例。完整数目与口径见实验。
- 下一步：名称路线已12:36:50.561766真实释放 .100 GPU0；已核公开BiomedBERT所有SHA，按固定120+30秒整文双形状工程更新。CPU原start UNKNOWN已恢复同PID完整终态，无重启。
- 禁止依赖 held gold 调 cut；未训练、未推断、未评分。原 sampled 成功记录及所有旧路线保持原样。

事前修订：86b37baf 的 balanced-BCE 准备保留且未执行任何 GPU/质量。为保持本次关键变量为患者条件和 latent bag，正式 NER 改用原 GP 的 multilabel categorical loss；完整几何不变，新增高分稀有负例影响/有限梯度 CPU 合同。
