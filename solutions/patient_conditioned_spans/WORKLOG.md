# 患者条件跨度路线台账

当前：E-20261024-PCS-001，完整CPU几何、实际生产autograd和原生HF分词合同已通过，整文GPU工程已完整通过，formal尚未启动。独立分支 `explore/patient-conditioned-spans`；只拥有本目录。

目标是改善患者归属与候选池外漏概念，检验整文共享编码、患者条件跨度与概念级 max-MIL；不重复已负的公共候选对 DocRE 或无患者条件 GP。

- 依据：[实验方案](experiments/E-20261024-PCS-001.md)、[注意事项](NOTES.md)、[证据索引](evidence/cpu-ready.json)。
- train32 F2+F3 / pure F1：87 / 39 患者，所有文献、锚点、全文窗口均保留。
- CPU-v2：813 个可监督正袋、1 个 masked 正袋；3 个不精确词片正跨度的覆盖候选不作为 NER 负例。完整数目与口径见实验。
- 下一步：固定fresh8epochs +完整F1 +严格/Sap规范化在共享900+30窗口运行；两完整端点共冻后各1v2。工程原120窗口实际15.558s，峰allocated3.755GB，参数109187713，两个整文形状全部真实更新通过，12:40:06.810949实际release。CPU原start UNKNOWN已恢复同PID完整终态，无重启。
- 禁止依赖 held gold 调 cut；未训练、未推断、未评分。原 sampled 成功记录及所有旧路线保持原样。

事前修订：86b37baf 的 balanced-BCE 准备保留且未执行任何 GPU/质量。为保持本次关键变量为患者条件和 latent bag，正式 NER 改用原 GP 的 multilabel categorical loss；完整几何不变，新增高分稀有负例影响/有限梯度 CPU 合同。

公开Sap41099库已.3→.100同SHA实际复制/fsync重开，1.5245秒；使用原policy .90 cosine/.03 distinct margin及全部字符串保护，无重编码。证据 [工程结果](evidence/probe-complete.json)。
