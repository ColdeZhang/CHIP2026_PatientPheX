# 患者条件跨度研究与执行决定

REPORT、sources、history、census、READY 保留原始研究字节；其中“尚未授权”描述研究完成时点，现由本文的root决定接续。

Root依据训练数据和实际查重，批准一个新增配方：共享BiomedBERT编码＋患者条件跨度＋患者/概念max-MIL监督，辅助全局NER，两个loss初始1:1；不把最近或全部同概念出现当作已证患者证据。独立分支 `explore/patient-conditioned-spans`，起点 `267e6ea6`，负责人 `/root/span_normalization`。

先完整CPU覆盖、坐标/跨窗/空患者/多标签及真实梯度合同。使用现有公开模型与train32，F1完整输入但不读其标签准备。预留.100 GPU0，须先确认名称生成原作业实际释放；唯一120+30工程据实核预算。初始8ep/末轮/固定seed，无epoch或cut搜索；若真实成本可容纳则正式900+30，必要调整先据实记录。两完整端点为强B所有关系并入新关系，以及实体仍同B的新关系单臂，先共同冻结再各一次v2；胜B .686338981272才候选。原dev/官方尚未扩展。

大产物仍在仓库外；完整过程由[方案分支](https://github.com/ColdeZhang/CHIP2026_PatientPheX/tree/explore/patient-conditioned-spans/solutions/patient_conditioned_spans)维护。BioMistral仅低优先级元数据备选，未下载权重或开训。本决定不构成质量收益结论。
