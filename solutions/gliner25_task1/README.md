# GLiNER2.5 Task1 原生监督适配

目标是检验预训练 boundary 提议与重排头经 train32 精确实体监督后，能否为真正强 Task1 池提供增益。原 dev 最佳仍 .754361，本线完整10轮正式训练、164窗F1抽取与固定规范化后，强并集从 .72870 降到 .72462：新增104个实体中23TP/81FP，固定配方负结案。精确结果见工作台账。

本线从 main a89f225 独立建立，原 schema 路线 E001/E002 及其负结果保持。先前 CPU readiness 中的“公开base/adapted双控制、仅disjoint/proper-nested、排NIL”是已被取代的建议。**最终固定规则：只训练 adapted；主端点为 strong8cb61ba9 ∪ 全部 normalizer 接受实体（mapped 或 NIL），仅精确原子去重，不增加重叠、冲突、NO 或 NIL 筛选；诊断 new-only 实体壳保原关系。两完整端点共冻后各一次公共v2，旧基线不重评。** 不把旧关系零样本输出当作匹配 Task1 base。

唯一配方与预算见 [plan.json](plan.json)，进度见 [WORKLOG](WORKLOG.md)。原生完整320窗目标及10轮采样身份已CPU核验，38条词内正标注无法由不变native splitter精确表达，保留审计且不扩边界。实际压力probe完成后已丢弃其权重，正式从原公开权重fresh完成1600步，final epoch10唯一；全部原始证据和检查点定位见台账。
