# 方法与来源

- [Self-Consistency](https://arxiv.org/abs/2203.11171)、[SelfCheckGPT](https://arxiv.org/abs/2303.08896)、[Universal Self-Consistency](https://arxiv.org/abs/2311.17311)：公开网页已缓存/SHA绑定原只读研究sources。这里用固定关系三元组对齐，无额外judge；不把CoT论文结论直接外推。
- Qwen/Qwen3.5-9B：HF固定c202236235762e1c871ad0ccb60c8ee5ba337b9a，ModelScope aa2aaf6060dd45e8ca24da2d37ffb0172eb985d6；运行字节相同，LICENSE以HF固定文件bbedc3fd为准。官方general nonthinking T.7/p.8/k20/minp0/presence1.5/repetition1；本控制明确保原任务presence0，不称完整官方preset。
- 实际text base8,953,803,264＋同一adapter=8,997,081,600，单模型<10B。bundle未用vision/MTP权重在loading证据披露，不改比赛计数规则。
- vendor两文件逐字节复用既有native入口：probe_runner.py的严格parser/持久writer，legacy_runner.py的数据/资产校验。导入不调用旧main；不会训练或跑旧probe。
- 原只读报告R1名称笔误已由root更正：历史S27实际DeepSeek-R1-0528-Qwen3-8B，非Distill-Qwen7B。此报告不作为原模型名的证据；新控制始终Qwen3.5-9B。

F0固定主方案复验正（+.010462），新增21仅9TP；与F1新增34的16TP一致，收益机制是完整保护B后补充召回，不能称新增高精度。F0候选32→48及A30硬件变化已披露。条件full64另已发现旧normal64是短提示，必须构造真正全文。CPU脚本不得命名tokenize.py：此次遮蔽stdlib导致torch import失败，已更名audit_inputs.py、独立目录恢复，0GPU/0tokenize原失败完整保留。
