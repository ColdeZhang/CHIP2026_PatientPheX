# Copy-grounded decode 台账

当前状态：**.100 实际 CPU 门通过，GPU39 正式计划已冻结，待推送/新鲜空卡门后唯一启动；尚无新质量**。准备结论见 [REPORT](REPORT.md)，本地证据 [readiness](evidence/readiness.json)，实际 [CPU 合同](evidence/actual-cpu.json)，唯一 [primary 协议](evidence/protocol.json)；实验 `E-20261024-COPY-GROUNDED-001`。

最近远端检查点：2026-10-24 14:47 UTC，`origin/explore/copy-grounded-decode` 的 `b00734176a5d8ec7d2eb372df628fdf38e26b7fc` 经 `git ls-remote` 与本地一致；本行提交可领先该回执一步。

- 2026-10-24：从 main `6f66672f75b23c6c9af2094765c335fde6717650` 建 `explore/copy-grounded-decode`。只做 CPU/no-gold 准备；不训练、不加载模型、不评分、不提交官网。
- 目标：保持 Grounded 9B train32 epoch2 d52 权重、39 完整 prompt 和三元组语义，只将自由文本生成改为有限原文三元组的 native-token trie 解码。一次候选为 `B4 ∪ (constrained_G ∩ pure_G)`，是否运行由主线程决定。
- 预先固定供给：既有 strong Task1 + GLiNER 1416 exact spans；按原文 offset 映射为 `[unit_id, exact_quote, occurrence_index]`，只去重同一三元组，不按患者距离挑 anchor、不补短语、不扩全子串库。候选不加入 prompt、不含预测 HPO ID。
- 约束：完整 JSON、EOS、无重复三元组；cap/未知前缀失败，不截断/补全答案；未来先保存 raw token IDs 和 native raw text，再解析与固定归一化。解码候选有排 F1 的 48 文档 B 祖先，整链不称纯 train32；F0 需重建排 F0 资源。
- 先核原格式、无 gold 供给和 token 合同，再提出成本/执行建议。完整样本和公开源码放仓外 `.../research/copy-grounded-decode-readiness-20261024/`；仓内仅代码与摘要。
- 最终 1416 exact triples / 907 文档内不同 quotes；39 prompts 原 SHA 未变，原文坐标逐项通过。native 合同 1839 串/62159 callback tokens，1.430 秒 CPU；零模型/fit/质量调用。
- 主线程在新预测/质量仍为零时将 G-only 改为 `B4 ∪ (constrained_G ∩ (G∪S∪Q∪C))`，原决定保留于协议。B4 外支持空间 62→133，额外 71 中 66 有候选 proposer 同文档预测 ID 对应；仅供给代理，不能当质量。
- 历史最终 G39 为 1103 秒级；CPU 准备不代表实际 HF generate 已验。下一步限主线程判断是否给有界完整推断，再按 raw-first/固定 normalizer/唯一 primary 流程执行；F0 另换排 F0 资源。
- 交接前补查原 runner，确认 all-finite guard 与前缀 `-inf` mask 有顺序风险；将“raw finite 检查先、trie mask 后”的实际接口合同加入协议，未删数值检查，未更改已通过的 native grammar 源码。
- 后继授权：.100 Torch2.7.1/TF5.20.2/PEFT.21.2 原 runtime，CUDA 隐藏做真实 CPU processor＋原39IDs/template 合同（120+5）；通过后同 d52 完整39 greedy/nonthinking/max1600、1800+30。原 A100/Torch2.6 与本次 A30/Torch2.7.1 差异披露，不称纯decoder数值因果。
- GPU0 须等 sequence 的 F094 和必要 Sap 最终真实释放，再 fresh gate 唯一启动；不会凭旧F1 release/空卡表抢占。后继固定原归一化政策 b59/train32alias/41099 bank，必要 FP16 GPU query 180+30；只冻一份16/39/1189 primary，再一次v2/旧0或零增量byteexact0。F1严格正才考虑F0及新的排F0候选资源。
- .100 CPU `proc_6dd24d19` 实际14:53:37 rc0、无超时：native7.527秒/wrapper8.390秒；234个真实processor调用状态通过，原39完整203588 tokens/模板/双decode一致，逆序mask→finite确实失败、正确顺序保留原logits并拒绝NaN/±inf。结果 `8c2a642b…`，CUDA未初始化、无模型/forward/generate。
- sequence 已确认原F094及195必要Sap结束，14:52:39.579562 GPU0=0/noapps；原wrapper `7dfef780…` 已绑定 GPU plan。正式源 [infer.py](infer.py)、[infer-plan](infer-plan.json) 保原同d52/39、greedy/nonthinking/max1600/1800+30；全raw持久后才逐token与完整三元组校验，任何无EOS/cap/非法行阻断质量。
