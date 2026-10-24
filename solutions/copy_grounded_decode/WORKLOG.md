# Copy-grounded decode 台账

当前状态：**CPU 准备完成，未启动模型/GPU/质量**。建议主线程决定是否进行一次完整有界对照，见 [REPORT](REPORT.md)。最终证据 [readiness](evidence/readiness.json)，唯一未执行 [primary 协议](evidence/protocol.json)；实验 `E-20261024-COPY-GROUNDED-001`。

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
