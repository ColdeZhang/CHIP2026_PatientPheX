# Copy-grounded decode 台账

当前状态：**独立fallback-v2完整F1已正完成：.704349598608，Δ+.003024221396；V1仍rc1/0quality。GPU已释放，待主线程决定排F0新资源的后继**。准备结论见 [REPORT](REPORT.md)，本地证据 [readiness](evidence/readiness.json)，实际 [CPU 合同](evidence/actual-cpu.json)，[真实启动](evidence/GPU-start.json)，唯一 [primary 协议](evidence/protocol.json)；实验 `E-20261024-COPY-GROUNDED-001`。

最近远端检查点：2026-10-24 15:14 UTC，`origin/explore/copy-grounded-decode` 的 `539c280352d048ba2a581fd763a31d2cc358e5ab` 经 `git ls-remote` 与本地一致；本行提交可领先该回执一步。

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
- 14:56:04.120818 实际唯一正式启动：`proc_4367dbef-1163-4dd1-8778-02436c18a636`，driver2748471/child2748482；GPU0 fresh=0 MiB/24164 MiB free，无其进程，GPU1 foreign保持。TERM15:26:04.120813/KILL15:26:34.120813，不重启/加时。运行源与计划在启动前 `544295b3` 推送核验；该提交已冻结任何无EOS/cap/非法行阻断质量政策，完整39失败不得冒充pipeline漏行或临时改为abstain继续评分。
- 后处理只改namespace/新source/schema绑定与既定同意公式，复用G严格定位、train32 alias、b59语义政策和原FP16 Sap。`collect_inference.py` 只在原child结束后冻原件ZIP；`postprocess.py contract` 已核旧字面前缀/非关系字节保护及其他旧流支持的新G能入primary，未读gold/调用模型/评分。

- 15:08:10 V1原child完整39后rc1退出：37EOS/37strict、2cap1600、完整EOS parser failure0；native723.208秒/wrapper725.933秒。GPU0实际15:08:10.053941=0/noapps。原39raw/IDs共89件逐SHA/fsync收齐，collection d28367ba；原V1不归一化、不评分、不改PASS，见 [关闭证据](evidence/V1-closed.json)。此前仅末尾进度见1cap，完整统计更正2cap。
- 父线程在无新gold/质量下批准独立 [fallback-v2](fallback_v2/decision.json)：这是看到无标签生成失败后的后设整行弃答策略，非原预锁V1通过。所有cap/nonEOS及原strict失败贡献空new，37完整strict行沿原parser；全16/39保留B4。只评既定B4∪validnew∩(G∪S∪Q∪C)，不重生成、不扩1600、不救前缀、不另评有效子集。源/decision先push，后仅必要FP16 Sap180+30与完整freeze后至多1newv2/old0。

- 独立v2 source/decision在da4c5a9c先push；提取37complete/2whole-row-abstain=498 anchors/333quotes、0rejected。实际142必要query plan2d3c22e9在539c2803先push后15:14:29.395233唯一启动；driver2749626/child2749637，8.243秒rc0，15:14:37.640730真实GPU0=0/noapps，13原件collection f0a89b5a。
- 完整16/39/1189的唯一primary先freeze0b5ea454，后1newv2/old0：Score .704349598608（B4 .701325377212，Δ+.003024221396），新增32=9TP/23FP、删0，mic下降/mac上升；完整终点见 [evidence](evidence/F1-fallback-v2-result.json)。原V1和原39raw保持；无模型重跑、无cap救回、无有效行子集/单臂质量。原dev .8尚未达标，本F1不冒称原dev。
- 交接入口：仓外 `research/copy-grounded-decode-v1/fallback-v2/final-review-entry.json`；代码 `fallback_v2/postprocess.py`（已执行extract-bind→normalize-bind→assemble→score，禁止重复score）。F0需排F0新pool及匹配B4/G/S/Q/C，由主线程决定，当前不启动。

- F1最小包186758B/a5e63746原write回执UNKNOWN；只读确认原目标已同SHA后fsync文件和祖先、六个≤32768B块实际回读全部成员吻合，无重传。回执ec20bb53见 [备份](evidence/F1-minimal-backup.json)。
- root批准同后设整行弃答协议转F0：全47/16/947，原B4 343/.728195628911；相同GLiNER2.5 task1 final10缺F0输出，现成54a88544权重，真实172窗62940 word tokens，先唯一次600+30与原accepted归一化必要Sap180+30，再完整47 trie2400+30。strong fold0现成79d3568e且全部训练祖先排F0，GLiNER/alias32=F2F3。F0 source/pool准备中，暂无新GPU；仅复用stage8相同输入17f9607e，不复用其异模型预测。

- F0同GLiNER172窗原唯一job proc_a87b4ea4、child2750169实际15:22:28.700565启动，原600+30；15:22:55.337158完整rc0、26.639秒，GPU0 15:22:55.352522真实0/noapps，GPU1他线不碰。1207 selected，原ZIP3355845B/cb201713回读中，尚未accepted pool。
- document_relations交F0后处理源码与两个合成合同，source-manifest ae6cb6dd。本线只读实核原315继承B⊆B4/S/Q、16/47/947与343B4；旧G/S/Q/C支持union在B4外141，见f0/old-support-contract.json。无F0新质量。
