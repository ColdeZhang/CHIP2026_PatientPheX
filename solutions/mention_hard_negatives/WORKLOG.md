# 当前状态

2026-10-24 UTC；branch explore/mention-hard-negatives，base23b053e6。只改solutions/mention_hard_negatives。

- 真实目标：补足生成头拒绝误提取跨度的监督，判断是否带来强Task1增益；不把训练成功当质量改善。
- 约束：训练/候选祖先仅F2F3，公开Qwen3-8B，沿既有三状态prompt/trie/原native训练器，不读heldgold、不新增GPU。旧2353 run proc_d77保持独立进行。
- 固定样本：2353全部对象/顺序保留，按每文ceil(HPO+NIL/4)追加578；2931总行=2222HPO/44NIL/665NO，无不足配额。每个新增跨度重新核与全部正式注释零重叠、原quote/offset一致。
- 两轮拟734updates/367每轮/尾3/warmup37；实际每epoch823233tokens/25797监督；新增max369/3，全数据仍383/28，Pareto与原三形状完全相同。完整F1仍固定1416跨度、强B并集/新头替换两端点；达到任一严格胜.728697536469才候选。
- 源复用：reuse/mention全部原字节，manifest记录精确origin/SHA；只有prepare与CPU新增578行检查是新代码。已有2353编码、1416纯输入及processor/真实FB证据只引用，不重复验证。
- 外存R/research/mention-hard-negatives-v1/prepared-v1/manifest.json；CPU proc_b044e900已12:35:48.589 rc0：实际7.410s、wrapper7.965s，CUDA未初始化；只新578行native/HF/掩码和1156答案prefix，旧编码/实际processor/FB证据复用。30原件ZIP40901579/1169122B全SHA回收。按原实测保守公式新总预算估4521.224秒，未新训练/质量。

当前完成CPU readiness，见evidence/cpu-ready.json；源码9801d488已push核。等待root结合原2353完整质量和空闲GPU安排后继，不能自行把CPU ready当训练授权。
