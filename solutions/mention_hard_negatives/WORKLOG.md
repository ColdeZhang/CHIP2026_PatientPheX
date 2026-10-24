# 当前状态

2026-10-24 UTC；branch explore/mention-hard-negatives，base23b053e6。只改solutions/mention_hard_negatives。

- 真实目标：补足生成头拒绝误提取跨度的监督，判断是否带来强Task1增益；不把训练成功当质量改善。
- 约束：训练/候选祖先仅F2F3，公开Qwen3-8B，沿既有三状态prompt/trie/原native训练器，不读heldgold、不新增GPU。旧2353 run proc_d77保持独立进行。
- 固定样本：2353全部对象/顺序保留，按每文ceil(HPO+NIL/4)追加578；2931总行=2222HPO/44NIL/665NO，无不足配额。每个新增跨度重新核与全部正式注释零重叠、原quote/offset一致。
- 两轮拟734updates/367每轮/尾3/warmup37；新native token和Pareto尚待实际CPU，不能用字符数冒称GPU成本。完整F1仍固定1416跨度、强B并集/新头替换两端点；达到任一严格胜.728697536469才候选。
- 源复用：reuse/mention全部原字节，manifest记录精确origin/SHA；只有prepare与CPU新增578行检查是新代码。已有2353编码、1416纯输入及processor/真实FB证据只引用，不重复验证。
- 外存R/research/mention-hard-negatives-v1/prepared-v1/manifest.json；先source/plan push，再CUDA隐藏CPU120+5，完成后报真实形状和新总预算给root。
