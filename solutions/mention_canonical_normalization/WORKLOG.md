# Mention canonical 工作台账

2026-10-24 UTC。独占分支explore/mention-canonical-normalization，base bc939eeb；主线程目标仍原dev公共v2≥.8，当前无本线质量。

- CPU数据完整：2353 train32目标=2222HPO/44NIL/87NO；16compound、8分支外正注释显式排除。拟每侧256原字符，保整个mention，不按标签选择上下文。
- 纯F1跨度池：16篇/39患者/1416行；强B独有530、共有783、GLiNER独有103。强B源8cb61ba9，GLiNER接受源6ac9aa51；不提示旧ID/候选，不读heldgold。
- schema：非空公开canonical数组→HPO；唯一{"NIL":"原quote"}→NIL原文；[]→NO空实体。strict parser拒未知/重复/错quote/不完整，不补猜。
- 正在准备完整实际native/HF tokenizer、全监督mask/EOS/cap及三状态PrefixConstrainedLogitsProcessor CPU合同；只复用已缓存8B小资产，无模型/权重/GPU。新训练预算未批。
- 复用原件在reuse-sources.json：canonical_names真实token/trie与doc原生train/probe/common/loss；不另造训练器。新target grammar和输入要自己实际验证，不能拿旧87患者合同冒充本线通过。
- 外存：R=/workspace/patientphex-runs/round2-20261023/research/mention-canonical-normalization-v1；prepared-v1/manifest.json是当前完整数据入口。起始Git基点已由root推送，本次源码待首检查点。

下一步：source/plan push后，.100自有namespace唯一CPU合同（CUDA隐藏/线程1，120+TERM5），完整原生token/shape与成本据此给root；不直接启动GPU。
