# 新全文训练模型 F1 两端点

只机械继承原完整80的合法CID/PID、raw-first证据核、实体/非关系字节保留和共冻→两次v2入口。新模型全80 fresh；强对照是旧短训练模型的全文推断单臂 .676546354662594。Primary 为新模型单臂，diagnostic 为旧短输入SFT32并集；各臂按总Score严格升保留，macro完整报告。两臂均升取Score高者，不改变原身份，不自动迁F0或dev。

`unbound-template.json` 无法执行。训练两epoch持久ACK绑定推断plan后，`prepare_disabled.py` 才绑定新的推断源码/plan（仍disabled）；唯一完整推断终点到齐后，运行 `bind_completed.py` → `evaluate_full80.py`。新质量前共冻两完整预测，旧基线评分复用，未额外生成或重解析模型原文。
