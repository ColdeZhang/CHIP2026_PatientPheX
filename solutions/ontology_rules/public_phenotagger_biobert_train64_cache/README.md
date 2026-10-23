# 公开 BioBERT 原生 train64 缓存

root已授权第三head无标签缓存。固定原2–8gram、296984坐标/244938不同lower词串，严格独立BioBERT H5，不复用其他模型分数或关闭的unigram策略。原native>.95五位舍入、字典/重叠融合/缩写/current HPO处理及真实六来源字段保持。

CPU原文坐标/计数准备120秒；fresh A100 GPU2 UUID空卡后，唯一完整native任务480秒+TERM30，包含加载/重建候选。真实首128批核tokenIDs/32长度、finite/sum1/17266列/实际GPU，然后继续同次缓存，没有弃置probe。来源后处理CPU120秒；全部native raw保留外存，不做质量。

`deploy.py` 从本目录外存原件部署并开启CPU准备，完成后运行远端 `train64-cache/launch_cache.py`，`collect_and_verify.py` 回收必要小产物并核原文/currentmap/来源/schema。固定源、计划及实际SHA见source-index，实际目录在 `research/public-phenotagger-biobert-train64-cache/`。未来候选并集/特征选择器政策须单独事前固定；本缓存没有独立评分或选择阈值。

实际完成：244938独立词串forward157.241秒，native含准备/加载232.331秒<480，来源CPU4.123秒<120；5903有效canonical键。首批128实际finite/17266列/词表门通过，64篇全部坐标、映射、来源schema和逐文SHA核验；23:52:15.981Z GPU2为空归还。输入/完整raw与大canonical保留外存，managed进程已退出；[完成索引](completion.json)和[输入合同自核](input-contract-check.json)区分自核与独立审查。一次提前回收只触发未完成状态门，未写archive或改动任务；原任务继续完成，没有重推。
