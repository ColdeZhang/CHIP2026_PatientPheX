# P-20261023-PUBLIC-PHENO-RAW-CENSUS

全输入CPU普查29.178秒、外层29.787秒完成，三个raw全SHA核验，源码367a29bb/计划316c01b8，summary74dd81f2。先写完无标签全64计数，再单独读已冻结217个FN诊断，不从错误查询挑正式候选，没有训练或质量输出。

|机械供给|新增全局跨度|四fold累计池行数|最大train倍率|原50.21秒按行数估计|
|---|---:|---:|---:|---:|
|原top1非None/current-valid|7408|299265|1.228|61.4秒|
|原top1..3 current-valid|262699|4939477|20.470|1013秒|
|原top1 5dp>.95/current-valid，绕过hybrid|1572|252014|1.032|51.7秒|

旧总数244794；C全局7397keys/7355spans，旧全部70615keys/61293spans，加1778新keys。上述耗时只为代理，未真正fit。A规模可控但引入低置信；B过大；C改变真实边界供给而成本小，root已仅授权C正式控制，另记[协议](../public_phenotagger_prehybrid_control/README.md)。

217准确坐标的实际淘汰阶段：PubMed top1 None78/原门110/hybrid29；Bioformer None74/原门120/hybrid23，abbr/currentmap0。正确linked top1在hybrid分别23/17，并集27；本次只定位hybrid阶段，没有重建具体竞争获胜项。

区别：已负lowergate在PubMed native>.5后仍经hybrid/abbr，新增canonical不是原raw超集，且当时基础池未含本次同Bioformer组合；不能当C的同基线单变量结果。已负top3只在原边界扩ID，不能触达本次1572新跨度。C保持原门槛，直接向选择器暴露native合并前的真实供给，后续质量需单独验证。
