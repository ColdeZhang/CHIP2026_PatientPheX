# 条件 full64 双种子部署准备

已完成 CPU，尚未运行原dev模型或评分。若root保留F0独立/融合控制，复用normal full64 `1a703b244b…`（config `c4a478a7…`）即可，不需重训或下载；实际.100资产SHA已核，native版本同F0/F1，single模型8,997,081,600参数。

原normal64输入 `e09bcc41…` 为局部短提示（202743 tokens/max5194），不能冒称全文。本次通过已审append构造，保持64行所有CID、PID及原短提示严格不变，再补全每篇患者标记原文；新expanded SHA `4621fec59e…` /2,126,088B，16文献/36患者/466概念/1006候选对。生成full64及其四折训练候选、held CRF/Sap/GLiNER与门控全部排原dev16，见preparation-v1/complete.json。原dev历史参与研发，仍不称盲测。

实际native CPU 7.043秒：AutoTokenizer/原生tokenizer所有64行IDs相同，published normalizer往返exact。每轮512091 tokens，min3357/max13230（index1），双seed128 fresh总1,024,182tokens。可保持input gate14400、generation1600，固定20261024/25每row SHA派生，T.7/p.8/k20/minp0/repetition1/presence0，无截断、修复或重试。

A800按F1实际728.463秒和输入量比例粗估557秒；A30可候选2100+30窗口，须待真实资源/最终计划再次固定，不能把估计当授权或吞吐保证。当前无GPU意图。fulltext native/SFT32在内部折的已验证策略迁移至full64，模型训练量变化明确披露；patient64/reverse/public42不在新增采样范围，root负责最终B901保护。

首CPU脚本名tokenize.py遮蔽标准库，在torch导入时退出、0tokenize/GPU；原失败完整保留，独立source-recovery/audit_inputs.py恢复成功，不曾更改F0 GPU或复跑其响应。实际入口/模型/输入/源与错误SHA均在readiness.json；原devgold未读。

外部完整报告/refs：`R/research/nonthinking-consensus-full64-readiness-20261024/`。
