# 全文患者概念关系

当前：唯一工程已真实21.528秒PASS并释放；root已授权fresh30轮240Adam/完整F1共享2400+30，待seed必要Sap实际release后fresh启动。分支 explore/long-context-relations 起点9709b712，独立目录，只复用旧DocRE完整公共候选与已验证BioClinical原厂资产/原生IDs。Task1负方案保持关闭，不加载其训练权重。

目的：检验完整文档内patient/phenotype token交互的实际价值。旧512窗已全覆盖文本；按任意旧窗start-marker几何，新增长上下文直接交互61训练正对+1994负对，不称此前错例。原4495对/727正/818可达性边界，F1纯输入2874对全部保留。旧10组同跨度概念歧义不掩盖。

固定候选：fresh149M公共MLM，span均值/全提及logsumexp→共享block-bilinear，全部pairs无权重BCE，文档均值、accum4；30轮240Adam，encoder2e-5/head1e-4/24warmup、seed20261024，最后epoch唯一。新骨干/tokenizer/上下文/取消显式局部attention-context为联合变化，二元目标不是首次。

提议部署：当前B4 F1 .701325377212保护并集，固定logit>0；完整预测先freeze至多1v2，不单评裸模型/扫cut，严格正才F0。正式已据真实最坏成本2226.652秒授权2400+30，不增加epoch或端点。

原生CPU只复核旧已编码48篇IDs、候选/owner与微型梯度，不载模型；GPU工程六次实际FB（先累积4→真实非零Adam1→驻留Adam后累积2）覆盖7494token和616pair两极，全state重开后丢弃。最长输入7494/17pairs，最多pairs616/3068tokens；不截样本、不重做GP探针。

证据外部 R/research/long-context-relations-v1；只小源码/计划/结果指针进Git。原BioClinical GP完整终审957a2f0d与49件小包c54cdecf由root登记，不在本线重复审/复制大state。

工程实际：16:53:37.575758→16:53:59.103917 UTC，6FB/非零Adam1、140参数全变化/梯度finite、563state张量重开回装，峰3.965GB；16:53:59.254489 GPU0=4MiB/noapps。probe db5d273c仅保存丢弃。正式与probe model/common逐字相同，全部data沿原冻结；源准备不占卡，等seed短Sap交接。

正式首次纯I/O source.zip上传16:56明确LUNA_OFFLINE，无deployment ACK、无formal start intent/GPU。原probe终态有效，暂停.3调用；恢复先核原源文件，不重复模型。root静核正式与单质量源码无阻断，并指出prediction文件需先fsync再freeze：已以排他完整写/flush/fsync/父目录同步修正，配方不变、未发生质量调用。
