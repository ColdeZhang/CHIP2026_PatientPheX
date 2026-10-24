# 全文患者概念关系

**负结案。** 原fresh30ep/240Adam及全部F1完整完成，唯一B4保护并集 .701325377212→.664715835189（−.036609542024），新增181=14TP167FP；micro/macro均降，Task1及全部旧关系保留。冻后1v2/旧0，关闭不F0/不扫cut/不加轮。已释放GPU，终态小包实际备份/回读完成，等待root批次独审。

分支 explore/long-context-relations 起点9709b712，独立目录，只复用旧DocRE完整公共候选与已验证BioClinical原厂资产/原生IDs。Task1负方案保持关闭，没有加载其训练权重。完整结果见experiments/E-20261024-LCTX-REL-001.md；外部R/research/long-context-relations-v1。

目的：检验完整文档内patient/phenotype token交互的实际价值。旧512窗已全覆盖文本；按任意旧窗start-marker几何，新增长上下文直接交互61训练正对+1994负对，不称此前错例。原4495对/727正/818可达性边界，F1纯输入2874对全部保留。旧10组同跨度概念歧义不掩盖。

固定候选：fresh149M公共MLM，span均值/全提及logsumexp→共享block-bilinear，全部pairs无权重BCE，文档均值、accum4；30轮240Adam，encoder2e-5/head1e-4/24warmup、seed20261024，最后epoch唯一。新骨干/tokenizer/上下文/取消显式局部attention-context为联合变化，二元目标不是首次。

提议部署：当前B4 F1 .701325377212保护并集，固定logit>0；完整预测先freeze至多1v2，不单评裸模型/扫cut，严格正才F0。正式已据真实最坏成本2226.652秒授权2400+30，不增加epoch或端点。

原生CPU只复核旧已编码48篇IDs、候选/owner与微型梯度，不载模型；GPU工程六次实际FB（先累积4→真实非零Adam1→驻留Adam后累积2）覆盖7494token和616pair两极，全state重开后丢弃。最长输入7494/17pairs，最多pairs616/3068tokens；不截样本、不重做GP探针。

证据外部 R/research/long-context-relations-v1；只小源码/计划/结果指针进Git。原BioClinical GP完整终审957a2f0d与49件小包c54cdecf由root登记，不在本线重复审/复制大state。

工程实际：16:53:37.575758→16:53:59.103917 UTC，6FB/非零Adam1、140参数全变化/梯度finite、563state张量重开回装，峰3.965GB；16:53:59.254489 GPU0=4MiB/noapps。probe db5d273c仅保存丢弃。正式与probe model/common逐字相同，全部data沿原冻结；源准备不占卡，等seed短Sap交接。

正式首次纯I/O source.zip上传16:56明确LUNA_OFFLINE，无deployment ACK、无formal start intent/GPU。原probe终态有效，暂停.3调用；恢复先核原源文件，不重复模型。root静核正式与单质量源码无阻断，并指出prediction文件需先fsync再freeze：已以排他完整写/flush/fsync/父目录同步修正，配方不变、未发生质量调用。

传输恢复：16:59:29唯一纯读在线且原8247B包不存在；同包write_file再次UNKNOWN，后一次纯读也UNKNOWN。sequence17:00:43明确.3离线，统一退避至17:03:48由本owner单次恢复。root允许若原包未落、run_command已恢复时，用独立source-transfer-v2.zip写同原SHA字节并fsync/reopen，再原成员解包；旧UNKNOWN记录不覆盖、不重打包，0GPU意图。该fallback目前仅源码准备。

17:04:25原路径纯读恢复8247B/SHA4f6a9f94，证实UNKNOWN上传实际已落；直接原包extract五成员和原probe data SHA23248ecf成功，未重传、fallback副本未使用。仍0formal start，等seed Sap。后继launch-ticket改用已恢复run_command排他写/fsync/读回，同300秒H69/原driver/实际启动预算不变；若该写未知仍先恢复原ticket，不重发start。

root资源修订：T5于17:05:05.631真实释放.100GPU0，seed必要Sap改该同缓存卡；本线不再等待seed的.3释放，可按原9a85正式计划/fresh门立即启动。只改资源队列，模型/数据/2400+30/唯一质量门不变；原等待计划及新resource-amendment分别保留。

正式实际：proc_e18cb722/PID1227634、child1227642，17:06:48.691644 UTC start，原TERM17:46:48.691644/+30。首正常观察2/30轮、16Adam，每轮4495对，stderr空。没有加载probe或GP状态；完整epoch30与2874held前不质量、不早停。恢复入口formal_ops.py observe，唯一原intent已存在禁止再start。

正式完整终态：17:15:06.020657 UTC rc0/497.329s，17:15:06.210255 GPU0=4MiB/noapps并交LongT5。final state63db2cca仅远端、563state ACK13eafe5b；2874完整分数290106B/e889877b已本地回收。9必要成员ZIP79375B/665c2265已实际全SHA/fsync/reopen。所有30轮/960次文献/每轮4495对完整，不是工程故障；原17:46:48硬截止不变且未触。

最终持久：terminal-backup-v1/complete.json a8ebe3bc；source-quality.zip 79027B/38e4b857，38内容+manifest=39成员全实读回核SHA/fsync/reopen；原9formal父包复用，不复制大state。稳定完整entry a6abb440已交root/validation，源码/结果按本分支推送，当前无模型作业或额外研究。
