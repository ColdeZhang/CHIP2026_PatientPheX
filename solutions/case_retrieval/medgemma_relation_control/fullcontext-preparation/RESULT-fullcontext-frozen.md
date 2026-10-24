# 闭集全文 MedGemma：固定配方双端点负结案

本次完整实现并评测了**原 CID→PID 任务在训练和推断中同时追加允许全文**；不是零样本全文推断，也不是此前开放 HPO/quote 目标。fresh MedGemma1.5-4B 公共基座，严格原 train32 的124条×2epoch，原答案/候选/CID/PID与7150监督tokens保持；唯一输入变量为末尾追加原文，训练容量明确14000。32 optimizer calls/248曝光、238语言投影r16、原native CE/AdamW/RNG恢复、BF16/SDPA等固定不变。

本F1是反复观察的内部开发折。强对照是旧 SFT32 .660942003；不是原dev、独立泛化或官方成绩。

| 固定结果 | Score | 关系micro | 关系macro | TP / FP / FN |
|---|---:|---:|---:|---:|
| 旧SFT32（缓存） | .660942003 | .686600221 | .542228853 | 310 / 73 / 210 |
| 主：全文Med | .592418621 | .528875380 | .425860167 | 261 / 206 / 259 |
| 唯一次要：旧SFT32∪全文Med | .627966889 | .616788321 | .480140298 | 338 / 238 / 182 |

主/次要Score分别低强对照.068523382/.032975114，关系micro与macro均下降，均不达事前保留条件。旧383关系、全文Med467、交274、并576；并集新增193关系中28TP/165FP。格式全部有效，80行/587CID无未决、无token上限触发，16文献/39患者/1189实体完整，所有非关联原始行bytes保持。此次主要失败是关系假阳性过多，不能归为截断/格式/任务未实现。

已有短Med缓存Score .603701549、并集.644663430均不重评；全文分别再低.011282928/.016696542。与短Med的汇总净变化是主模型TP+8、FP+59；这是既有两个固定配方的比较，不是新增选择/组合，不足以否定其他基座、schema或全文SFT策略。

两完整fullpred于04:22:07.858832Z共同freeze `28f5d4be`，首次gold marker在.861029Z之后，各唯一调用原v2 all/text，旧指标复用。summary `cf69328e`；主pred `f0f83c8c`、union `3063f98b`；raw `c4fdf9e8`。解析仍用原ddba候选框架，实际全80输入2a95逐行绑定原native CPU审查。

工程实际：两epoch loop510.186890+507.378265=1017.565156秒；第二invocation530.063407仅指第二段。两load/训练/两完整持久导出及回读共1264.673774秒<2700。最终训练进程allocated峰52.003317GB、reserved75.243717GB；第一epoch第15步日志allocated峰52.004076GB，不能将第二进程峰冒作全campaign精确最大值。token线性800.510秒曾为估算，实际两loop高27.1%，不改写估算原件。

两个完整checkpoint各27件358631877B已持久至.100 `research/medgemma-relation-control/fullcontext-formal2ep/checkpoints/epoch-001,002`，fetch分别47.764058/39.155741秒，均独立重开全文件SHA。epoch1 checkpoint f9ac6ea7/adapter19fee977/Adam635ed4e4，epoch2 c4044dac/adapter47798260/Adam0bfa4203；fingerprint完全相同，仅训练状态推进16→32。源含9代码、完整Adam/scheduler/Python/NumPy/Torch/CUDA RNG、实际CPU依据。

固定80 native282.138秒、含控制回收307.880645秒<900；模型共4,329,881,968参数。GPU2 f516于04:21:32.942369Z实际空81156MiB，release `9443432d`，无后继GPU。训练原截止04:40:33.082110Z+TERM30，推断原截止04:31:39.653610Z+TERM30均未延长；两个current state已保留旧字并关闭。

正式入口独审41e51269；最长probe审2cc62bee与CPU审4a6544d2复用。H48新17.716秒CPU只证明固定原80文本Auto/原JSON编码解码一致，序列化与未用音频特殊符号差异独立保留。H49窄核确认先raw write/flush/fsync、后独立解析，无热改或额外模型检查。

**决定：关闭此固定全文Med配方；不迁F0/原dev/官方，不补阈值、组合或额外epoch。** 完整引用见terminal-manifest.json。raw/fullpred后续持久备份及最终独审回执在工作索引维护，不重写本结果或重复质量计算。
