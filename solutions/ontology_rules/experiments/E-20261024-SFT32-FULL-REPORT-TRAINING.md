# E-20261024-SFT32-FULL-REPORT-TRAINING：全文训练可行性

## 目标与边界

验证在原Qwen3.5 SFT32训练输入后追加完整允许文章，能否在可接受成本下优化相同关系目标。推理时追加全文的两折单臂/旧并集有限正结果支持研究这一变量；对更强三方的四源控制已因F0下降关闭，不能把它当本训练收益。正式训练和质量试验尚未获授，当前只做CPU来源/目标合同及唯一GPU工程压力探针。

保持原124条、F2+F3共32文献、880CID/87患者/2506对和全部7170监督token；原候选模型来源和F0/F1排除链保持。只在原user证据之后追加完整供给文章，原answer、system、CID/PID、局部证据、原型和其他字段均保持。1122passages可精确重建原文，255个given mention中254个插入marker、1个重叠mention按既有政策跳过；该文献所有患者仍有marker。F0/F1验证gold未读入这次准备。

## 已完成CPU事实

- 原训练402594 tokens、max5523；全文1009181、max13549（唯一idx98）、中位7999.5。原生Qwen tokenizer/Auto/template及完整往返解码124行一致，无截断，工程cap13824。
- 实测native22.406947秒；全部7170监督ID与原任务一致。总token比2.506697，平方长度和比6.047086；两轮2881.348秒只按旧1149.460秒乘token比的代理，不是测量或上限。
- 实际A800 Torch2.7.1+cu126/TF5.20.2 CPU selected CE合同2.329901秒通过：288基础组合、同形状head梯度/累积1、2、4、8（含尾4）及忽略位扰动均符合声明容差。logit梯度最大绝对差0，loss最大差7.629e−6，不能称loss逐位相等，也不能当全模型GPU等价/显存证明。

[CPU摘要](../qwen35_sft32_full_report_training_readiness/ACTUAL_CPU.md)；外部根`R/research/qwen35-sft32-full-report-training-readiness`。exact原124训练的全文实际终点未在此次盘点范围发现；Med模型全文训练是另一配置，不混作本方法已测。

## 唯一GPU压力探针

父线程授权420+TERM30；source165e9bfc、plan32f089df、entryadf673dd。先actual15件base SHA和GPU2 UUID/空卡/空间ACK，唯一proc_6439c798-0c8d-44e7-a441-b2d11f2eab8c/PID48140于06:11:16.510Z开始。恢复使用原`operations.py inspect`；不能因客户端状态不明重复start。

fresh base/r16/alpha32/dropout.05，原12目标模块，adapter43,278,336，总8,997,081,600参数。BF16、原生SDPA/GDN、不加载旧adapter、冻结基座与head；selected CE仅在FP32前选中有效监督位置，完整BF16 LM head保持；gradient checkpoint use_reentrant=False。

按H51压力政策，原seed20261023 epoch0首8条各mean/8并做非零LR1e−4 AdamW更新建立真实状态，随后3条累积再处理最长13549。最长在原shuffle位置56/第8组首条，故选择其前3条119/55/69形成刻意跨group压力次序；不宣称重放formal顺序/warmup。12个样本105332tokens逐input IDs/labels SHA与既有native audit精确相同。该probe状态退出即弃，无checkpoint、推理或质量。

记录每次前/反向和长样本前的Adam/grad库存、finite、三家族LoRA B实际改变、基座无grad、allocated与reserved峰值、时长与释放；依据实测再估计全两轮和逐epoch持久导出预算，不能把probe通过当收益或正式训练许可。

## 工程准备修正

首次本地prepare缺原helper同级模块路径，在任何编码产物/GPU前退出；补目录后12项编码契约通过。preflight模型manifest实际为list/name/size，部署前静态检查时已修正，未执行初版单独保留。所有这类准备失败不算模型或技术路线失败。

## 当前结果

唯一探针已PASS/rc0：wrapper288.173982秒、训练窗口260.794100秒；12次完整前/反向和1次真实AdamW更新。最长13549前/反向42.930650秒；最长前已有496个Adam状态和496个有限梯度，三族LoRA B已实际改变，基座始终无梯度。峰allocated32,689,242,624B、reserved59,114,520,576B；06:16:04.859 GPU2 used1/free81156、无计算进程。48件终点已回收逐SHA，zip23537B/8ad38b1e，source/CPU增量独审dabb18f6通过，实际统一工程审待完成。无checkpoint或质量，probe状态已丢弃。

12微批实测计算260.519191秒/105332tokens，按全部124长度投影两轮：token加权4992秒、平方长度加权4503秒、均行5384秒、按最慢实测seconds/token6395秒。原2881秒代理低估了这次实际单步成本；这些新值也都是估计，不是正式训练事实或保证。

同r16既有checkpoint五件合计519,867,104B（其中adapter173,181,072B/state346,666,642B）；A100此前checkpoint耗时含在epoch总时长，不能当A800→.100网络测速。项目盘当前约127MB，正式训练如获授权需新私有SHM保存全状态，每epoch持久至.100并全SHA/fsync后继续。已向root建议总绝对预算8400+TERM30，包含两个每epoch≤600秒持久导出预留；预算尚待决定，正式训练未执行。四源控制结论另见[独立结案](../four_source_fulltext_control/RESULT.md)，不在本实验内更改已冻结端点。
