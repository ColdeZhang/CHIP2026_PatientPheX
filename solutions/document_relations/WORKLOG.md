# 文档级关系工作台账

- **当前：30轮已唯一正式启动**。`proc_3973bb9b-954c-4b6a-afff-3e71eb2f1baa` / PID1199462，.3 GPU0，10:19:15.037771 UTC启动，TERM10:34:15.037758、KILL+30；fresh4MiB/noapps、盘176.7GB，首epoch1/30、8Adam正常。恢复见[evidence/long-start.json](evidence/long-start.json)；旧long-resume是启动前交棒记录，不可据它重新start。
- 起点：main `0c0672d`，分支 `explore/document-relations`，独立worktree同名；用户目标原固定dev≥.8。
- 当前实验 `E-20261024-DOCRE-001`：复现ATLOP式多提及池化与局部attention交集，扩展为覆盖全文的重叠窗口，固定公共三模型提供HPO候选；不复用任务训练候选，避免新的留出泄漏。
- 先完成F2+F3 train32→F1初版。正文全部编码；物理提及只归属一个完整窗口，跨窗重复不重复计权。患者/概念多提及logsumexp，ATLOP式pair context＋分块双线性头。首版不加额外实体图Transformer。
- 二元关联使用单logit BCE/零logit阈值；不把二元退化的ATLoss当新优势。正例和负例都按实际关系标签生成，无人工evidence标签。
- **当前：两表示完整F1终点已完成，均未过强N门**。marker union **.663128103**（比unmarked +.017472459，但仍低N .676546355），replacement **.558854734**；不扩F0/full64/dev。两run共新4v2/旧0。GPU已10:06:58实际释放；完整[比较](evidence/comparison.json)、[marker运行恢复](evidence/marker-runtime.json)。
- **首版已完整负结案**。8ep/64Adam及全F1完成、GPU已09:54:09释放；两固定终点共冻后新2v2/旧0：union **.645655645**、candidate-replacement **.534242627**，均低强N **.676546355**。不扩F0/full64，不改当前best。完整[结果](evidence/quality-v1.json)、[诊断](evidence/diagnosis-v1.json)、[运行/恢复](evidence/formal-runtime.json)。
- 恢复：外部 `/workspace/patientphex-runs/round2-20261023/stage8-document-relations/`；源码入口本目录`prepare.py`。

旧local pair CE、identity bridge、doc-HPO max-MIL都有真实负终点；本控制区别是完整文档的一次共享编码及实体全提及表示、pair共有attention，既不是局部独立片段分类，也不是取出现max分数。详见NOTES及实验协议。

准备已完成：train32候选4495对、727/818训练关系可由公共候选覆盖；held2874对。仅正文/原患者及冻结public候选进入held，未读F1 gold。CPU编码已独立部署8件/412068B并唯一启动，等待真实token/window及pair几何统计；尚无GPU。源码处于初版工程状态，CPU与GPU终点尚待。

实际CPU编码1.250秒：训练113210 tokens/300窗、最长7023/18窗；held58522/156窗、最长7729/20窗，两个gzip真实回收逐SHA通过。原run_command的1秒wrapper超时丢失退出码，原子进程完成且已不存在；未重编码，后继改用start_process。训练几何720/727正对有attention支持交叠，7对无交集仍由两端entity表示分类；不加额外全局fallback。held输入2172/2874交叠，无gold。首版不宣称完整跨窗注意力传播。相同span但不同HPO的10组中仅2个训练患者组标签冲突，保留为当前无概念名输入的限制，不临时加入额外语义头。

目标Torch2.7.0 CPU合同3.376秒rc0：checkpoint/direct logits和梯度exact，窗口排列不变、重复owner拒绝、空query窗和disjoint pair明确通过。8ep正式方案之前，root授权唯一120+30工程（最长真实Adam→短doc累积→最长已有Adam+grad），计划2542fd9d、源9638ef0，span已实际释放.3后唯一start；正式预算待实测。

实际probe已09:47:27成功：wrapper10.095秒/native4.329秒，3前后向/1Adam，203梯度全有限、三类权重实际更新；已有Adam+累积grad的最长18窗0.942秒，峰allocated2.990GB。正式fresh8epochs/64Adam及全F1 logits获root600+30，125.6秒训练外推仅参考，formal已唯一启动；首次观察epoch1/8、8Adam、4495全部训练候选，loss .628909、19.195秒，实际fresh4MiB/187.4GB空间门通过。最近远端已核57567f812e691653562ae23e3a0f9c1c4bbb18cf。

正式实际162.877秒rc0，完整状态1.319GB在.3持久，原始2874 logits已回收SHA `ccce0f52417100c636e6d412416c17ba03fa870b3f3bfe448b2f401ce6e4956a`。主要失败为新增214条只有36TP/178FP，多患者候选AP .413；总候选覆盖469/520，不能归因仅候选缺漏。质量入口原假设pred含patient字段在读gold前失败，机械改按既有association并与encoded患者核同39ID，原失败源/空目录保留，未重训/推断。源d3a761f已推并远端核准；下一步根据真实诊断选择机制，禁止F1扫cut伪改善。

后继marker-v2已获root唯一600+30，排Sap→GP feature control之后。只改有原ATLOP依据的显式`*`边界和起始marker表示；原模型/全候选/loss/8ep64Adam/cut/两终点不变，fresh公开base。实际全48篇CPU剥marker重构原tokens精确、完整marked mention单owner/共享与相邻边界合同均通过；train319窗/held164窗（原300/156），发布词表`*`=14，无新参数。当前等待队列，未启动GPU；计划见evidence/marker-formal-plan.json。

marker-v2部署已完成：4件source/encoded在.3 CUDA隐藏CPU写入/fsync/重开逐SHA通过，receipt09379831…；source/plan检查点ea482a4已push并远端核准，正式尚等span释放。另导出融合候选逐提及边界/原文/三public来源，1069概念2180提及/360052B，stable key与全部2874logits匹配，见evidence/fusion-support.json。

marker改变输入长度，维持510窗上限的必要重映射已披露：训练正对attention支持交叠720→713/727、无交集7→14；held无标签交叠2172→2053/2874。所有原提及仍参与，无新增fallback或mask。不是完全同token窗的数值等价对照。

marker正式178.409秒rc0，8epoch/64Adam/16held全部完成，state1.319GB SHA14858de2…持久；scores2874/SHA8c94001c…实际回收。候选AP .44811→.54738，多患者AP .41319→.50542；新167条43TP/124FP，仍不足以改善强N。端点前两完整预测共冻，未作cut/epoch选择。源47ae85d已push并远端核准，当前整理两run持久交接；固定更充分训练仅为待协调后继，不冒称已执行。

两run小型终点已95件/321669B持久备份并从.3实际读回逐SHA，见evidence/two-control-backup.json；完整state不在小包、原远端locator不变。待协调后继固定30epochs/240Adam，唯一变化是训练时长及相同warmup比例/线性decay按新总steps缩放，不按F1挑epoch；源仅参数化3行、默认仍8，尚未部署/启动，原已运行源保留于95件包及原远端source。

固定30epoch/240Adam后继现已获root批准唯一900+30，排GP F0真实释放后，fresh同marker/base，仍两完整共冻端点/各1v2旧0；若两臂仍不升强N则关闭本组。低成本CPU实际核过：只3行epoch参数/状态断言改变，其余与已完成marker源码逐字相同；960doc/134850pair passes、24warmup，held唯一loop在全部epochs之后。新独立source-long-v3保留原remote source不动。批准计划8ad76336…与CPU证据见evidence/long-formal-plan.json、long-cpu-contract.json；当前未GPU启动。

比较标签补全：固定N .676546355是normal-fulltext强对照，并非历史F1整条流水线最高。已只读核历史public-new LGB支持组合 .686338981、mic .738647830/mac .591769159，16文献/39患者/同1189实体逐项相同，metrics SHA59242df4…。其支持gate train48排F1、含F0，资源条件不同；不改当前门、不新增端点。只胜N只能称待融合候选，不能称全流程新best，见evidence/historical-full-pipeline-reference.json。

恢复后已唯一启动30轮，原启动意图在transport持久。下一步低频看原进程/进度，不依赖本地跨轮watcher；完整16F1完成再freeze两终点、各1v2。完成实际release交grounded Sap（owner sequence_recover_0518）。同时root临时分配同槽CPU只读GP F0增量审，独立写validation/span-oof-F0-final-review，不影响本训练。
