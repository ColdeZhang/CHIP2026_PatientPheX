# 条件化融合台账

2026-10-24 UTC；分支 `research/conditional-stacking`，起点 main `da0f7992`。

- **当前**：整条固定配方已关闭。F0微正后，同theta原dev迁移与旧B4全预测相同，继承 `.768400942542392`，低于现best `.7692356699419443`，未达.8；原dev0新v2/0gold/0新fit/GPU。[统一结案入口](evidence/originaldev-entry.json)。
- **关键结论**：旧 G/S∩/Q raw union561 中有31条被交集隐去的单采样支持，跨11文献、三个旧bit模式内都有变化。成熟条件化 simplex 权重可表示此差异，建议固定6参数、向等权2of3收缩的唯一低成本 CPU 控制；覆盖不等于涨分证据。
- **隔离**：实际 S 适配器4f4815→32文献checkpoint→F2/F3双向候选祖先已绑定；F1候选32，F0候选48排F0。禁止 B1/含F0的train48 gate或其概率进入 F1 特征。
- **交接入口**：[短报告](REPORT.md)、[来源/祖先](evidence/ancestry-and-source.json)、[无标签计数](evidence/signal-coverage.json)、[公开成熟实现](evidence/public-sources.json)、[无gold合同](evidence/contract.json)。只有一个配方；目标/梯度纯函数已实现，无拟合调用或新预测。
- **下一步**：原dev50成员/81351B包cb8d837f已服务器落盘/完整实际回读；[备份](evidence/originaldev-backup.json)，receipt aca57b9e。已交回root，没有排队实验/fit/GPU/质量。保留现best，不凭F0微正晋升。

## 本次边界

复用已核 G/Q 和候选的准确祖先元数据，不重审大包/模型，不读训练正文或任一 gold。初始供给统计仅F1；15:13 root指令后补读两折完整纯输入/预测作零参数集合合同。96合成模式通过，人工标签梯度max2.70e−9；完整F1/F0与旧B4各0增0删，不称byteexact。读过既有质量摘要用于查重，公开论文与代码下载到仓外。原始金标/混合标签表0、fit0、质量0、GPU0。

## 协议修订

15:12原提议以三源作门，尚无执行；15:13 root要求保C固定补充并对更强B4。已在报告保留原提议和修订理由；唯一当前协议以plan.json为准。未据F0标签、得分试验或不同拟合结果选择配方。

## CPU实施

15:18：运行器目标接口已无gold通过F1全561/F0全431（分歧31/44）；零参数仍等于原多数。原plan SHA7d53ce36保留，新execution绑定实际源/NumPy2.3.5/SciPy1.17.0。fit原生timeout60+5；后续装配/质量各120+5作工程上界；每阶段独立exclusive产物，失败不自动重试。精确命令在execution；结果目录 `/workspace/patientphex-runs/round2-20261023/research/conditional-stacking-v1/`。全部GPU调用0。

15:18:28.810645→15:18:29.812980整个CPU控制完整退出。fit实际.012403s、9迭代/11函数梯度、目标8.647475→8.424862，按原relative-reduction终止门success；不是gtol收敛。六系数/真实梯度均保留。15:18:29.289611参数先冻结，.627035完整F0冻结，随后唯一v2。新增1FP/删1FP、TP不变、macro+.000557244；两个变化均9患者且分歧0，不能归因新分歧信号。source14fca616启动前已push并核远端，原prepared7d53ce36未改。

15:22:07 F0持久包完成：首次仅CPU元数据校验命令因字符串占位替换SyntaxError，尚未文件操作；保留原错误、原上传，只读恢复后实际全包回读58成员SHA一致，receipt bc6fa22c。15:23原dev计划另行绑定实际full64 G/S1/S2/Q/C与原B901；原prepared/execution和F0所有结果均不覆盖，暂无额外fit/质量。

15:23:23原dev完整闭合：实际G250/S289/Q265/C300，379 rawunion、30采样分歧；S纯集与原protected源减去B901语义完全核对，零theta重构B4。已学习theta全输出仍425关系=旧B4，直接继承旧bff390f3原文件、旧分.768400942542，0新gold/0v2。所有416 B901关系、1061实体、36患者及对象/原关联前缀保护。原dev另有1篇8患者文献超F1最大7，范围外推限制继续保留。

最终远端检查点8ae43b9e已核origin同SHA。原dev包原件一次上传，首read_file UNKNOWN原错误保留；随后仅对原文件CPU分块完整回读和fsync/reopen，50成员全部SHA一致。无重生成/重传或质量重跑。
