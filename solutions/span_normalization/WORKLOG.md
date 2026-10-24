# Span normalization 工作台账

- 目标：原dev公开v2 Score≥.8。只对强系统的实际收益作结论；本路线原dev未打开、官方未访问。
- 当前：E-20261024-SPAN-001已完整训练/推断/规范化/双端点评价，固定追加负结果，不采用。第二控制E-20261024-SPAN-002已冻结为原强候选池新增GP边界分数/valid，3新nested/final模型及4全量目标评分已于10:03:40.602957Z完成（277.219秒），GPU0归4MiB/noapps交DocRE，CPU唯一fit/v2完成：F1 .728697536469→.730812709670（+.002115173201）。完整审并行，下一步同超参F0迁移准备。
- 方法：BiomedBERT+GlobalPointer整体start/end、RoPE64、多标签CE；8epoch/3e-5/batch16/384stride64，BF16encoder/FP32head+loss；原Sap与train32 exact aliases规范化。固定原始池logit≥−2、预测>0，不扫cut。
- 数据：32(F2+F3)→F1 16篇；原48→F0草案在任何执行前改为主协调统一端点。强F1是train48，故本比较是部署增益而非纯架构因果对照。
- 结果：强.728697536469→追加.701293121901（−.027404414568），GP-only .663575801621；关系底座完全相同。共2次新v2/0旧重评，完整预测先共冻；两格式均0错误/0警告。
- 供给：全池2133条/2134拆ID单元，强池外447中仅12精确HPO TP+3 NIL通配；最终追加360条/361单元，51精确TP+6 NIL通配、304 FP（错HPO27/边界假阳277）。不能把简单追加失败等同否定学习融合，但独立边界供给弱，不能盲扩同坏池。
- 资源：.3 GPU0原入口6.766秒0forward失败，pooler机械修复后保持原09:49:09.705568+30截止。恢复proc_f83b21dd/PID1196954于09:45:05启动，95.520秒完成，09:46:40.782904释放并通知root/docRE；未使用新的600预算。
- 训练完整：1148train窗，2263/2266可表示跨度、3token端点不可表示、10嵌套；held588窗/16篇全覆盖，576优化步。模型108990080参数；峰值allocated4.119GB/reserved4.526GB。CPU实际编码/数学合同通过。
- 证据入口：[结果](evidence/result.json)、[运行指针](evidence/runtime-pointer.json)、[实验](experiments/E-20261024-SPAN-001.md)、[方法](NOTES.md)。全部2133候选/分数及28件实际远端回收证据存仓库外R/stage8-span-normalization；Git只存源和紧凑索引。

- 第二控制实际CPU4.682秒PASS完整4组token/mapping，GPU源baf4d381，21件部署SHA通过；HTTP413按原ZIP分块恢复。首次inner0评分628窗9.343秒，14730/14763有效（33未对齐missing），13439个低于−2的分数完整留存。GPU0实际fresh4MiB/noapps，详见evidence/oof-runtime-pointer.json；该GPU作业已完成，禁止重启。

- 第二控制F1增益来自mention TP+3/FP−18，实体1313→1298；两Task1分项提高、关系exact。模型/预测小包已实备份.3。完整证据evidence/oof-result.json；暂不原dev/官方。F0已有同参数LGB102列（勿写101），计划102→104，46733训练/3186阳性，1新fit/1新v2。

- E003 F0同超参迁移已冻：旧102→104，46733/3186/15028；严格排F0，复用GP23，其余3新模型；同900+30。独立generic入口保留F1源码原SHA。F0 GPU已10:13:23.020215Z唯一启动proc_b616d04f/PID1199035，当前正常推进，运行/期限见evidence/oof-f0-runtime-pointer.json；F0唯一质量.739506259147，对强.740572140440小负−.001065881294；无原dev/官网。

- E003已GPU272.023s/CPU16.639s完整结束，F0小负，26加46删、mention TP−6/FP−14/doc TP−5；metadata见evidence/oof-f0-result.json，.3模型/预测备份31191e6a。docRE增量审并行。
- 后继E004已授权四折研发补全（非盲测）：保留已知F0负/F1正原预测，补F2/F3同配方，旧103→105；只缺GP01和两个GP48，8全目标pass，预估338秒≤900+30，排DocRE30真实释放后；源/计划/CPU合同待准备。新F2/F3一起freeze后各1v2+唯一pooled，保留门严格pooled升且3/4折非降。

## E004 固定四折补全（2026-10-24）

已知F1小正/F0小负后，按主线程授权补齐F2/F3，真实103→105旧接口CPU检查中；3新GP+8全边界pass，预算900+30，排DocRE30实际释放。两新预测共冻后各1次v2，再唯一pooled；事前门pooled严格升且≥3/4折非降，否则关闭。见[实验记录](experiments/E-20261024-SPAN-004-fourfold.md)。F0终审a3f855a0完整PASS负终点已收；未原dev迁移。
