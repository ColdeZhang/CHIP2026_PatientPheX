# Span normalization 工作台账

- 目标：原dev公开v2 Score≥.8。只对强系统的实际收益作结论；本路线原dev未打开、官方未访问。
- 当前：E-20261024-SPAN-001已完整训练/推断/规范化/双端点评价，固定追加负结果，不采用。第二控制E-20261024-SPAN-002已冻结为原强候选池新增GP边界分数/valid，3新nested/final模型正式运行；proc017ed228/PID1198013，09:59:03.316368Z，TERM10:14:03.383655Z/KILL+30。
- 方法：BiomedBERT+GlobalPointer整体start/end、RoPE64、多标签CE；8epoch/3e-5/batch16/384stride64，BF16encoder/FP32head+loss；原Sap与train32 exact aliases规范化。固定原始池logit≥−2、预测>0，不扫cut。
- 数据：32(F2+F3)→F1 16篇；原48→F0草案在任何执行前改为主协调统一端点。强F1是train48，故本比较是部署增益而非纯架构因果对照。
- 结果：强.728697536469→追加.701293121901（−.027404414568），GP-only .663575801621；关系底座完全相同。共2次新v2/0旧重评，完整预测先共冻；两格式均0错误/0警告。
- 供给：全池2133条/2134拆ID单元，强池外447中仅12精确HPO TP+3 NIL通配；最终追加360条/361单元，51精确TP+6 NIL通配、304 FP（错HPO27/边界假阳277）。不能把简单追加失败等同否定学习融合，但独立边界供给弱，不能盲扩同坏池。
- 资源：.3 GPU0原入口6.766秒0forward失败，pooler机械修复后保持原09:49:09.705568+30截止。恢复proc_f83b21dd/PID1196954于09:45:05启动，95.520秒完成，09:46:40.782904释放并通知root/docRE；未使用新的600预算。
- 训练完整：1148train窗，2263/2266可表示跨度、3token端点不可表示、10嵌套；held588窗/16篇全覆盖，576优化步。模型108990080参数；峰值allocated4.119GB/reserved4.526GB。CPU实际编码/数学合同通过。
- 证据入口：[结果](evidence/result.json)、[运行指针](evidence/runtime-pointer.json)、[实验](experiments/E-20261024-SPAN-001.md)、[方法](NOTES.md)。全部2133候选/分数及28件实际远端回收证据存仓库外R/stage8-span-normalization；Git只存源和紧凑索引。

- 第二控制实际CPU4.682秒PASS完整4组token/mapping，GPU源baf4d381，21件部署SHA通过；HTTP413按原ZIP分块恢复。首次inner0评分628窗9.343秒，14730/14763有效（33未对齐missing），13439个低于−2的分数完整留存。GPU0实际fresh4MiB/noapps，详见evidence/oof-runtime-pointer.json；严禁重启活跃作业。
