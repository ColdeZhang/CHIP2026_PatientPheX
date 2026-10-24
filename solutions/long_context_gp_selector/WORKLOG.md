# BioClinical GP selector 工作台账

分支 research/long-context-gp-selector；起点 main5333e7f8。仅修改本目录。当前 root 已授权唯一 F1 控制，尚未启动GPU模型；原生CPU接口7.885秒已PASS，等待schema真实GPU1释放。

固定忠实 strong48：F0/F2/F3 严格 OOF 新 GP 两列，最终48训练供F1；复用GP23→F0，新拟合GP03、GP02、GP023。候选、101旧特征、160trees/.3/nested均不变，只增valid/raw-logit，唯一完整F1 v2，严格胜 .728697536469 才保留；不扫cut、不自动F0。

运行预算：.100 GPU1 三fit/四完整评分共享900+TERM30；本地固定LGB/完整质量60秒。部署和CUDA隐藏CPU接口在正式钟前。当前两机库存已核，目标空卡、392GB空间；源 .3 两次离线/UNKNOWN的原请求与恢复保留，实际CPU模型导出及1.2GB私网复制已完成11.778秒、全部SHA/fsync/reopen；原HTTP PID1227574已实际不存在。目标复用boundary已安装的独立TF4.50/tokenizers.21.4，不改共享包；Torch2.7.0cu128→2.7.1cu126/hub.36.2→.29.3如实披露。GPU1现由schema原1085 F0推断占用（17:03:03真实start），本线等其actualrelease，不预占、不轮询他人作业。

外部入口：`research/long-context-gp-selector-readiness-v1/` 的 cpu-v1/既有readiness、activation/实际库存、formal-v1/完整数据、asset-copy-v1/复制意图与恢复证据。所有大数据/模型不入Git。

已知原新GP直接并集和固定veto负；旧BiomedGP两列F1正/F0负/pooled负，本控制仅检验新全文BioClinical分数，不声称该组合机制首次。当前旧GP23的1313分数缺15104候选，禁止充当完整GP48输出。

准备的第一次Python比较 tuple/list 容器不一致、第二次CPU映射低效已定位；均无模型/拟合/质量，保原源与停止记录在activation。重建仅缓存相同边界字典，所有目标全覆盖。

正式源计划53f716b3已完整部署；唯一CPU已rc0；actual64/native+tinyGP+model-only载入完整见observation-1792861803790532935.json，不重跑。后继只执行operations.py start一次，再按observe/pack/collect恢复原句柄；必须schema实际交卡且fresh门通过。源/计划/预写质量已完成，详见experiments/E-20261024-LCGP-001.md。

完整单一恢复入口 EXECUTION_HANDOFF.md 已写齐。root要求阶段final释放槽并在schema实际release后followup唤醒；没有GPU start intent、没有watcher或NN时钟。当前源码/运行计划保持53f716b3，后续metadata/传输恢复不改变NN合同。
