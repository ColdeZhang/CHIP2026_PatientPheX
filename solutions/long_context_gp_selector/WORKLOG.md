# BioClinical GP selector 工作台账

分支 research/long-context-gp-selector；起点 main5333e7f8。仅修改本目录。当前 root 已授权唯一 F1 控制，尚未启动模型。

固定忠实 strong48：F0/F2/F3 严格 OOF 新 GP 两列，最终48训练供F1；复用GP23→F0，新拟合GP03、GP02、GP023。候选、101旧特征、160trees/.3/nested均不变，只增valid/raw-logit，唯一完整F1 v2，严格胜 .728697536469 才保留；不扫cut、不自动F0。

运行预算：.100 GPU1 三fit/四完整评分共享900+TERM30；本地固定LGB/完整质量60秒。部署和CUDA隐藏CPU接口在正式钟前。当前两机库存已核，目标空卡、392GB空间；源 .3 MCP 16:56:16离线，和source owner统一退避至16:59:21。源实际 TF4.50/tokenizers.21.4 位于 gptsovits 环境，私有复制到.100，不修改共享包。Torch2.7.0cu128→2.7.1cu126如实披露。GPU1短Sap先给schema，已16:57:09真实释放。

外部入口：`research/long-context-gp-selector-readiness-v1/` 的 cpu-v1/既有readiness、activation/实际库存、formal-v1/完整数据、asset-copy-v1/复制意图与恢复证据。所有大数据/模型不入Git。

已知原新GP直接并集和固定veto负；旧BiomedGP两列F1正/F0负/pooled负，本控制仅检验新全文BioClinical分数，不声称该组合机制首次。当前旧GP23的1313分数缺15104候选，禁止充当完整GP48输出。

准备的第一次Python比较 tuple/list 容器不一致、第二次CPU映射低效已定位；均无模型/拟合/质量，保原源与停止记录在activation。重建仅缓存相同边界字典，所有目标全覆盖。
