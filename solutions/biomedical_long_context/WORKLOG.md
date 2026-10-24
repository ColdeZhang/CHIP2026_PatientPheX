# BioClinical ModernBERT 长上下文 Task1

**已负结案。** 固定 F2+F3 train32→完整F1；8轮256Adam及17窗推断真实完成，最后state持久，全部1816候选按原Sap规则规范化。原保护并集唯一冻结后1v2/旧0：Score **.728697536469→.713455325552**，新增210=37TP173FP；不扩F0、不扫阈值、不选择中间epoch。关系与旧1313实体完全保护。详见 [实验](experiments/E-20261024-LCTX-001.md)、[完整入口](evidence/final-entry.json)。

分支 `research/biomedical-long-context`，起点main21dca5f3。只改本路线；原dev最佳.769235669942未达.8，本次未评原dev或官网。root另外预锁的GP<0删除控制复用同1313分数，其负结果单独归root记录，不计入本线一次质量。

- 原厂 `BioClinical-ModernBERT-base@c3648aa8`，149112704实际训练参数，MIT；ModelScope所见NeuML不同嵌入微调未混用。原厂完整权重598635000B/SHAea738868实际核过；首次HF网络0B失败保留，CDN69.771秒完整下载。
- 训练32全文122147 tokens/max7492，32窗；F1纯输入16篇39患者62699/max8371，完整17窗。旧GP同2266正跨度；新tokenizer2265可表示/1词内边界。初census2274比旧多8项已按旧phenotypic admission准确解释并纠正，原v1保留。
- 原生CPU全部48文献token/offset合同通过。工程23.737秒、4FB/2Adam、非零更新136参数张量、547state张量保存重开；工程statecc522仅保存、不用于正式。
- 正式16:37:07.419540→16:39:57.392305 UTC，169.973秒，256有效Adam，峰4.108GB；最后完整state15798e36共1.79GB仅远端持久，不下载/不入Git。17窗原始1816跨度和全部1313基线跨度score均同一次forward保存。
- 原Sap规范化16:41:10.834555→16:41:34.668257，23.834秒；16:41:34.790395实际GPU0=4MiB/noapps。1816全候选归一化，1012通过原GP gate；本线已无GPU任务。
- formal10成员127318B/1077bff9、norm8成员122782B/57670f8d均实际服务器回读、全SHA/fsync/reopen。稳定终态包只增源码/质量/指针，不重复模型或父包。

外部根 `R/research/biomedical-long-context-readiness-20261024`；R=`/workspace/patientphex-runs/round2-20261023`。后继仅root提出的全文患者关系缺口CPU研究，不改变本次冻结配方。
