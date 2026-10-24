# 来源与方法边界

只复用 root 已缓存的 BioSyn/Sap README 与 ACL 页面，没有重新下载。

- [BioSyn](https://github.com/dmis-lab/BioSyn)：MIT 代码；Sung 等《Biomedical Entity Representations with Synonym Marginalization》，ACL 2020，DOI `10.18653/v1/2020.acl-main.335`。缓存 `biosyn-paper.txt` 实际是 ACL 元数据/摘要 HTML，**不是全文已读**。成熟思想是同义词边际似然与训练过程更新候选；本配方去掉 sparse 分数，使用 Sap 初始化和集合正例，不称原论文复现。
- [SapBERT](https://github.com/cambridgeltl/sapbert)：README 显示 MIT 代码许可；所用已缓存 `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` 发布模型原记录为 Apache-2.0，revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`。区分代码与模型许可；不引入额外 UMLS 标注文件。
- 旧 `semantic_neural/run.py` 是 `.eval()` + `inference_mode` 编码，下游 LR 训练不改变编码器；旧 `run_linker_rerank.py` 在得到 Sap 候选后删除 Sap encoder，再训练独立 Biomed 分类器。旧 Top8 Qwen 预测候选序号也没有更新 Sap。
- 新信号是共享编码器两端都接受任务监督，训练后重新编码全部41,099公共名称/EXACT同义词。不是把 K 改名，也不是再训练旧 cross-encoder。
- 盲区：词面编码器没有文内上下文；同名多概念只可防止错误负例，不能学习上下文消歧。新距离会影响未出现在 train32 的公共概念，灾难性遗忘是实质风险，唯一完整 F1 终点裁决。

精确文件 SHA 与实际供给见 `evidence/readiness-entry.json` 的外部入口。不得从旧“未命中”或一个负配置推导整个方法家族不可能有效。
