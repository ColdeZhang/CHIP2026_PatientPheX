# 当前最佳结果的资源清单

2026-10-23，只读现有源码、产物和 manifest；没有训练、模型加载、推断、评分或查询远端 GPU。对象是原 dev **.732257765**，未达到 .8，不是已验证的新输入部署包。所有本地文件实际 SHA、大小和精确路径见[资源索引](../evidence/round2/best-pipeline-resource-inventory.json)。远端路径来自既有成功运行记录，本轮没有重新确认远端文件存活。

| 环节 | 必须保留的资源与现状 |
|---|---|
| 固定输入约束 | 原 train64 `cb45e7df…`、HPO 2026-06-23 `a5092cbd…`；新输入保持原全文坐标和指定患者结构。训练派生别名只来自 train64。 |
| Task1 本体/CRF基础候选 | 原本体源码/规则和完整 train64 CRF `pipeline.pkl`（42,778,483B，`8753c5f2…`）及 `crf.model`（453,232B，`ee0f34ba…`）。现成功运行环境 Python3.12.14/sklearn1.8.0/crfsuite0.9.12。 |
| Task1 Sap候选 | 固定 SapBERT `090663c3…` 与实际 FP16 权重 `dc5e5071…`、tokenizer；train64 类型判别器 `reconstructed-detector.npz`（6,926B，`adb8ff8c…`）、有序42,575别名bank及编码缓存。全部来源在 `sap64_reconstruction_manifest`。 |
| 两个公开 PhenoTagger 源 | 固定 `lingbionlp/PhenoTagger_v1.2@02ce3f52…`；PubMed权重 `d58af55a…` 与独立 Bioformer `867819ac…`。必须保留 native 分词/词形还原/原词典、17,266列旧head、旧ID到指定HPO映射、缩写解码及真实来源sidecar；两个score分别进特征，不能取跨模型max替代。协议维持原top3/五位舍入/严格>.95。 |
| Task1 最终选择 | 已保存实际 `selector.pkl`（295,637B，`db2fc288…`），包含97列DictVectorizer与HGB；同特征定义、原全池greedy、全局cut .3。独审已经重载16篇全部13,144概率exact，推断不需要重fit四折。 |
| Task2 候选输入 | 当前最佳沿用旧CRF+Sap64加GLiNER48的固定候选策略；GLiNER模型 `25bb0b35…` 位于 `.100/sequence-round2/gliner-outer0/formal-fixed10ep/epoch-010`，匹配medium tokenizer和旧解码/Sap筛选。**该GLiNER不在当前Task1 HGB池中**。不能无新验证直接用新HGB实体替换关系候选池。 |
| Task2 关系模型 | `.100/case-retrieval/qwen35-preparation/models/Qwen3.5-9B` 与 `full64-dev-transfer/adapter-epoch002`；adapter权重173,181,072B，`1a703b24…`。15个基座文件SHA、原runner源码、prompt与解码配置见 `relation_actual_manifest`。原成功环境torch2.7.1/transformers5.20.2/peft0.21.2，BF16、SDPA、thinking=false、batch1、14000输入/1600输出tokens；逐8候选局部证据提示和严格合并入口由检索路线维护。 |
| 最终装配 | Task1选定实体与独立Task2关联合并；Task2不要求是Task1实体子集。现dev完整预测 `6401a24e…`，关联逐对象等于 `aa107790…`。不得以新实体集合重新过滤既有关系。 |

## 部署前的实际缺口

现 `select_public_bioformer_dev.py` 是带原dev固定SHA断言的实验拟合入口，不是通用推断器。后续应另写只加载实际selector/vectorizer的薄入口，将旧全池候选和来源特征完全复现；用已冻dev候选做不读gold的字节回放，再处理新输入。不要为了部署重训HGB或改阈值。

CRF pickle 里的估计器引用模型路径，迁目录必须同时保留/显式重绑 `crf.model` 并验证；仅有Sap判别器NPZ也不足，仍需相同bank顺序、词典和编码器。公开模型还依赖native文本恢复与来源字段。已有各模块成功记录，但本轮没有运行单一端到端新输入任务，因此不宣称当前目录可以直接一条命令跑test。

## 已交接的关系支持集

F0/F1 cut .3 完整OOF预测分别为 `public-bioformer-cut-calibration/cut-3/fold0.jsonl`（`298cc874…`）和 `fold1.jsonl`（`8ea3a1da…`），精确路径/完整SHA/48训练与16输入的隔离核对在资源索引 `support_handoff`。两个HGB均排除自身目标16，三套下层train32 OOF来源已审；但全局cut .3使用全train64实体标签选择，含F0/F1，因此新增关系支持过滤应标作开发诊断。只提取文档×原子HPO实体集合，不能采用文件中的旧规则关联覆盖SFT输出。

后续有界整合：候选池→既有HGB部分已有[薄推断入口](load-existing-selector.md)，无需重fit且不绑定dev文献；全16概率/预测字节回放通过。上游原文候选生成、CRF跨目录模型路径和完整Task2新输入执行仍保留原限制。
