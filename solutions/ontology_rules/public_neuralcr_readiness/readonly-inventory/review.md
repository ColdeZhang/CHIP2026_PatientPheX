# NCR / PhenoLinker 公开资源增量（只读阶段冻结）

**NCR 找到了真实公开 HPO 成品；PhenoLinker 尚未核实到独立的临床文本抽取版本。当前没有新的质量结果。**

| 资源 | 本次实际确认 | 决策边界 |
|---|---|---|
| NCR / NeuralCR | 官方 `data-team-uhn/NeuralCR`，release **1.1** 和当前源码均指向 `1af8851b5e4a1ebeec7ced6c2f50d2dd34d4db0f`。官方 release 提供 `HP.tar.gz` **583,483,718 B**、`pmc_model_new.bin` **1,345,465,351 B**；HEAD 200，明确 Range 请求206。 | 当前只取 HPO 包64 KiB前缀和 fastText 290 B元数据，没有完整权重。HPO包首成员为630,900,616 B的 `HP/ncr_weights.h5`；旧 sickkids 网址受到本环境代理403，不能解释为模型不存在。完整大小/SHA、全部head形状、runtime及模型许可证附件待后续获准就绪检查。 |
| 临床文本 PhenoLinker | EuropePMC、Crossref、OpenAlex、GitHub仓库名/README、HF模型API以及三个精确网页查询，当前只定位到已知 **Mellandd/PhenoLinker** 基因–表型图关系预测工作（DOI `10.1016/j.artmed.2025.103177`；预印本 `10.2139/ssrn.4756046`）。 | 未找到可核验的另一篇临床HPO抽取论文、官方代码或checkpoint；不能把旧图模型结论泛化成所有同名资源均不匹配，也不能声称其他版本不存在。本次不为未确认身份假定权重、offset能力、许可或参数。Google返回挑战页，未作有效缺失证据。 |

NCR机制是 **fastText 子词词向量 → 单词宽度1的CNN → max-pooling → Dense → 含HPO祖先聚合的概念embedding头**，10个独立核心平均softmax；没有全文上下文编码，max-pooling丢失词序，不预测患者关系。它既有 `get_match` 短语链接，也有真实 `annotate_text` 全文候选生成，输出 `(start,end,HPO,score)`，end为Python字符串排他位置。固定源码原生枚举 **1–7词**，替换斜杠/换行/tab/连字符，再按非ASCII字母分块；随后阈值过滤和最长优先不重叠选择。其1词与层级表征是可检验的新供给机制，不能据名字或论文跨语料分数声称本赛收益；当前已有PhenoTagger unigram负结果也不能直接替代NCR验证。

训练来源：2019 JMIR论文《Identifying Clinical Terms in Medical Text Using Ontology-Guided Machine Learning》（DOI `10.2196/12596`）和固定源码说明监督训练来自 **HPO名称、同义词**，负例来自Wikipedia随机ngram；fastText在MEDLINE/PubMed无标签摘要预训练。论文的人工标注临床/摘要集合用于阈值、超参和评测；源码也支持带人工标签的early stopping，不能把论文说明当成已发布每个checkpoint的完整训练证明。README标明旧HPO 2019-06-03；release1.1的实际ontology/config版本尚待解包。代码MIT已取得，release模型/fastText未发现独立权重license声明，不能伪称已验包内许可。后续仅取公开模型与本体元数据，不取得外部人工表型语料；公开权重的历史评测来源不自动等同禁止直接使用外部标注集。

**参数：fastText真实矩阵头已核**：输入 `(2,666,817,100)`（666,817词＋2,000,000子词桶）、输出 `(666,817,100)`，两量化flag均0，合计 **333,363,400**，不是按文件字节除4猜的。NCR完整CNN/head数量仍待全部H5 dataset shapes；只看CNN会严重漏计。生产总数 **9,597,467,990** 已含唯一实际full64 LoRA 43,278,336，余 **402,532,010**；加完整fastText后仅余69,168,610给CNN。纸面配置举例约128.8M CNN，故不能先宣布直接追加合规。若替换新增BioBERT而保留旧PubMed/Bioformer，可腾出总524,119,836、扣fastText后190,756,436供CNN；这是待实查形状的一个替换控制方案，不是已验预算或已授权质量运行。固定祖先稀疏矩阵与重复展开派生表示需逐项登记，不能重复计成独立学习参数或漏掉实载变量。

最小后续契约：完整资产SHA/安全成员清单→准确H5/fastText矩阵shape、共享/未用矩阵账目→原生CPU4合成短语与字符坐标（空文本、ASCII/Unicode、标点/连续空格/连字符/换行，避免当成比赛质量）→仅在资源预算和接口通过后，由owner冻结唯一训练内候选试验。必须保持原始full_text和全局offset映射、旧HPO列顺序、HP:None/旧ID到2026有效分支策略；native最长去重或保留raw候选两者事前择一，不看到gold后改边界/阈值。旧TF1.13源码不支持TF2原生，fastText256 B头还显示标准v0.9.1字典布局不一致；目前均是待实际CPU检查的接口风险。

此文冻结于只读阶段。后续新增授权下载/CPU检查另建计划与终点，不改写本阶段未验证项。所有访问URL、HTTP状态、范围、时间、文件SHA见本目录 source-index.json 与访问凭据；无赛事输入/gold读取、NN、GPU、训练或质量评测。
