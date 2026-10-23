# E-20261023-004：监督序列标注首版

状态：进行中；分支 `day1/sequence-crf`，基础版本 `ac8a752`。主线程统一筛选与提交，本路线不操作比赛网站。

## 目标与假设

以训练折监督的 BIO 序列标注学习表型边界，再链接到指定 HPO 并判定患者归属。假设：词形、局部上下文和本体词表特征能够学习字面词典之外的变体，同时减少词表中的非表型误报。与其他路线的主要区别是使用线性链条件随机场学习完整标注序列。

只使用主线程提供的 64 篇训练折；验证 16 篇文献仅用于独立评估。若首版合格，再用完整训练集重拟合 B 预测。禁止使用旧归档词典/实现，禁止把验证金标用于建词典。

## 已查阅的成熟方法（2026-10-23）

- [sklearn-crfsuite 官方教程](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html)：词形、前后词上下文与 L-BFGS 弹性网训练。复用软件包，独立实现本任务特征与输出。
- [BiomedBERT 模型卡](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)：MIT；从 PubMed/PMC 无标注文本预训练，作为有 GPU 后的监督 token-classifier 升级候选。未下载或使用权重。
- [GLiNER 官方仓库](https://github.com/urchade/GLiNER)：成熟零样本/可微调 NER 框架；具体 checkpoint 的训练监督资源尚未核验，因此本次不直接使用未知来源权重。
- BioBERT v1.2 的 Hugging Face README 请求返回 HTTP 错误，未据此断言模型不可用。

当前运行资源只有共享 CPU；远端 GPU 正由主线程接入。先完成可复现 CPU 首版，不把缺 GPU 解释为路线效果差。

## 计划实现

1. 按原 passage offset 保留全局坐标，BIO 标签来自训练金标；明确处理重叠实体和否定项。
2. CRFsuite 词形/邻词/本体词表特征，固定首版超参数。
3. 仅训练折别名 + HPO v2026-06-23 概念链接；所有输出 ID 限于 HP:0000118 分支。
4. 使用患者给定指称与局部上下文做归属，完整覆盖指定患者，包括空集合。
5. 用主线程统一新评测器报告四项近似分数，并执行坐标/ID/患者覆盖校验。

## 环境与产物

自有环境 `/workspace/patientphex-runs/day1-20261023/sequence-crf/.venv`；大产物位于同一路线目录的独立 run_id。CPU 线程限制为 1，不写入权重、全量预测或日志到 Git。

首个最小自造病例 smoke 已通过：原始非零 passage offset、末尾实体边界、无效 HPO 排除、指定患者覆盖，以及输入中恶意注入 entities/association 不影响预测。CRF 训练 3 轮，1 文档 10 token，端到端约 2.71 秒；这只是工程检查，不是验证分数。指定本体解析得到 19,120 个概念、42,567 个名称/同义词（含该自造训练别名）。

正式划分尚待主线程就绪；实际命令、指标、耗时、输出 SHA-256 与结论将在完成后补充。
