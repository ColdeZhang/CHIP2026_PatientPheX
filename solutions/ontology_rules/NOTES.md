# 第二轮本体路线：成熟方法调研

2026-10-23；只读研究，未修改仓库、未读 dev 标签、未调用外部表型标注集。保存来源快照与 SHA 于旁边 `research/sources-manifest.json`；下列收益均为待验证假设。优先顺序：先文档内缩写与词面规范化小消融，再考虑本体远监督模型。

## 1. 文档内缩写解析优先于全局短词词典

**可靠来源**：Schwartz & Hearst (2003) 的字符逆序对齐法，现成实现见 [scispaCy abbreviation.py](https://github.com/allenai/scispacy/blob/main/scispacy/abbreviation.py)（Apache-2.0）；PhenoTagger 的 `abbre_resolution.py` 也采用该算法。[PhenoTagger README](https://github.com/ncbi-nlp/PhenoTagger/tree/10f8ee3e1ea4c26fb44c47df28a555082748ebf2) 默认 `abbrRecog=True`。

**可直接实现**：

- 为每篇文献从原文提取 `long form (SF)` 和 `SF (long form)`，短名 2–10 字符、至多两词；逆序匹配短名字符，首字符必须位于长名词首，长名最多 `min(len(SF)+5, 2*len(SF))` 词。
- 只在长名能由本体/训练折别名高置信链接时建立本篇局部 SF→HPO 映射，重复短名可沿用；不把新缩写写成跨文档常量，不依赖人工猜缩写含义。
- 若同一 SF 在文中有多个不同长名，保留作用域或放弃自动传播；尤其不要用“首个定义”覆盖全文。对有明确定义、却映射到非表型长名的短名，可抑制全局词典给出的表型误映射。
- 先保存字符位置，扩展发生在匹配字典层，不能改写原始文本后再计算 offset。

**当前训练数据适用性**：只读 train64 发现约 808 个大写括号缩写模式（粗 regex 计数，非 808 个合格表型缩写）。现模型只按是否大写允许短名，未解读本篇定义，因此该改变有明确机制。单独报告新缩写召回、被消歧去掉的错误、以及正文/其他节差异。

## 2. 有界的词形／词序无关匹配，保持方向信息

**可靠来源**：[ClinPhen get_phenotypes.py](https://github.com/kingmanzhang/clinphen/blob/master/clinphen_src/get_phenotypes.py) 使用 WordNet noun lemma、医学形态变化与 low/high/abnormal 各自词汇组，按子句求词集合交集；其 README 限学术、非营利和个人免费使用，若复制代码需保留来源并确认适用许可。[fenominal HpoConceptMultiWordMapper](https://github.com/monarch-initiative/fenominal/blob/main/fenominal-core/src/main/java/org/monarchinitiative/fenominal/core/impl/hpo/HpoConceptMultiWordMapper.java) 将去停用词后的词汇簇集合相等作为多词概念匹配，词序不敏感；fenominal 为 GPLv3，建议借鉴算法独立实现，暂不引入其资源包。

**可直接实现**：

- 保留当前精确 trie；另建 `(词数, 排序后的规范化词 tuple)` 索引，为连续短窗口提供词序变体候选，例如 `dilatation of ureters` 对 `ureteral dilatation` 仍需要形态/同义关系，而单纯乱序只能处理相同词。限定同句和最大词距，不能把整段词袋都当一个实体。
- 医学形态映射可首先从指定 HPO 的**同概念同义词**中估计单词替换；只采用有足够支持且不引入方向冲突的替换。单复数变体可直接生成，`hypo/hyper`、`decreased/increased`、`left/right`、`absence/presence` 绝不合并。数词与解剖部位不能删除。
- 首轮只试少量功能词（of/the/a/an）忽略以及二至五实词窗口，要求边界仍为原文 token span；不把程度词、具体解剖修饰语一概删掉。缩写、同义词范围 EXACT/RELATED、词形、词序必须分别消融。
- 多个 HPO 共享规范化键时保留候选并用训练折先验/上下文判别，不能任取排序首项；不能机械把全部候选视为真实复合表型。

**为什么不是直接安装工具**：ClinPhen 输出的是文档级概念，不满足本比赛精确 span 与多患者归属；它的子句/旗标过滤可借鉴，但要自行保留 offset。fenominal 默认词汇资源与本体也未必等于指定 2026-06-23 版本。

## 3. PhenoTagger 的本体远监督可补规则外的映射

**已核验来源**：[ncbi-nlp/PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger/tree/10f8ee3e1ea4c26fb44c47df28a555082748ebf2)，MIT，当前核验 commit `10f8ee3e1ea4c26fb44c47df28a555082748ebf2`；论文 DOI [10.1093/bioinformatics/btab019](https://doi.org/10.1093/bioinformatics/btab019)。它明确宣称 ontology-driven，无需人工标注训练数据。

**真实训练链**：

1. `Build_dict.py -i hp.obo -o dict/ -r HP:0000118` 从本体 names/synonyms 建词典、lemma 与映射。
2. `Build_distant_corpus.py -d dict/ -f NEGATIVE_TEXT -n N -o distant/` 把每个名称/同义词视为概念正样本，从未标注文本随机取 n-gram 负样本，并过滤标点/不合适首尾 POS。
3. `PhenoTagger_training.py -t TRAIN -d DEV -m pubmedbert -o MODEL/` 拟合 HPO 类别加无概念类别；`-d none` 时使用固定 epoch。预测以最多 8-gram 候选和 dictionary+neural 组合；默认 neural threshold 0.95 是原论文设置，不能保证适合本比赛。

**本项目适配**：用指定 HPO 构建远监督正例，仅从当前训练折原文采负例；已有 BiomedBERT/SapBERT 可作为 encoder，先尝试对比学习的同概念短语或候选分类/重排，避免一次训练 19k 类别后期全模型过大。对精确 trie 外的高召回候选用模型判别，精确匹配与语义候选分别报告；普通医学预训练不等于人工表型标注。最终仍需患者归属模块。

**不可直接照抄的地方**：发行版模型对应 HPO 2023-10-09；README 训练示例以外部人工 `GSCplus_dev_gold.tsv` 选模，不应拿其预选 checkpoint 的实验声明代替本比赛数据约束。最稳妥是新建远监督训练且只用本训练折内验证。另源 `Build_dict.py/get_all_child` 构造父子边时只在循环后保留最后一个 `is_a` 父节点，不能拿来替代本项目已正确处理多父节点的 HPO parser。

## 4. 当前实现可以低成本先验证的标签问题

从 train64 实测实体 note 分布：None 4762、NO 155、D 3。当前 `Gazetteer.__init__` 把 NO 也计入 options/training_aliases，并在噪声过滤统计中把所有实体都视为 annotated；公共 evaluator 忽略 NO，CRF linker 已显式过滤。

建议把“NO 不计正例先验/白名单”作为独立消融，而非直接宣称修复必然涨分。NO 中的词面在词典层仍可能有价值，但不能以肯定病例的标签频次使用；否定出现的 span 不能作为实体/关系正例。未读取 dev 例子，不作定向补丁。

否定处理本身可继续参考 [negspacy](https://github.com/jenojp/negspacy)（MIT）四类 pattern：pseudo、preceding、following、termination；已有实现基本具备。更重的 NegBio 依存模式有适用性，但要先确认错误主要来自否定，不能期待替换工具直接解决未知表型召回。

## 最小实验顺序

1. 固定 train48→inner-fold0，单独 NO 正例过滤、文档内缩写，再做二者组合。
2. 固定同一输入，加词形／限定停用词乱序匹配；记录候选增量 TP/FP、实体与关系分项，表现稳定才跑另外三折。
3. 若规则改进仍明显受未知表达限制，启动指定 HPO 本体远监督，训练/验证资源严格折内；不要把更多全段词袋关联或广撒上位词当成召回提升。

本报告仅研究，不含任何新模型分数或达标声明。

## 后续深入：PhenoTagger 实际候选、训练与融合，不安装 TensorFlow

### 源代码确认

- `ml_ner.py/build_ngram_testset_filted()` 每句枚举 2–8 gram；拒绝含 `, . ! ; : ? ( ) [ ] { }` 的窗口，拒绝首尾为介词、冠词、连词、代词等 POS，以及 is/are/was/were/had/have/has/be/been/also。单词主要走字典通道。没有预先神经门控，也没有 ANN。
- `allentity` 列表虽保存唯一短语，但仍向模型输入写每次窗口，因此不是实际编码去重。源中 `Ngram=min(Ngram,len(sentence))` 被修改后跨句保留，短句可能永久收窄后续窗口，独立实现应修正。
- `Build_distant_corpus.py/train_pos` 为各 HPO 名称与同义词原始/lemma 形式生成概念分类样本。`train_neg` 在纯文本随机选起点与 1–9 的长度（numpy randint 上界10），仅排除在词典内的完整字符串，再过滤 POS/标点。原方法不保证负例中没有未见表型。
- `decode_tsv` 取模型 top1，非 `HP:None` 且概率超过阈值才输出。`combine_ml_dict` 给字典匹配1.0分，与神经结果合并；重叠区同 HPO 留最高分，同 span 不同 HPO 取最高分。默认可保留不同概念重叠，选最长是独立可选步骤。

### 规模诊断（只有 train64 原文）

用独立 Python 首尾功能词/标点过滤近似（不声称等于 POS）：2–8 gram 共 **1,427,181** 窗口；过滤后 **258,503** 次、**221,103** 唯一词面；排除现词典占区后 **218,027** 次、**185,959** 唯一词面，统计1.70秒。词典占区来自 train64 拟合，统计只用于估算吞吐，不能当跨折候选训练构造。证据 `research/candidate-count-diagnostic.json`。单靠唯一短语缓存约节省14%，不是数量级节省。

### 推荐最小代码接口（独立实现，不照搬未知许可代码）

`semantic_proposals.py` 可分为五个可缓存阶段，所有文件名/数据/模型版本放 manifest：

1. `propose_spans(doc, dictionary_spans, gate, config) -> list[Span]`
   - 从本篇 passage 原文 token offset 构造候选；词典已有可靠区先不生成普通2–4gram补充，长跨度交给已有规则/CRF候选。禁止跨 passage / 句末标点。
   - 使用当前训练折训练的轻量 char/word binary span 门控，例如 LogisticRegression 的 `char_wb(2,5)` 加边界词/长度。门控目标是表型跨度，正例含真实 `-1`；负例是未重叠 gold 的窗口、过长/过短边界扰动和 NO。先在内部验证评估门控召回，不能直接设置“每句只留 top1”后宣称召回问题解决。
   - 记录门控前后候选数及 gold 覆盖上限；这一步只减少编码量，不输出最终实体。完整监督候选生成须折内排除当前文献的派生别名，避免训练自包含。
2. `build_hpo_index(hpo, train_aliases, encoder) -> AliasIndex`
   - 固定 HPO branch、EXACT synonyms 与训练折别名，去除 NO 在正例先验的贡献；保存词面→原子/复合 ID。SapBERT CLS+L2 编码，全 aliases 矩阵可一次缓存；身份=模型 revision+权重 SHA+HPO SHA+训练文献哈希。
   - 不训练19k-way随机分类器起步；先使用语义原型检索对照。相同概念多个 aliases 的最高相似度归为该概念；top2 **必须是不同 HPO**，否则两个同义词会制造虚假小 margin。
3. `encode_unique(spans, encoder) -> dict[surface, embedding]`
   - 同一个encoder输入只编码一次，映射回原offset。若输入含上下文，则缓存键必须包含上下文，不能只用词面。SapBERT 标准输入可只词面，保持模型预训练用法。分批 64–256、max length 32/64，避免把整篇文章当一个向量。
4. `rank_candidates(spans, index, calibrated_model) -> CandidateScores`
   - 初版保存 top1/top2 不同概念相似度、margin、精确/形态/缩写/语义来源、训练折词面可信度、span门控得分。LogisticRegression 学“候选 span+ID 是否完全正确”；训练别名特征和候选来自交叉拟合，别用 dev 反向构建。
   - 可再加 BiomedBERT cross-encoder：`[context with marked span] [SEP] HPO name + short definition`；只对每span top3候选算，正例来自当前训练 gold+指定HPO同义词，hard-negative为检索到的其他ID，普通非表型span另外采样以学拒绝。不需TensorFlow，不需大LLM。
   - `identifier=-1` 是真实表型但无HPO，与原PhenoTagger `HP:None` 拒绝类不同；必须先区分实体存在与标准化成功，不能把所有 `-1` 当负例。
5. `merge_candidates(dictionary_spans, neural_spans, config) -> entities`
   - 首先仅增加不重叠高置信候选；单独做允许替换包含短词的更长跨度实验。不要固定词典1.0压制所有神经纠错，也不要统一最长匹配。涉及复合 HPO 的 span 合并仅在训练/词典有明确复合映射时允许，不机械把候选 topK 拼成复合ID。
   - 保留原文切片与完整患者coverage；然后交给现有患者归属模块，以端到端四指标而非实体召回决定是否采用。

### 同赛代码中可以验证的策略（尚未核验其分数）

- [irryxeek/patientphex-chip2026@1e38029/lexical_recall.py](https://github.com/irryxeek/patientphex-chip2026/blob/1e3802970c54a6bd79158257808b74f5c2875f23/reproduction/patientphex_532512/lexical_recall.py)：分别开关 RELATED/BROAD/NARROW 同义词（只无歧义键）、英美拼写/连写 surface、少量功能词去除后的排序词袋。`neural_frontier.py` 用 BioNER proposals + SapBERT top1/top2 margin，分别比较只增加与允许包含替换。
- [AnXin-HITSZ@363df72/baseline_sapbert.py](https://github.com/AnXin-HITSZ/chip2026-patient-phenotype-recognition/blob/363df729fbda89d25861942a535a656a2dc024cd/scripts/s1_identify/baseline_sapbert.py)：仅词典未占据区1–4gram，唯一词面批量 SapBERT CLS 检索，cosine阈值后按分数贪心。其候选仍可跨句、只用top1、没有拒绝学习，不能假定代码已适合直接采用。
- [SuperRs7@d7c4453/hpo_dict_match.py](https://github.com/SuperRs7/chip2026/blob/d7c44531395267717f75b6c872a596e04cae17e0/patientphex/scripts/hpo_dict_match.py)：规范化时同步保留字符反向映射，折叠连字符/英美拼写而不破坏原offset。

以上只读源码及README，未下载或使用其数据/预测/答案。除已核验的官方 PhenoTagger MIT、scispaCy Apache2 等外，上述同赛仓库许可未在本子任务独立核验，当前仅研究思想；不能把README分数当我方复现结果。


实际实现：`enhance.py` 保留CPU负结果；`gpu_candidates.py` 用冻结SapBERT+训练折表型判别器；`merge_semantic.py` 仅按候选阈值追加本体遗漏跨度并重跑规则归属。不得直接加载已用本赛事全部80篇训练的公开冠军成品来计算dev16留出分。
