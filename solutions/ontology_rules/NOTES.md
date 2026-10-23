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


上下文分类概率经验：对完整候选池做1:3负采样，再给正例权重3，优化后的有效先验接近.5。直接把raw softmax当总体候选概率会产生大量误报；可用训练池先验做odds修正，但文献内采样并不保证严格label shift，仍需独立折验证。当前修正改善CE单独分数但不足以胜树模型；不得把校正本身当达标证据。


二层候选的再交叉验证边界：现有3×train32→inner16候选足以给完整train48拟合二层模型、评价外层fold0；但若直接把这48篇候选行再按文献划成32/16选阈值，meta训练行的上游32篇生成器可能含meta验证16的标注。不能仅因二层按文献split就称整链文献隔离。若需要严格阈值OOF，应生成对应train16→inner16上游候选，然后meta train32→inner16。迭代增加层级时先画出每个生成器可见文献集合。

归属模型不能只比Accuracy：本轮宽池患者-概念只有约2.6%正例，固定p.5提高precision但降低recall；与规则并集的新增FP须区分错患者、背景正实体不属任何患者、概念识别误报。两折union都负，不把“增加模型”自动当增益。


真实PhenoTagger与简化试验的边界：官方BERT分支微调整个encoder、CLS→全HPO+reject多类softmax、maxlen32/Adam5e-6/batch64/默认100ep；旧双头冻结encoder，只学二元表型/肯定，HPO链接另由SapBERT做。后者负结果不是前者复现。上游成品checkpoint未公开完整选择谱系；README示例GSC早停不能证明所有权重都如此，也不足以作为纯远监督来源保证，故本轮从通用Biomed和2026新head独立训练。不得把HP:None reject与本赛真实未链接表型-1混淆。

短文献负例不足：将总负样本平均分配给每篇文献会使短文本在避让gold/本体词面后耗尽；不复制负例硬凑。当前改为每篇最多两倍配额的合格池，再全局无放回抽取到正负1:1，记录实际每篇分布及全部来源，仍只是弱负例。

多类远监督的诊断：HPO正词面与None弱负约1:1时，19k随机分类头初期可以只学None而让总loss/accuracy看似变好。必须拆分正HPO与None的loss、accuracy及None输出比例；用固定终点训练样本自检区分欠拟合和推理链故障，不能据此选heldout checkpoint。早期“先做语义原型、不从随机19k头起步”是当时的低成本建议，已在Sap/树/二元头对照后按root决策转向真实全encoder多类方法，不是禁止研究该方法的规则。

当前30ep之后才考虑的未检验假设：若正HPO仍欠训练，可用同一Biomed encoder的HPO词面向量按类聚合初始化类原型，或使用规范化metric classifier，减少随机头稀疏曝光。必须同一编码空间、固定类词面/多ID冲突与norm/temperature协议；已有Sap bank不能直接作为Biomed分类器权重。当前训练、LR、epoch和p.95推理均不因此改动，不预先声称原型更优。

弱负“去重”的范围：当前每文献词面/坐标不重复，但不同文献可出现同一字符串（58,559行只有51,574全局唯一词面）；完整词面不与已知正例冲突也不排除其包含正例子串，不能把所有弱负都称人工确认负例。实际全局计数与160子短语风险已独立审查，未据此临时改冻结数据。

断点校验需先确认两份作业均已到比较终点：本轮第一次读取split文件时仍step2，和direct4自然不同；managed退出后两边step4、model/Adam/全RNG逐项完全一致。不能把进程尚在运行的旧checkpoint误报为resume数值漂移。训练自检的样本顺序、offset与Adam moments/RNG都应验证，不能仅凭重新加载成功认定恢复正确。

原型初始化的成熟来源：[Qi等，Low-Shot Learning With Imprinted Weights，CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Low-Shot_Learning_With_CVPR_2018_paper.html)将同一网络embedding按适当缩放直接设为新类权重，并可继续SGD；论文的视觉任务结果不等于本赛医学文本收益。已保存公开abstract/来源SHA，不下载外部标注数据。该文支持可检验的初始化思路，不决定本任务温度、负类原型或预算。

冻结阈值边界的差异：本项目采用原始softmax p≥.95；上游v1.2先round(p,5)再严格>.95。两者数值默认相同但边界不逐位等价，本轮保持已经冻结的项目适配，不因为审查发现该差异回扫阈值。


- 成熟AutoPCR结构参考（不改变当前100ep控制）：root缓存MIT代码c2b46a9c088c696533a43c5cb8e50c7500f055fe与btag304论文于外部research/autopcr。其BioNER/句法协调短语拆解、Sap置信路由、定义/同义词链接与本体hard正负自监督可作为后续假设；现成30B/80B超过模型限制、Stanza i2b2人工临床源/GSC选择不能照搬。本项目8B链接对照不应称完整AutoPCR复现。当前仅记录，不取其语料/结果/权重。


### 协调短语公开方法研究（100ep训练期间只读，尚无新模型实验）

AutoPCR固定MIT commit c2b46a9c088c696533a43c5cb8e50c7500f055fe 的 [benepar.py](https://github.com/yctao7/AutoPCR/blob/c2b46a9c088c696533a43c5cb8e50c7500f055fe/src/ee/benepar.py) 用依存conj与成分树，把共享前后缀同每个分支重组成虚拟query；其区别是查询词面不连续，单纯扩大1–8gram不能产生此词面。普通≤8词名词短语通常已在宽池，不能默认句法提取提高召回。

关键迁移限制：上游可给虚拟query保留整个父短语坐标，论文按同概念+跨度重叠评价；本赛要求精确offset/length且text必须等于原文切片。必须分开source token intervals、model query、output原文span，不能用虚拟字符串长度造坐标。后续可分别控制：①仅新增连续句法span；②固定物理span仅改共享成分query；③固定概念/关系仅比较父跨度与分支anchor投影。每项都应先训练内冻结，不同时混改、不把重叠命中冒充exact命中。

只读fold0 train48计数：3612肯定记录去重3609跨度，内部含and/or/逗号83（2.30%），其中26多ID、57单ID；同词法句1923相邻非重叠跨度对中1598的间隔含连接标记，主要是独立列表。列表不证明共享头/ellipsis，也不证明当前模型漏检；83跨度与1598跨度对不能相加估收益，and不能一律拆成多个肯定表型。未加载解析器、未下载模型、未读验证/dev gold或运行预测。完整备忘、计数定义、匿名例型、可复现脚本都外存，只收[小索引](evidence/round2/coordination-research-index.json)。这组假设不改变进行中的固定100ep。

### 已拟合词面与真实候选精度（固定100ep及3ep适配）

100ep训练词面probe全对不能证明文献表型识别：同fold0原输出新增191仅60TP/131FP；独立Sap固定.65/.92+pure ID一致过滤后36中22TP，Score .725539。加入1749真实肯定词面、回放及3ep新Adam后，训练real acc .99825但raw .694907、相同Sap .721696，均弱于父。该具体配比结案，未用失败错例重选词。下一信号应考虑原文语境/候选真伪或概念竞争，而非默认更多词面曝光。

root复核冠军卡与旧context_calibrate.py的结构差异：旧CE是±180字符/max160的已选span-ID真伪分类，负采样/类权重后用直接阈值或固定线性混分；没有将严格OOF CE输出作为特征重新fit二阶段树，也没有top-k概念对比。旧CE1/3ep负结果不能排除后两机制，但严格嵌套OOF有额外成本，当前不立即重训。冠军卡是方法描述，不是可复现运行代码；80篇成品权重仍不能用于本地留出分数。等待新候选池提供明确额外信息再定义控制。

原目标划分必须独立报告：同Pheno100+Sap政策在开发fold0由.710359升.725539，迁移原dev（Pheno48/HGB-Sap64）仅.646244→.646360。过滤新增28实体9TP19FP、13关系3TP10FP，关系F1还略降；内折局部增益和名义总分最高不足以支持继续投入或升级默认。保留混合训练规模这个解释边界，但不能用“再训64也许更好”无证据追加昂贵训练；当前关闭Pheno-only追加预算，不从原dev错例调阈值。

有限候选不足与NIL不同：top5没有等价概念，只说明本次召回不足或语义不确定，不能证明整个HPO无对应概念。首个9B有限选择控制统一ABSTAIN、不输出新-1，明确损失未映射实体召回。原始NO、非表型拒绝与真实NIL是三个不同语义，不能把一个通用拒绝码当作它们共同的gold标签。

患者关系CE已经实际训练过，勿重复当新机制：序列分支9453c0f的run_neural_relation.py（SHA09ae97d291add225b295fe5d053f84bb35a788efaca66d9c1ef5240d97b2ae5d），train48的5114个患者-概念pair/1327阳性，BiomedBERT4ep共130.046秒；同旧CRF实体fold0 .617746，弱于rules .656999。已有target/other/phenotype三组[unused]标记、header患者ID/别名/HPO名、最多3段±220字符启发式排序拼接，max384 only_second截断。训练用gold正实体而非OOF预测候选；目标marker仅给定mention实际落窗才有，未单独审计截断后保留，多证据并非学习式MIL。证据在sequence_crf/fold0-neural-relation/及其evidence/round2/fold0-neural-relation-manifest.json，源码与实际manifest一致。若日后只改OOF候选分布或MIL，须说明该最小变量及训练端证据；当前不实现或再训练，先完成176链接控制。

后续root按训练证据批准一次不同目标的max-MIL过滤：1474个OOF doc-HPO袋中783正/691负，原规则394FP中319是整篇概念无患者关系，其中177仍有gold正实体。过滤器只判“是否关联任一给定患者”，不重分患者；75个正确概念错患者FP不在它的能力范围。417/783正袋所有±220窗都无给定patient mention，因此不能将无marker当负标签。旧患者CE与旧aggregate树负结果均保留；新控制同时改变候选分布/目标/聚合机制，若有益也不能归因为单一因素。[训练诊断](evidence/round2/relation-bag-training-diagnostics.json)与[唯一固定运行](evidence/round2/relation-mil-run-plan.json)。

BiomedBERT词表不等于BERT原始词表：本地实际30522词中无[unused1/2/5/6]，直接convert四者均为UNK=1，初始CPU guard因此失败。新MIL显式注册四special tokens为30522..30525、固定seed扩embedding，并逐出现核完整表型和成对patient标记，3949个实际输入均通过且无截断。旧患者CE传的是字符串，可能拆成普通WordPiece；不能把本次直接ID错误外推为旧模型必然无效。失败事实/修复在外存relation-mil-fold0/initial-marker-failure.json；检查在[CPU证据](evidence/round2/relation-mil-cpucheck.json)。


### 多层OOF要追踪候选生成器的训练来源

将旧CE的11180训练行按文献切成train32/heldout16，表面上没有同文献，却不能构成严格OOF：旧行来自另一个train32候选生成器，它包含未来CE-heldout16的全部标签。实际三种切片的两个候选来源均与CE留出集交叠16篇。应将每个派生资源的训练来源沿整个链追到原文献，而非只看最终分类器的行ID。记录在ce-oof-feasibility/isolation-matrix.json；本次在新训练前发现，未产生不合格新模型或分数。

若保持旧CE的OOF候选训练recipe，需三种train16生成器各预测另外两组16，产生六池再组成三个CEtrain32。旧outer48仍有效，可复用14511分数并补32个CRF-only键；新增内层45113分数。若改用CE自身train32候选，heldout隔离可以成立，但训练自包含使难度/别名分布改变，且必须重训一致的outer48，不能称旧recipe复现。当前成本建议只是估算，无新GPU/训练；[可行性证据](evidence/round2/ce-oof-feasibility.json)。


严格OOF CE作为HGB唯一新增列已实测，不能再列为“未尝试”：三层候选/分类器隔离、84→85列/.4、旧outer模型只补32键，.714808→.701372四项均负。它区别于旧直接CE阈值/blend，但本次仍不能证明所有上下文特征无用；当前固定recipe关闭，不因目标差距继续重复同变量。

强已选OOF池与任意词面链接难度不同：train48三nested CRF+Sap的720误报仅44是正确正跨度上ID错，1364漏原子中45为正确跨度但ID错，其余主要边界/漏检。45目标在同折当前全候选与精确词典中均不可直接选；故现有固定ID二元CE只能筛选，不能重新找出这些概念。没有本组冻结top-k时保留未知，不把另一fold的Top8覆盖混入。对于新GLiNER/复杂短语候选，错误组成可能不同，当前统计不外推所有新span；只支持先定位候选与跨度瓶颈再分配预算。

HPO alt_id需要单列解释：上述45个raw-ID漏原子中4个gold为alt_id，模型已经输出相同概念的canonical ID；冻结v2不归一化，因此本地仍记错，但不能把这4个解读成语义链接失败。诊断允许另报等价概念，不能悄改金标/评分来增分。


受控NER增强研究见[短提案](experiments/P-20261023-NER-AUGMENTATION.md)：同HPO完整实体替换有生物医学少样本文献依据，但收益在大训练量明显减弱；相邻段落上下文只有通用NER证据。提案时两者均未在本项目训练，不能把可替换词面数量当预计召回；随后仅首项执行了下述有界控制。保持既有BCE与当前large作业，不重跑Pheno词面分类或百万伪标签。


同HPO整实体增强已尝试，不应再次列为未检验方法：原1150窗保留并加353个无损变体，fresh medium固定10ep/3760步，相同Sap/关系只让nested16 Score增.0001915（1TP/−1FP，关系完全不变）。该微增不能支持继续扩展，也不能解释为同义词增强的独立因果收益，因为训练步数同增30.56%。保留原基线，关闭当前配方；未达到.8。

迁移检查点不能只比raw span键：旧32 A800→A30的1378跨度键完全一样，但浮点分数有约1.72e-5漂移。只有同目标Sap环境、固定阈值与装配跑完后，完整pred字节精确恢复历史f203，才能报告全链复现。旧32模型仅用于对照推理，新模型必须从已核公共medium fresh训练，不能把train48或old32继续训练冒充控制。


FLERT 风格上下文不能直接拼入 GLiNER tokenized_text：原 SpanProcessor 会枚举上下文的未标注负跨度，并允许其参加解码；仅高层输出后过滤不能消除此前 greedy 重叠抑制。最小接口应让 transformer 看到上下文，而 words_mask／text_lengths／RNN 和跨度头只保留原目标窗，原 masked BCE 分母与 NO 语义不变。跨 passage 的虚拟 encoder 序列不对应连续原文，输出必须走原 word_offsets。1150训练窗97.48%可加上下文只是可行性，39.04%跨 section，不能当预计增益；当前统计和源审通过，实际专用接口 CPU 检查54.061秒完整通过；新增接口独审PASS（42fb9b45…），见[上下文提案](experiments/P-20261023-NER-CONTEXT.md)。

- 2026-10-23 context strict-load：Transformers 在构造 encoder 后会把嵌套 `_attn_implementation_autoset` 设为 true。无权重 processor 的配置与实际构造模型不能未经检查就按整个字典相等。先实查唯一差异，只允许这个 false→true 构造标记、要求 eager 和其余字段完全一致；不要泛化忽略所有私有字段。CPU 失败证据在 `ontology_rules/ner-context-control/run/cpu.state.json`，修复独立 `attempt2`。
