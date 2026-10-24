# 患者归属与池外补全：只保留一个优先机制

2026-10-24，只读研究＋固定训练32篇CPU统计；**没有读新dev/F0/F1标签、模型加载/训练/推断、评分、MCP或下载权重，没有改任何成功实验/仓库。** 当前已接受原dev .755239908536 / macro .660598133；保持实体两项时，距.8还需关系两项均值约+.08952。小幅共识不能单独补足这个差距。

**建议第一优先级：患者条件的多跨度联合抽取，使用患者—HPO多实例关系监督，取消“最近跨度必为该患者证据”的强制伪标签。** 这条机制在已核历史中尚未真正执行；有现成BiomedBERT/跨度头/坐标与规范化组件，但新的患者条件头和联合目标尚需实现。第二项不凑数：BioMistral确是未见执行的领域基座备选，但当前证据不支持把一次换模型列为同优先级GPU项目。

## 1. 只读训练证据

`census.py` 可直接CPU复现，本次最终0.337秒；只读已冻F2+F3 train32（32文献/87患者）和固定公开模型预测。明确核训练IDs与旧provenance相同、排F0/F1；没有新增held标签入口。

| 固定统计 | 数量 | 含义与限度 |
|---|---:|---|
| 有效canonical患者—概念关系 | 814 | 与评测literal数不同，不能称v2得分或oracle |
| 原文存在可定位正表型跨度 | 814/814 | 只证明词面存在，不证明跨度属于该患者 |
| 同关系有多个候选跨度 | 426/814（52.3%） | 最近一个并非唯一可行监督 |
| 既有grounded最近目标，另一患者更近 | 170/814（20.9%） | 是歧义风险，不是170个已证错标签 |
| 最近同概念正跨度距该患者任一锚点>500字符 | 208/814 | 长距离可见性风险，不等同必须多跳 |
| 同概念同时属于多位给定患者 | 223关系 | 不能用患者间单选softmax/强制唯一归属 |
| 三固定public概念池外但原文有跨度 | 85/814（10.4%） | 只相对这三个public来源，不冒称强全流水线缺85个 |
| 上述85中多跨度 | 17 | 还有68个单跨度，候选供给与归属应分开诊断 |
| 真空患者 | 5/87 | 必须保留空集合，不保证每患者至少一阳性 |

完整训练统计/来源在`census.json`。基于原grounded公共规范化器，4个无效关系literal明确排除；所有原文offset/text都核验。文档一次编码字符552042，若每患者完整重编码1530337（2.772倍）；这是字符代理，非真实token/GPU成本。

## 2. 机制及与已做工作的实质差别

成熟依据是 **CasRel的subject→object条件标记**、**关系多实例学习的证据选择**，再复用本项目已验证的GlobalPointer/完整窗口实现。CasRel本身是句子级、有人为三元组span标签；这里是明确的新任务适配，不能直接声称原论文方法已复现。不给每个同HPO出现强行贴患者正标签。

- **输入**：给定全文、给定患者IDs及全部原锚点。共享BiomedBERT窗口编码；按患者聚合锚点表示，在每个token跨度的head中显式条件化。每个患者都可选多个跨度，同一跨度也可属于多个患者；患者没有本窗mention仍保留全局锚点表示，不能静默删窗/患者。
- **监督**：一条全局表型跨度辅助目标，用真实训练entity标签学习“哪里是表型”；另对每个训练 `(patient, canonical concept)` 建其全部精确可定位跨度的bag。关系正例只要求bag至少有一个支持跨度，负bag约束全部实例为负。建议先固定max-MIL bag BCE（不扫聚合方式），不把所有正bag成员或最近成员当已证患者证据。未知/无法规范化标签单列，不补成确定负例。全局NER与patient-bag两项损失初始固定1:1；这只是待冻结配方，未经梯度/质量验证。
- **输出**：患者条件跨度分数＋offset，先在全文解码再走已冻public/train32规范化；允许生成旧闭集概念池之外的原文表型。保完整raw候选及分数，固定阈值/规范化政策；Task1强实体和全部B关系保护，主要只检验新的Task2供给。
- **真正新变量**：患者条件的可训练span score及不强制nearest的bag监督；不是在旧已选概念上再做一个二元头，也不是再次无条件GP+Sap。

历史查重已核到实际源码和终点（精确SHA见`history.json`/`sources.json`）：

| 已做方案 | 真实输入/目标与终点 | 此提案为何不同 |
|---|---|---|
| 旧BiomedBERT患者CE/identity bridge | 固定预测概念→±220/384三窗二分类；bridge追加128身份token，已有完整负终点 | 没有患者条件全token跨度输出，不能补池外span |
| 旧max-MIL | **无患者ID**、所有给定patient共用marker，doc-HPO是否属于任一患者；.711354→.702753 | 本次bag键是患者×HPO；不是单纯换max聚合名称 |
| pair-conditioned全文 | 固定880/587概念，patient-concept查询汇聚全文，pair .557825 < 强旧.660942 | 固定池doc-pair分类，没有可训练条件span+NER联合监督 |
| ATLOP式DocRE | 三public池＋全提及汇聚/共有attention；30ep并集.669662 < N.676546，meta也负 | 未用patient-instance bag学习或输出池外跨度；只换图也无充分收益依据 |
| 无条件GP | 1个表型head、8轮；直接追加负，GP特征四折pooled负 | 新目标要回答“谁的哪一段”，不再重训同无条件head |
| grounded患者引用SFT | 全文无闭集候选，但**每个正HPO只选最近的一个跨度作teacher目标**；内部F1/F0已有正 | 本次判别式多跨度/潜在证据目标不同，不重做同quote SFT |
| GLiNER2.5 joint | 公共jointIE患者/表型span→relation＋固定规范化，F1并集负，F0共识也负 | 未在比赛标签上监督patient-conditioned联合span；后续Task1-only适配不等同此目标 |
| 共指/格式/名称 | LLM共指、F-coref marker、patient-first、response-code、标准名称生成已有完成或当前运行 | 不再增加同类marker/改JSON/全空间名称生成项目 |

## 3. 最小可证伪控制、现成入口与成本

现成可运行研究入口：`python /workspace/patientphex-runs/round2-20261023/research/patient-attribution-completion-readiness-20261024/census.py`（CPU、无held gold）。源盘点入口为`history.json`。**本报告没有实现新训练器，不能直接把旧GP main当新模型启动。** 后继应独立目录/分支，复用以下已存在组件并显式新增患者条件头/数据bag：

- `explore-span-normalization/solutions/span_normalization/{prepare.py,gp.py,train_infer.py,check_geometry.py}`：完整窗口、原始offset↔token、无条件span头及持久状态；旧GlobalPointer只输出B×L×L，必须新增患者维度，不能误称已实现。
- `explore-document-relations/solutions/document_relations/{prepare.py,encode.py,marker_encode.py}`：全文患者全部mention映射、物理mention唯一owner、完整窗口。只复用机械接口，不复用见过新hold的task checkpoint。
- `explore-grounded-generation/solutions/grounded_generation/{prepare.py,normalization/strict.py}`：canonical/NIL/compound处理与可定位训练span；**不复用construct的nearest teacher选择**。公开Sap bank及已核asset可复用，train词典须仅F2+F3。

建议顺序：先全部train32及纯F1输入CPU构造，不碰held标签；列完整bag可表示覆盖、超过原max-span长度/跨窗跨度和冲突。不能为迁就旧32-token跨度上限删训练目标；如需改变上限，必须事前显式冻结、重新核内存。CPU数值合同覆盖正bag多实例、全负bag、多个患者同概念、空患者、错误offset、跨窗重复和有限梯度；最大真实训练窗（最多患者/跨度）再做一次有界工程检查，丢弃probe权重。

随后唯一fresh train32→完整F1，建议沿小模型固定8ep与既有seed、末轮，不做epoch/cut搜索。共冻`B∪new`主端点与`new-alone`诊断（实体仍同B），各只1v2，完整报告micro/macro/新增TP&FP及池外占比；旧强B原metrics复用。若不胜强B，关闭该配方；若正，再固定F0同配方复验，而不是原dev选患者/阈值。若只在旧pool内有信号，也只支持归属机制，不能声称完成池外召回。

成本仅代理：旧同BiomedBERT GP32八轮+全推断95.52秒、DocRE32三十轮+F1 639.32秒；共享编码条件头应明显小于重新每患者跑9B，但bag聚合/反传需实测。建议首真实工程120+30秒；完整训练及推断申请900+30秒前须用真实吞吐重估，不能靠旧95秒保证。无需新下载，基座约110M＋小头，参数实际核后应远低10B。若只能用逐患者全文重编码，字符代理2.772×且encoder路径改变，先报成本不偷偷延长。以上是建议预算，不是已获GPU授权。

## 4. 第二候选筛查：BioMistral暂不立项

在当前所有worktree的md/json/py检索未找到BioMistral/MedITRON执行记录；这只是已搜库存结论。MedGemma已真实跑短输入与全文两组，均负（单独.603702/.592419；并集.644663/.627967），不能重新列为未试领域模型。

BioMistral-7B固定revision `9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5`，model card声明基于Mistral-7B-Instruct-v0.1，以PMC Open Access文本继续预训练、Apache-2.0，发布表标为Further Pre-trained。此描述与允许公开预训练模型相符；卡上QA SFT benchmark不是本赛人工表型监督，更不能把benchmark权重默认拿来使用。配置形状推导7,241,732,096参数（未核实际weight header）；本次只取卡/config/tokenizer-config元数据，**没有下载权重**。

限制很具体：继续预训练sequence length2048，config上限32768/sliding4096不等于16k全文质量已经可用；模板只接受user/assistant、不接受system，无法直接把当前Qwen模板换模型名。还无本地权重/实际token及显存合同，预计需至少约14.48GB BF16模型存储，ModelScope镜像/版本尚未核，不能声称现成可跑或预算已知。若root后续希望仅测试领域预训练变量，先保持一个已冻短闭集任务/原train32标签，完成原生token与参数/许可门，再独立fresh SFT及完整F1；这仍是模型先验控制，**不改变池外供给上限**。目前不建议为此占五卡现有训练，也不为凑第二方向下载资产。

## 5. 不重开的旁支与公开来源

GAIN的mention/entity双图确实不同于现有ATLOP的独立窗口attention交集；公开MIT代码可借鉴。但现有DocRE训练727阳性中只有7无共有attention支持，且pair-query/30ep已说明“有全局表示”不保证得分，尚缺对当前错误的独立train-only图路径供给证据，暂不另立第二项目。DREEAM需要evidence监督或可靠teacher；本题无人工患者证据span，不能把上述所有同HPO提及当真evidence来复制论文。

公开原文/代码均已保存网页及SHA：CasRel ACL2020 `2020.acl-main.136`，MIT代码commit `3faef3f9cc3da1523a01587635f3d33a5603729e`（原实现Keras/keras-bert，需适配现PyTorch）；Selective Attention MIL ACL2016 `P16-1200`；GAIN EMNLP2020 `2020.emnlp-main.127`，MIT代码commit `ffe186f614bc290e53a86166ec9909596a9153f5`；DREEAM EACL2023 `2023.eacl-main.145`。不下载其DocRED/NYT/WebNLG标注或benchmark checkpoint。一个MRC猜测URL404已记录，不把未核链接引用为来源。

结论只是机制准备，不是高分承诺：优先验证患者条件跨度＋bag监督能否提供可用且独立的新关系。若完整内部对照仍无增益，应如实关闭，不能把训练词面供给数字当预期TP，更不能据此承诺达到.8。
