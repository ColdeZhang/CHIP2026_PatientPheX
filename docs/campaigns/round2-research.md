# 第二轮研究索引：直接指导可检验的改进

本页由主线程维护，只保留共同方法依据；具体实现、实验过程和取舍由三方案各自记录。下载的论文/原文在仓库外，[来源与 SHA](round2-research-sources.json)可用于核验，不重复查站点。

## 主办方原始证据

已下载并读取[官方评测综述](https://arxiv.org/pdf/2610.29606) 3.3、4.2–4.3 节；HTML 路由未成功，PDF可用。论文前三名 B Score .7389/.7247/.7169 是官方不同测试集结果，不当作本地目标或可复现保证。

| 来源 | 可实施的思想 | 本轮落点 |
|---|---|---|
| 官方基线 §3.3 | Qwen3-8B 先扩充指定患者称谓/共指，再判断每批8个表型的归属，局部上下文前后各500字符；greedy，输出上限1800/2600 | 原有既定mentions上的关系对照已完成；全文LLM共指此前尚未实际运行，现由序列路线单独控制验证 |
| Agriv §4.2 | 多个PhenoTagger+监督跨度+HPO候选，SapBERT检索，两阶段LightGBM与上下文交叉编码器；LLM证据及位置特征经OOF融合 | 本体与CRF补候选/标准化，检索研究可学习关系判断 |
| 第二名 §4.2 | 多源候选投票、LLM处理重叠/否定；生成患者HPO集合与候选对分类的互补SFT | 检索零/少样本之后再决定是否折内LoRA，避免只扫阈值 |
| 第三名 §4.2 | 监督跨度、SapBERT、本体图特征；关系模型及LLM assertion复核，保留关系分数用于恢复 | 实体与关系使用可追溯候选池，避免最终实体阈值硬截关系 |

另已直接查看PDF第6页Fig.2，避免纯文本抽取漏掉图片中的提示词。共指阶段要求严格原文跨度、全局位置、仅指定患者、无歧义才收录，每患者每passage最多25处；不能把亲属、对照、泛指群体默认归给目标。归属阶段还排除否定、假设、待排除与仅家族史，不能凭疾病常识推断。实际实现若采用occurrence编号/P001等更稳格式，须明确是本项目改写而非官方原样复现。渲染位置与SHA见来源清单。

冠军公开[权重卡](https://huggingface.co/rigidk/CHIP2026-PatientPheX-Agriv) revision `7a98ca95683eb920b09cc9141cdf512dc5f1fa41` 声明所有自训标签来自官方80篇。可研究结构（BioLinkBERT-large、Qwen3-8B LoRA r16、单/多患者LGBM），但这些已训练权重包括本仓库dev16，不能直接用于本轮本地验证；五折成员未公开核验，也不能推定兼容。本轮只读取模型卡/元数据，没有下载该自训权重。

17:29 UTC重查本项目已失败的上下文CE与上述模型卡：旧实现只对固定span-ID做二分类（局部前后180字符、max160），随后直接阈值或固定权重与树混分，未获得收益；冠军卡描述的是折外CE分数作为二阶段树的训练特征，且有不同的候选来源。因此旧负结果只关闭已试配置，不证明所有上下文融合无效。若实施新控制，须补相应训练范围内的折外特征及候选来源，不能用训练内CE分数冒充折外，也不能为复现借用含原dev的成品权重；18:31 UTC已批准一次严格OOF控制的CPU准备，需三组train16生成六个池以保持二级候选来源隔离，再训练三组train32 CE；GPU预算12分钟。原84列/.4树与加一列CE控制不改其他参数，18:38:18启动并在约290秒内完成全部GPU阶段，两树终点 .714808→.701372、四项均降，停止本配置。旧实验入口为[本体E-007](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/experiments/E-20261023-007.md)第七阶段。

[PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger)提供从本体构造远程监督样本的流程，无需额外人工表型标注。现成v1.2的HPO2023词表与比赛指定2026版本不同，且示例外部GSC开发集不能照搬；若采用，使用指定HPO与当前训练折重新构建，保留原文跨度。

已进一步审查 PhenoTagger v1.2 固定代码 `10f8ee3e` 与权重卡：代码MIT、权重Apache-2.0。梯度训练使用HPO远监督，官方示例另用GSC人工开发集选择checkpoint；代码也允许无dev固定轮次，但成品实际选模谱系未公开，不能断言每个权重都用过GSC或全为纯弱监督。旧softmax头为17,265个HPO加None，当前2026分支有2,036个ID不在旧头，不能仅替换词表。本轮从已有通用Biomed编码器与2026本体重新训练，多类encoder全量微调区别于此前冻结encoder二元MLP；先测真实耗时，不把小预算试验说成完整100epoch复现。完整源码接口/多父节点限制和来源SHA见外部 `research/phenotagger-v1.2-review/` 与来源清单。

官方参考文献中的 [AutoPCR](https://doi.org/10.1093/bioinformatics/btag304) 已读取全文与 [MIT代码说明](https://github.com/yctao7/AutoPCR/tree/c2b46a9c088c696533a43c5cb8e50c7500f055fe)：BioNER加名词短语/协调结构抽取，SapBERT置信度分流，LLM根据候选定义与同义词链接；另以HPO词面和检索难负例自监督微调linker。原方法的30B/80B LLM超过本赛上限，Stanza i2b2临床监督组件和GSC开发集选参也不能直接搬用。可借鉴坐标可追溯的协调短语分解与难负例构造；重组短语不能伪装成原文连续跨度。本轮仅研究方法，未使用其外部标注语料、结果或任务权重，尚未实现或测分；来源位于外部 `research/autopcr/`。

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)（Apache-2.0）作为可能的新基座做了独立可行性核查：全部775个权重键的形状推导与API总量一致，完整参数9,653,104,368，文本部分8,953,803,264；四个权重文件共19,306,310,880字节，LoRA参数仍须另计。稳定Transformers 5.20.2原生支持；普通非packed短推理有纯PyTorch回退，packed训练则需要对应FLA/conv支持。混合线性注意力要求重新选择LoRA层，不能直接复用旧8B训练器。先在自有隔离环境测短输入/单步成本，长上下文显存、吞吐和质量均未知；A800项目盘实际可用约9.45GiB，当前不能放下新权重。.100/.3有空间。源码、版本与磁盘证据见外部 `research/qwen35-feasibility/independent-review/`；这项审查没有运行模型。

## 本项目代码复查后的两项新控制

GLiNER虽输出嵌套跨度，旧装配会拒绝所有重叠。训练内nested32→16在既定Sap `.65/.92` 和否定筛选之后，被拒854项中773为精确重复，57为不同HPO的严格包含；其余为同ID边界变化、同边界异ID和交叉跨度。唯一固定实体对照保留全部旧实体/关系，只允许不同HPO的严格嵌套，新增26TP/31FP、文概念15TP/4FP，Score .653745→.657806。尚只证明该训练内控制小幅有益；交序列路线验证outerfold0，不改正在运行的全64原终点政策。源码/冻结计划/预测在外部 `research/strategy-review/nested-overlap-control/`，summary SHA `be75a6e20c973eab9b54b429120f0ee4ddfd54e5cb25d7d2a58e2fa3ce6d0373`。

旧患者关系聚合树没有LLM特征，固定投票/门控也不同于学习其互补信息。已确认严格OOF的204条提示只在独立answer字段含监督标签，尚无通用9B的这些响应。下一控制先用固定通用基座推理，再比较同候选、同超参数下有/无LLM选择、空分配和解析状态特征的两棵树；不能使用在全部train48上SFT后的模型回推这些训练文献，冒充OOF特征。已批准CPU准备及25分钟推理预算，18:25:14开始204条通用9B推理，实际735.19秒全部完成；原树 .653192→加4事实列 .649124，固定.5控制负，不继续扫本折。严格训练内校准尚需额外二层候选隔离，现有4127行不能直接再分折就称完全隔离。两项结构复查的短报告位于外部 `research/strategy-review/README.md`，SHA `a77887ffcd930c4d62846496db7006e02e8f6f2a37dc2c57ee4180feb74c48a2`；不把外部方法名称当实际收益。

GLiNER large-v2采用固定公开版本 `urchade/gliner_large-v2@9d0533ed97ade535f232b28dc673999c27c43159`，459,495,168参数、权重SHA `36bb03139034378b093de421ebc4a3052b7cfabbc089f4644d5be16d51aafc14`。按偏好先查ModelScope的两个准确路径未找到可核版本，再从HF固定源下载；不推断魔搭全站不存在。与medium同NuNER谱系和实际tokenizer，但上游head及recipe不同，称较大checkpoint替换，不作纯容量因果归因。实际strict load、1744训练窗collate及最长4×384单步独审通过，峰值9.875GiB；fresh train48固定10轮于18:44:30.701启动；训练阶段75分钟cap，前后推理各300秒另计，不从smoke权重继续。来源和计划由[序列路线](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/evidence/round2/gliner-large-feasibility.json)维护。

补查原GLiNER论文§2.2 Eq3及实际0.2.22上层调用：默认是未加权BCE（alpha=-1/gamma=0）；底层函数名含focal不代表实际启用focal。旧medium48训练末正/负logit梯度绝对量约630.53/626.82，未支持数量失衡导致正例塌缩，不能据此追加focal训练。训练48严格OOF漏项1364中908无重叠、411边界/嵌套、45同跨度ID错；其中4是gold alt ID与已预测canonical的口径差异。纯链接错误不是本池主要瓶颈，不追加大规模Sap微调；详细训练侧来源在两路线记录。

本体后续研究 [Dai/Adel COLING 2020 的 NER 数据增强](https://aclanthology.org/2020.coling-main.343/)：i2b2少量50句时完整mention替换F1 35.1→40.1，但500句仅70.2→70.8、完整数据87.8不变，不能将小样本收益外推。本项目只借鉴方法，将官方训练折中完整实体替换为同canonical HPO的name/EXACT词面；没有引入论文的人工临床标签。CPU已保留1150原窗并追加353窗，全部真实分词/跨度/NO标签校验通过；数据和更新数同时增加，属于配方初筛。独审通过后一次固定10轮短训练，实际唯一终点 .653745027→.653936525，只有mention净修正1项，doc和关系不变；极微正增益无实质改善，关闭配置，不迁移原dev。相邻段落上下文（FLERT）后续已进入固定配对控制，状态见下文和总台账。来源、边界与初始方案见[本体短提案](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/experiments/P-20261023-NER-AUGMENTATION.md)，执行计划以最新台账为准。

另发现 [GLiNER-BioMed large](https://huggingface.co/Ihor/gliner-biomed-large-v1.0/tree/f79771210ab2111a2081ef31620fdb7342f602e9) 可作为后续领域基座候选。[论文v2](https://arxiv.org/abs/2504.00676v2)声明105k自动医学标注预训练与19k通用后训练，后者含1878条WNUT/OntoNotes/MultiNERD人工通用NER；teacher有4个医学示范，其人工/生成来源未声明。发布资料支持这是预训练基座，未发现八个人工医学benchmark的微调权重混入证据，但不声称全链路无人为医学示范。公开zero-shot microF1 59.77相对large-v2 37.38只说明值得试验，不是本赛Score；充分监督下的提升也未显著。源码/论文/发布元数据只读独审通过（`32fcebae…`），实际权重SHA `7e3591fd…`、477,981,696参数及严格加载也已核验。唯一nested16零样本完整Score .652014251，低于监督medium32的 .653745027；纯跨度F1 .453801高于旧medium零样本 .277295，但不能当完整Score。固定监督32×10适配已完成，保留原显式BCE/优化器/1150窗，从公开基座fresh训练1660.16秒；完整内部Score .654424302，比medium32仅+.000679275，独审通过。下一项只改变邻近上下文，不据微增扩大模型规模。ModelScope准确模型/文件API均404、阳性对照200后可回退固定HF源；只说明已查准确路径。新配置内嵌encoder_config且词表128003，与旧loader有两处接口差异，应使用独立入口，不能更改正在运行的large训练。完整来源见[清单](round2-research-sources.json)。

## 其他同赛开源记录

`irryxeek/patientphex-chip2026` 的多来源跨度、SapBERT不同概念margin、上下文关系融合及正负报告值得研究；`yuanpinz/chip2026-PatientPheX` 给出逐occurrence、逐患者与联合患者判断、结构范围筛选；`AnXin-HITSZ/...` 与 `xi-ix/HPO` 有词典未覆盖片段的语义候选做法。其自报分数未独立复现，训练划分和评测不同，不能直接排名或沿用阈值。许可未明确的实现只参考思想；没有引入他们的标注或预测。

`wangzgui/CHIP2026` 虽为MIT且README描述完整架构，但固定commit `dfec694b2cb85e946148de90612467b8e78ee0c7` 只有README/LICENSE/gitignore，不是现成可跑实现。方法名称、公开下载和环境安装都不替代本项目实际消融结果。

## 原论文表格：可取得，但首个冻结政策没有增益

复读官方README §参赛规则3：允许公开外部资源，禁止外部人工标注表型数据集。据此，本轮可研究原PMC论文的未标注表格作为Task2外部证据，保留出处并在技术报告披露。Task1仍只输出给定全文内可核对的坐标，不把外部表格位置拼入输入。

按冻结train48中出现TABLE的文献顺序选前两篇，未依照gold漏例选取：[PMC11203018](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11203018/fullTextXML)、[PMC11235872](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11235872/fullTextXML)。均HTTP200、CC-BY，XML完整SHA见来源清单；前者第二表区分历史病例与“This Case”列，后者含13个家系成员列及认知/精神症状等行，而给定TABLE输入仅有caption。当时只确认存在额外原始证据，尚未跑分或证明收益；后续冻结对照见下。后续应按明确患者称谓对应列，处理No/Normal/unknown、脚注和历史病例，不把整表分给所有患者。

随后独立审查固定fold0的全部7篇含TABLE输入，7篇原文全部取得，15张表共552,366字节XML；保存8处rowspan、17处colspan和7组脚注，9项解析/ID边界检查通过。完整临床编号规范化能对应7个临床表患者和9个酶/基因型表的Subject，后者不能直接视为表型证据；全程未读gold、未生成预测。来源、许可、SHA及结构化缓存位于仓库外 `research/pmc-tables-fold0/`，结果manifest SHA `afeacad97fb8cc8d95a35a08469514008a14c377bc8caf5db43f0a83d80f5e75`。先在train48定义并冻结映射规则，再作fold0对照。

后续CRF路线按train48前两篇定义并冻结严格患者身份、临床表标题、精确HPO/训练别名与明确肯定单元格规则；固定fold0对照中新增31/29条关系全部FP，未获收益。外部临床事实不等于本赛标注范围，见[H25](../../HANDOFF.MD#h25外部原论文的临床事实未必属于比赛标注范围)；后续优先给定文本，不继续围绕本折扫描表格阈值。


## 后续只验证输入中缺失的上下文（19:50 UTC）

患者关系旧CE的训练侧CPU复现表明：同4127个OOF候选中，top3选窗/384截断没有使任何样本丢光已有目标患者marker；969个canonical正例中579本来就在全部局部窗无目标marker。因此“只放宽截断”缺乏支持。固定12篇正负配对阅读发现病例叙述延续、群体指代、缩写和时间极性，但小样本不代表整体比例，也不生成特例规则。下一控制只给旧实际输入追加可定位的近邻患者身份句，最近身份可为目标或其他患者，保留竞争身份/否定/背景、原active tokens与末尾SEP；同OOF候选、同CE两臂都重训，不复用gold候选旧模型冒充对照。CPU候选/标签/输入隔离、覆盖及实际单步已通过；双fresh正式对照已授权，实际job/结果以总台账为准。

[ATLOP](https://arxiv.org/abs/2010.11304)与[EIDER](https://aclanthology.org/2022.findings-acl.23/)只提供可定位证据连接的机制参考，未使用其代码/模型/DocRED标签。ATLOP的threshold-class在单二元关系下等价于logit差的BCE，不单独包装成新校准。严格OOF CE单列融合实际判断span+HPO正确性，已经失败；旧患者CE确有header/marker/多窗，故新变量只称身份上下文。969与旧SFT967正例仅差两个HPO alt-ID规范化，4127候选键完全一致；两新臂采用同一canonical政策，历史SFT不重写。详细设计、固定训练例和源码/论文SHA在来源清单，研究报告使用启动时的旧最高分，当前最高以总台账 .715760 为准。

本体路线另作FLERT式相邻上下文CPU检查：1150目标窗中1121能加context，新增token均81.21；449跨section、850跨passage type，不能把可加上下文当作有效证据。原2236正/87NO标签、1125538有效跨度和offset不变；仅context参与encoder attention，word映射与解码限于原target。固定batch padding代理1.3158、attention平方代理1.5521不是GPU成本实测。专用collator/低层target解码与独立训练/推理CPU入口已通过；选择同BioMed基座、原.3环境准备固定10轮控制，仍须实际全CPU/单步成本检查。


## 生成式实体抽取：暂不重做9B训练

只读复核确认8B Task1 SFT与换折复验已实际完成；原305窗覆盖train48全文，不是只抽305句。生成quote后广播所有出现有表示缺陷，严格quote+出现序号可逆，但train32当前窗口约束外仅39/2274跨度（1.72%），单靠该变化缺乏大幅增分依据。旧8B固定Sap后fold0有小幅收益，fold1未复现；当前关闭9B重训提议，保留详细报告中的复开条件。

[UniversalNER](https://arxiv.org/abs/2308.03279)的JSON实体问答、保留空目标可借鉴；其官方评测会规范化词面并去重，不符合本赛逐出现原始坐标契约。只研究论文/源码，没有导入其训练集、权重或外部人工表型标签。完整报告与来源manifest在 `research/generative-ner-feasibility/`，SHA见[来源清单](round2-research-sources.json)；该研究未训练、未读原dev或外层gold。


## 显式focal配方：固定单变量配对准备

已查历史8个GLiNER实际运行manifest均为显式BCE；BioMed固定发布配置另有alpha=.75、gamma=2、reduction=sum，而库Trainer的默认-1/0会覆盖这些值，不能据配置字段推断历史训练已经使用focal。下一控制借用该alpha/gamma，仍沿用本项目valid-span mean，并固定原1150窗/10轮/2880步，因此不宣称完整复现上游sum训练。训练监督极不平衡，但既有medium/BioMed训练探针召回较高；这是类别与难例重加权假设，不能称为修复模型塌缩。

纯函数CPU的解析梯度、显式上游helper对照、极值、padding、gamma0退化、全负batch与gradcheck通过。A800空卡但磁盘不足，选择.100 A30同runtime fresh BCE/focal配对，避免把环境差异归因于loss。实际运行与终点见[序列台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/WORKLOG.md)；准备证据为该分支 `979dd4e` 的 `evidence/round2/focal-loss-proposal.json`，此处尚不声称质量增益。


## 公开PhenoTagger直接复用的判断更新（20:40 UTC）

旧审查确认上游示例以外部GSC人工开发集选checkpoint，但公开单个权重没有完整选模谱系；这项不确定性不能自动升级成比赛规则禁止。官方综述§4.2记载头部方案使用PhenoTagger，Agriv固定模型卡明确列出公开 `lingbionlp/PhenoTagger_v1.2`；结合规则允许公开预训练模型/工具，本轮决定先测试该公开模型，而非只接受从本体重训。此为基于现有资料的实施判断，不冒称主办方已单独确认本次用法。仍不导入外部GSC人工标签，也不使用含本地dev的Agriv自训权重。

固定revision `02ce3f529820dccca46bae168c1933ba25253125` 的单个 `pubmedbert_PT_v1.2.h5` 为491,312,288字节；准确ModelScope接口404、阳性对照200后回退固定HF。只在A100自有TF2.12/Transformers4.30.1环境做CPU原生加载，保留17266列顺序、按当前HPO canonical/alt规范映射，不用新词表替换旧head。旧head仍缺部分2026类；实际SHA、逐变量恢复与合成词检查记录在外部 `research/public-phenotagger-reuse/`，尚无质量结论。采用from_config后完整h5严格恢复可避免重复下载基座，必须实际验证全部权重覆盖，不能仅以加载函数未报错为准。

实际CPU原生加载已通过：201变量、122,759,794参数先置NaN，再由Keras H5按拓扑严格恢复，每个变量与原文件逐元素相等，完整检查16.459秒；未转换PyTorch、未下载第二份基座。原17266列映射后覆盖当前17090个概念（178旧alt列规范化、3列无效丢弃），仍缺2030当前类；只在原argmax后映射，不提前删列重算概率。首次因Transformers4.30.1未发现tensorflow-cpu失败，使用上游FORCE_TF_AVAILABLE/USE_TF开关后通过，未修改共享库。四条合成词只说明接口与标签列正确，实体控制和实际吞吐仍在准备。冻结短报告及来源见[清单](round2-research-sources.json)。

另按固定train64统计类覆盖（不读held/dev）：4765正mention中100为NIL/非HPO，余4665个linked mention有4626个（99.16%）在原head规范后至少一个ID可覆盖；1406种canonical类中覆盖1378。原字面ID覆盖4617，公共v2并不统一规范所有alt ID，不能把语义覆盖当实际匹配率。旧head缺2030本体类目前未构成训练侧的大频次缺口，但不代表候选抽取、阈值或任务分数；来源见清单。原生候选的跨句上限问题和无标签修复证据已写[H36](../../HANDOFF.MD#h36第三方候选生成器会因跨句可变状态静默漏项)。CPU实测256候选26.972秒，65,406种词序列估约115分钟，转为自有TF2.12/cuDNN8 GPU环境准备，未把估计当实际全量耗时。


公开Pheno的其他head只读可行性（21:10 UTC）：Bioformer-8L是原cased-v1.0的官方改名，配置推算含同17,266类head约51,380,594参数，公开h5为205,710,168字节；BioBERT约121,587,826参数、486,624,416字节。均尚未下载权重/载入/测质量，不把文件小等同吞吐或互补收益。BioBERT配置和vocab已从魔搭固定revision下载，与原HF版本逐字节相同；Bioformer准确魔搭路径404，固定HF配置/卡可用。上游即使设置cased tokenizer，load_data仍逐word.lower()，应保持真实预训练表示；候选词串可重用，跨模型forward不可重用。先结案当前PubMedBERT跨折与选择器控制，再决定是否做较轻Bioformer的互补来源实验；完整静态报告及快照在外部 `research/public-phenotagger-other-heads/`，report SHA `d0d09fbe…`。

Bioformer原生CPU随后已实测通过：固定公开h5全SHA，137变量逐一NaN覆盖后原生严格恢复并与H5数组完全相同，实际51,380,594参数/17,266原列序；四合成词接口无任务质量含义。原生阶段9.323秒不含导入，合成forward0.380秒。root增量源码/实际证据复核 `c481da2a…`，沿用既有TF环境和HPO映射；计划在原dev的PubMedBERT缓存完成后，再做独立train64无标签Bioformer候选缓存，不把静态可用性当互补收益。
