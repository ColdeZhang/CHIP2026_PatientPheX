# GLiNER2.5 Task1 监督适配准备

当前仅CPU准备：不加载神经权重、不forward/backward/fit、不读取新held gold、不评分、不下载。E001两单臂负、root共识F1正/F0负的关闭决定保持；本提案是另一个学习变量，不能把原负改成正。

**判断：有区别于旧路线的原生监督机制可作一次有界候选控制，但目前没有相对强池的独有正确供给证据。** 在训练质量前，不能把新模型、可训练接口或候选数量视为收益；不支持直接开多seed/四折大实验。

## 已试机制与差别

| 已有路线 | 实际做过什么 | 本提案的区别与限度 |
|---|---|---|
| GLiNER medium-v2/large-v2/BioMed | markerV0、固定宽度span head；medium/BioMed train32真实10epoch、large train48真实10epoch。绕过旧gliner0.2.22 loss错位后用显式valid-span mean BCE。medium同卡BCE/focal和解析cut也已完整跑过且未胜匹配BCE。 | 不是“首次试GLiNER监督”。2.5已有预训练boundary proposer、共享候选池、reranker、count/abstention，原生多项loss与上述BCE不同，不能称纯容量或同配方对照。 |
| 原强池追加旧GLiNER | F0/F1净23项只4TP/19FP，固定追加负关闭。 | 历史差说明新增源也可能只增加FP；2.5新权重的实际独有正确供给仍未知。 |
| BiomedBERT GP | 医学编码器＋新RoPE64整体跨度头，train32真实8epoch。固定追加及最终四折特征控制均负。 | 2.5的边界与候选重排头已预训练，不是随机GP头；encoder、预训练来源、loss、分窗、规范化都不同，不能孤立归因为架构。 |
| 本线GLiNER2.5联合关系 | 193581591参数的原生boundary+relation公开权重，无本任务微调；患者/表型联合schema、beam32，然后train32词典/publicSap。 | Task1只查询phenotype，不受患者提及/同窗关系匹配约束；训练仅使用真实实体offset，不制造mention-pair关系gold。 |
| 旧“GLiNER2-large未采用” | 当时容量替换研究无法确认同训练配方，未找到本2.5 boundary Task1实际训练。 | 这条旧说明不是本机制已被否定，也不代表本机制有效。 |

准确旧终点与上游source SHA在 sources.json；不从过时NOTES中的“尚未focal”推断最终未跑。公开模型HF 54785d51、代码 ca4f6823、权重7274094d，全部参数/头库存沿已核E001；完整公开预训练样本谱系仍未知。Task1不用relation/record/classifier任务头，但193581591库存不减去这些权重。

## 一个固定原生配方

recipe.json是唯一建议，不是训练启动授权。复用原公开权重，原生ExtractorTrainer，10epoch最终唯一终点、batch2、累积1、encoder LR1e-5/task LR5e-4、AdamW .01、linear/warmup .1、clip1、BF16、seed20261024。不提供eval_data，不早停/选best；原生loss、proposal预算192、gold注入1→.25（前15%保持）与count/abstention保持公开配置。这里没有复制旧span BCE补丁；新原生loss/梯度/实际显存仍须未来真实短probe确认，CPU目标合同不代替它。

train32仅F2+F3（SHA38c353），F1纯输入962b76a3；旧prepared数据/规范化谱系明确不含F0/F1。全报告按原passage顺序拼接、显式原offset映射，native384词/64重叠完整320窗，batch2整除所以原生drop_last=True实际不丢尾样本；每epoch160更新，10epoch1600更新。max_len是原生**词token**上限，不是词片上限；原生Trainer回落model4096，另核全部真实wordpiece IDs及无截断。

只一个phenotype正类。官方非NO实体（含compound和NIL）按精确offset映射为SurfaceSpan，重复坐标合一；HPO ID不进入神经目标。NO不另立分类，原文负上下文保留、NO跨度审计并排除正目标。不能声称所有NO都得到pair负loss：稀疏pair loss只作用实际候选，未提出的NO没有被强制注入。密集start/end/inside目标与padding/query masks按真实原生张量核对。

原生词边界不能精确表达38条正annotation（如连字符复合词内部、GCD1的子串、ataxias的子串）；保留完整原坐标清单，不扩边界、不借gold修改推断tokenizer。它们没有对应可预测的native span，不把覆盖率声称100%。所有能表达的正目标均需被完整窗口覆盖；dedupe_contained_entity_spans=False防止上游默认删除真实嵌套，max_gold_per_query64，容量超限和missing均raise，不skip/fallback/truncate。

## F1供给与比较

绑定root指定的真正强Task1：lightgbm.nested.jsonl SHA8cb61ba9，1313实体，已有Score .728697536469（metrics4099a624）。关系共识的1189实体壳2a65056f较弱，不作为新Task1强收益基线；GP单折正但四折负的后继也不换入。

未来若获执行：公开base和唯一epoch10都用相同Task1 schema、extract_entities threshold .5、overlap_policy=allow、原384/64全F1窗口。保存全部声明native候选、概率、offset及拒绝原因，重叠窗口同原跨度取最高分；原生cap仍限召回，不能称穷举。无关系任务、无额外患者约束。

规范化复用strict31d380f5、原train32词典和public41099 Sap frozen policy b59eb3f8（cos≥.90、distinct-concept margin≥.03、max32），不重编码bank、不调cut。主臂保持强base及非entity对象，仅追加合法HPO且不交叉/不冲突的disjoint或不同概念proper-nested新实体；按模型分数、长度、offset固定序。NIL原始诊断保留，主追加不使用NIL。先共冻完整paired base/adapted供给，再按root指定一次质量预算比较；E001双类型联合解码不是匹配的Task1零样本控制。

本准备没有独有TP/FP数据。1600步的实际训练耗时/峰值尚未测，旧GLiNER的BCE耗时与本线68–80秒推断不能直接拿来承诺训练成本。下一合理动作是未来一次可丢弃的最长2行/全负及混合batch原生loss、梯度、proposal recall工程检查，之后才决定完整训练预算。

## 过程和证据

复用已有5.522秒CPU小API合同及train32结构census，未重做该旧合同。新增只核完整320窗真实目标/输入：首次run_command返回UNKNOWN，恢复证实原进程消失/输出空，rc与原因未知；不当作模型失败。managed actual-v2保留真实审计异常：检查器误读空的model_texts字段。上游train collator实际把normalize后的record.text放original_texts，因此仅修诊断字段，在独立actual-v3恢复；旧异常与输入/标签保持。新运行始终CUDA隐藏、线程1、120+TERM5、0神经forward/fit。

已经完整构建的真实train32统计：2275正annotation、87NO；去重后2236可表达正跨度、87NO跨度，38词内正annotation单列；8严格嵌套对、2交叉对；320窗有2642正target实例、99NO窗口实例、74窗无正目标。覆盖和类型统计使用原train32，不读取F1答案。actual-v3另保留检查器误把原生惰性dense targets当已构建张量的异常；collator明确build_dense=False，真实loss用dense_targets_from_pairs构建，因此只修检查器使用同一纯tensor helper并独立核每个字符目标。字段修复不是改loss或训练数据；随后实际导入位置也核正为gliner2.models.boundary.targets_device，不能把仅调用它的processing.targets模块当作公共导出。该ImportError发生在数据/模型计算前，独立actual-v4失败保留，最终actual-v5只修import并收紧到60+TERM5。

最终actual-v5完整320窗合同 **8.352759秒通过**：2642个正目标实例全部精确、max49 gold<64、271窗原生补句末点、max698真实词片；所有mention mask、通过原生纯helper构建的start/end/inside、text/query/padding mask及独立tokenizer.json ID对照都通过。0模型参数/forward/backward/优化/评分。原生稀疏候选loss的真实negative覆盖仍未测，不能用这份dense目标合同替代真实native loss/FB/Adam。

Train.native为833524B/SHAf23981c2，全部目标及实际IDs审计1026390B/SHA216b7d47；精确完整SHA和服务器locator见actual-v5/complete.json、collection.json。此前错误都是新审计代码不贴合实际接口，不能归为上游模型失败。

原生ExtractorTrainer本身不自动holdout：constructor与train均显式eval_data=None，不调用独立TrainingDataset.split；max_train_samples=-1/max_steps=-1、early_stopping=False/save_best=False，epoch策略在无evaldataset时只保存，不计算验证指标。最后final来自第10epoch。DataLoaderFactory默认validate=True会sanitize甚至过滤，因此另做原生loader的完整320行多重集合一致性与10轮LengthGroupedSampler/BatchSampler(batch2,drop_last=True)覆盖检查；未来实际trainer也必须核320样本/160batch、指纹和最终1600updates，不以配置名替代真实运行。

原生loader实际补核 **0.037998秒通过**：validate=True 后320行规范JSON多重集合不变，2642目标/74空正窗；10轮原生sampler每轮160批、320唯一行，无遗漏。CPU TrainingConfig显式fp32仅为配置构造，未来GPU BF16仍待实际验证；最初补核shell代码换行SyntaxError和第二次CPU误请求BF16的RuntimeError保留，均非模型前向/训练失败。

另发现原生train对OOM无条件continue，即使strict_training=True；未来完整路线必须先原生记录failed batch再立即raise，不能静默删样本。此为失败处理护栏，不改loss。

root已在准备结案后另行授权新explore/gliner25-task1路线的一次180+30秒GPU压力probe；正式训练未授权。root最新端点取代上文初提额外筛选：主strong8cb61ba9并全部规范化接受实体，诊断new-only保原关系，二完整端点共冻后各一次v2；无本准备评分。后继独立记录，不改旧E001/E002。
