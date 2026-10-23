# SFT 后继的只读判断

当前结论：不因固定12训练F1 .883721或残留15错误追加epoch。若还需一个低成本可否证机制，已有adapter开启thinking比重训更省准备成本，但其收益和生成长度未测；这里只核事实，没有启动新GPU或训练，也未读新dev/outer gold。

## 旧48 SFT能否继续

A800上epoch002的 `training_state.pt` 仍在，346,667,053B、SHA `d2dcc780…`，本次CPU实际重哈希并加载：496个Adam状态均step52，43,278,336参数的两moment齐且有限，Python/NumPy/Torch/CUDA RNG齐。**optimizer lr=0，scheduler last_epoch=52**。原代码固定2epoch/52步，拒绝已完成目录，循环亦不产生第3轮。因此保存了完整可恢复状态，但终点不是尚未结束的暂停点；追加需要新的明确学习率日程/指纹/独立输出，不能称原计划无缝续训，也不能只恢复adapter冒充完整状态续训。原数值resume等价未证明。

同fold0、947实体的原终点Score **.695443948**，低rules **.711353522**；2epoch实际1,855.10秒，第二轮911.54秒（26次更新）。多一轮约15分钟只是旧A800耗时代理，不能据此承诺新训练成本或收益。

## 只保留两个目标候选

| 候选 | 能减少什么，保留什么 | 已做范围和成本边界 |
| --- | --- | --- |
| 单patient-concept的yes/no词目标（monoT5式） | 无CID复制，每pair显式正/负；一概念可对多个患者为yes，保留多患者集合 | 机制研究已做、9B未跑；旧Biomed患者CE及身份控制也已做且负，不能把换9B当目标新意。原train48独立pair前缀4.145M tokens，为现有204prompt的6.39倍，少解码不保证便宜；不推荐先重训此项。 |
| 固定候选顺序的患者集合数组 | CID仍在input，response只给按该顺序的患者列表，无输出CID复制；空列表和多患者全部可表达 | 原全患者自由HPO控制负，但与此固定候选目标不同。train64既有265行转换后所有关系集合roundtrip exact，含169个多患者候选；suffix从15410降到6498。只是重序列化，负关系仍由集合排除表达，并未证明更强监督。 |

同一多标签集合也可编码成按候选×患者顺序的0/1矩阵，本次CPU实际计11451 suffix tokens，5328个pair都有显式标签且关系exact；这是第二候选的另一编码，不增列第三条路线。它依赖固定轴顺序，autoregressive CE并非独立BCE/校准后概率。集合/矩阵分别仅约节省原总训练token的1.03%/.46%，新prompt指令尚未构建计数，更不能据输出缩短推断训练显著变快或分数提升。当前不改loss、不建立新训练。成熟yes/no方法和固定来源沿用[已有机制研究](continuous-relation-confidence.md)，不重复调研。

## Thinking 是未测的短推断变量

已保存的12份实际9B推断manifest均为thinking=false，含base/48SFT/full64SFT、全文和本次训练拟合；训练模板也固定false。已冻结的[官方9B模型卡](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)明确默认thinking，通过 `enable_thinking=False` 关掉，且不支持用 `/think` 软指令替代。真实模板true会把assistant前缀从闭合空think切成未闭的 `<think>\n`；实际tokenizer中两个think标记均非special，不会被现有decode的skip_special_tokens抹去。模型卡支持公开base接口，**没有承诺仅JSON训练的本adapter仍有有用思考能力**。

原parser能剥离最后一个 `</think>` 后的final JSON；有思考文本的raw本身不再是严格JSON，必须分开记录原raw和严格final。CPU反例还显示：opening已在prompt而suffix未闭think时，宽松parser可能从思考段抓取JSON。若后续做thinking控制，须先要求closing存在，再对final做原CID/患者集合校验；未闭合/触顶按失败记录，不能拿思考中的JSON作为答案。原非thinking结果不改。

原max_new_tokens1600包含思考和答案，开启thinking可能耗尽预算；模型卡推荐sampling不等于已批准本任务的greedy控制。最小候选应只改thinking、沿用预先固定输入/adapter/greedy/1600与硬时限，记录闭合、final格式、触顶和实际耗时；仅训练样本改善仍不能当泛化收益。**当前未冻结新的GPU计划、未运行。**

[实际CPU状态、统计和来源](../evidence/round2/qwen35-successor-readonly-review.json)。

## 后继实际决定与严格融合库存

上述thinking唯一短控制已获root授权并执行，随后按实际格式失败提前停止：第一条72.048秒、1600token、真实hit_token_limit=true且未闭think，无严格final；只完成1/12，未解析训练targets或运行fit，launcher仅核targets既有SHA。20:50:53Z停自有作业，20:51:01Z fresh释放A800。独立增量审67b49e51通过；不延长预算、不重试、不据partial报训练或泛化分数。[终点](../evidence/round2/qwen35-thinking12-result.json)。

另按root要求核实：之前204条四事实树使用**公开base、adapter=None**，不是折外SFT。因此其负结果不能直接否定SFT+LR学习型融合。当前9B只有训练48与64的epoch002；旧48见过全部inner1/2/3，full64还见过outer0，都不能回推自身训练行冒充OOF。将full64数据切成48，候选生成器仍可能见过outer0；将旧48 SFT数据切成32，候选生成器仍见过将被留出的inner16，这两种捷径均不满足隔离。

严格outer0内，三个已有train32→held16候选/无答案prompt可复用（80/62/62，共204条、4127pair）。六个16→其他16的语义池已存在并核train/input IDs与来源SHA，CRF+Sap仅2→3和3→2已有完整装配；其余四方向需补CRF预测/装配，已有generator2/3 CRF可供只读复用，generator1需CPU fit。**没有现成的合格train32 SFT checkpoint，完整三折OOF需要3个fresh模型。**

旧48两轮实际1855秒；简单按文献数估每个32模型约1237秒、三份约61.8分钟，仅为未构建新数据前的代理，不是实测或授权预算；204条旧base推断759秒也不是新SFT32推断实测。最小可行单meta-fold可只训一个32（F2+F3），其两方向候选已齐，预测F1的80提示/1557pair，仅在F1固定拟合一个低复杂度融合，然后复用SFT48→outer0预测。此法只有16篇meta训练，不能称全48 OOF或独立meta验证；任何LR概率也要由排除相应innerfold及outer0的模型生成，不能仅凭缓存features叫它OOF。元模型/阈值须在outergold前固定，外层仅用于检验。当前**只读库存，不构建新训练、不启动GPU**。[库存/来源](../evidence/round2/qwen35-sft-fusion-reuse-review.json)。
