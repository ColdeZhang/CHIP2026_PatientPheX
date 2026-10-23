# CRF 序列标注研究笔记

只保存可复用的方法来源、工程经验和待验证假设；试验结果以[实验过程](experiments/E-20261023-009.md)为准。已确认跨方案坑回报主线程更新根 HANDOFF.MD。

首日结果见全局实验索引；第二轮不能只优化关系选择，首日实体固定时关系oracle总分上限 .7803；增强实体后上限 .805345，距 .8只剩很窄余量，仍需要同时改善覆盖和患者归属。这是gold-aware规划诊断，不是真实候选成绩。

- 内部折的患者构成/长篇多患者文献权重与原dev不同；两次患者规则/上下文增强均有内折收益但dev下降。必须同时保存分歧，不能拿一次提升代替完整证据。
- 预训练底座是训练缓存的一部分，除了训练数据、源码和超参数，也要将权重/config/tokenizer的SHA写入指纹；仅记录模型路径不够。
- 显式cohort归属只在数量与候选患者数相容时触发，并限制临床章节或明确our指代；泛称patients不能直接向所有列出的患者广播。

- 哈希映射不能以 basename 作为唯一键：不同子模块都叫run.py时会静默覆盖。角色名或唯一相对路径为key，并用同名不同内容的测试验证两个依赖都保留；已回报主线程。
- 神经HPO候选重排训练时，至少扣除本篇独有训练别名；否则训练样本正例通过精确别名进入候选，会高估泛化并学不到真正的语义判别。

- 同训练语料/相似NER结构的错误高度相关。新增跨度三路一致也只有30/112完全正确，不能因多数票就默认有足够精度；应先查相对现有实体池的边际正确率。
- Qwen候选归属提示虽明确排除泛化/阴性，zero-shot仍在固定fold0产生大量FP。先验证、记录，不以语言模型“更强”的直觉替代实测；小thinking对照应保持输入与预算可比。

- O 编号来源是官方“文章编号”schema，P 编号是标注顺序。完整编号匹配不能退化成裸数字搜索；当前家系分隔符等价化版本只是探索假设，四折不稳定，不升级默认。公开表格沿用更保守的唯一全串身份匹配。
- Qwen thinking 模式在固定输出预算下可能未输出最终答案。必须检查闭合think与生成上限；拒绝把推理中提到的示例JSON当最终决策，单独报告有效率。

- 公开原文表格可以匹配患者、含临床事实，仍不保证属于比赛的关系标注范围。首个严格表格增量31条全部为本地金标FP，不能用“事实正确”替代评分验证，也不能按金标漏/错例反向筛选表格。
- 二阶段模型的训练候选必须对该训练文献也是OOF：outer train48需再划3份train32→heldout16；不可拿train48拟合过自身的预测给SFT当候选特征。gold仅进入独立answer，不补候选、不改提示证据。
- SFT必须用真实聊天模板逐样本验证prefix token完全一致与answer-only loss，禁止静默截断。最长样本单步smoke和正式训练目录隔离，不把smoke权重续训成正式实验。

- Gold患者-HPO关系通常没有对应证据span指针。用该文任一同HPO正实体计算患者距离，只能给出乐观锚点可见性，不能当真实归属证据或共指金标。原文±500全局距离也不同于按passage裁剪的真实prompt窗口，应分别统计。
- 扩充患者编号不等于完整共指；全局patient anchors与候选局部marker也不同。未来共指应只读原文、输出精确可回定位字符串/患者ID，并固定同候选比较，避免把newspan、实体与关系同时换掉失去解释。

- 患者共指即使JSON可解析、text精确存在，也可能错在定位或身份：短he裸子串次数会落在the等词内；模型还可能把表型词当人称。应以完整词边界和唯一局部原文context定位，并单独审查患者身份，不能只靠合法坐标验收。
- 新共指短语不能隐式升级成全文alias：保留原始给定alias处理，generated mentions只使用核验过的真实坐标；旧alias广播另作可归因控制，不修改共享规则。

- LLM响应缓存复用必须比对完整输入row和实际模型/adapter/解码器/运行时身份；同row_id或prompt条数相同不足以证明可复用。变更行重推、未变行复用时逐行记录原响应文件SHA。

## GLiNER 结构方向：准确版本比家族名称重要

三nested训练诊断将下一步放在跨度覆盖：1364实体FN仅45同跨度缺ID；371/501关系FN来自候选外。411“有重叠”只是操作性分类，不能全部算可修边界；四context上限仅损失3个marker，不足以优先再做同池MIL。诊断已独立复算，完整范围/边界见training-loss-diagnosis与review。

优先研究 [GLiNER medium-v2](https://huggingface.co/urchade/gliner_medium-v2/tree/f7da09ddffab92deb4f2a815662d88b7e0d909d5)：官方卡明确NuNER数据，模型/代码Apache2.0，数据MIT；[NuNER论文](https://arxiv.org/abs/2402.15343)明确C4原文→GPT3.5自动注释，通用语义类别与span匹配不同于现有任务BIO。v2.1的pile-mistral卡仅license，不能因同GLiNER家族就继承v2谱系；v1含非商用许可，不作为首选。独立来源审查确认公开谱系，但全checkpoint训练/选择清单未公开，保留限制。没有使用含本赛train80的成品模型。

[冻结最小提案](experiments/P-20261023-GLINER.md)只先在train48按输入患者数/字符中位选择三篇，固定phenotype/.50、完整词与实际token预算、原passage坐标；只资源/定位pilot。后续若获分配才做同折Sap过滤与不重叠增量，固定阈值/规则比较，避免把多个改动同时包装为增益。可选2epoch自训已按后续冻结方案完成，结果见下文。

训练FN独立分层补充：908无重叠FN主要是短而未在相应train32见过的词面（中位2词，836≤5词，仅39词面已见）；仅6项>GLiNER的12词宽，411overlap中仅4项>12。大章节漏率无明显全章节硬截，CRF chunk跨界仅1项。因此GLiNER假设是通用类别条件span预训练的互补覆盖，不是更长跨度上限。该模型195.176M、3篇101窗3.058秒/0.830GiB已证明成本可控；未评分，不宣称候选优质。详见冻结提案和 `validation/nested-loss-review/miss-strata.json`。

GLiNER训练接口坑（训练前已拦住）：gliner0.2.22的SpanModel.loss位置调用与BaseModel._loss新增prob_margin参数错位，negatives=1被绑定为label_smoothing=1，原0/1目标全变.5；正logit2的正例梯度因此反向。实际A800库base.py SHA b3ef985596410c07a447f18f5e199339b97ca4f603441a9ff36f9242d949e5bd、独立数值probe SHA de238b865de637eea05dccaafb6458d0be0df1bdb9bed5cb15a3934a7fc9e1d1。不要把库版本/成功退出当loss语义正确；自有trainer显式从logits算BCEWithLogits+validmask，正负/全负/尾batch/无效span梯度检查通过，不改第三方环境。此前所有GLiNER推断未走loss，结果仍有效。


GLiNER两epoch监督控制表明通用span预训练可学本题边界：nested16纯span F1 .2773→.6600，预定融合后相对CRF .6910→.7006。但边际98TP/147FP仅40%精度，不能把单模型提升当最终得分提升，后续先冻结链接过滤及更长训练对照，不能在同inner16扫阈值。epoch2训练probe recall仅.705，有继续固定预算的依据；该inner16已用于路线开发，未来不得称新盲测。原2ep scheduler已走完，真实optimizer/RNG存在也不使追加8ep等同从头总10ep；若改总预算应明确fresh或重启schedule政策。

固定fresh10的训练probe R提升至.926、留出span R .696，监督能继续学习，但更多跨度带来FP；预定union只从2ep .700614到10ep .702135。下一步优先已有HPO语义链接/表型检测对新增跨度的固定过滤，验证“原文跨度→正确概念→患者”全链价值，不继续用span分数替代比赛Score。总10与2ep从头schedule不同，源/设备一致，不能称等价追加8轮。
### 2026-10-23：监督span作为补充源的外层验证

固定10ep GLiNER并不胜过已有CRF+Sap实体池（outerfold0纯span .681736 vs .750596），但非重叠86TP/122FP说明存在互补。沿训练内已冻结的Sap p≥.65/cos≥.92与原否定规则筛选后，只加26实体（16TP/10FP），完整Score .711354→.721504、四分项同向；新增关联7TP/4FP。应保留原实体池，把新span当受控补充，不用纯模型F1否定增量来源，也不把训练probe .975召回当泛化结果。没有改阈值、类别或别名。

根线程已在结果前约定原dev低成本迁移：仍用train48 span模型（48⊂64、dev不在训练），HPO/Sap/CRF/LR取既有train64资源；基准必须是已独审 .700635947 LR版。两臂分别保留旧关系的实体增量、相同LR处理完整实体池，区别于本次fold0较强rules。当前先冻结迁移协议，不能把span训练48与链接训练64的不同来源藏在同一train SHA；结果另记，暂不再训练64或扫描组合。本轮实际来源/预测/保留检查点见 `evidence/round2/gliner-outer0-results.json`。
原dev迁移已得到同向小增益：同train48模型+train64 Sap/LR，32新增22TP/10FP，entity-only .706112/full-LR .711325，相比原.700636四项均升。该观察支持保留固定过滤的增量候选；不证明把全部64训练满会继续增加，也不应把GLiNER原始1103候选整池加入。下次优先由根冻结与本体新候选的合并政策，再统一比较；不要在原dev反复选择候选顺序。

- 固定跨来源追加也可能总体微升而实体mention下降：Pheno追加12只有4TP/8FP，同LR新增关系2TP/5FP。保留entity-only小收益.712586但停止扫描；全64规模对照仍隔离于这项组合。硬件/torch版本换机时复跑原48模型同输入，不能以复制权重SHA相同推断输出必同。

- GLiNER支持nested span但旧装配一概拒overlap，会无意把表达能力压回平铺实体。训练侧固定异概念proper containment在nested/outer/原dev总分均温和正，然而原dev新增35只10TP、mention下降；合法嵌套能力与值得全收不同。保原池/全部关系的单变量实体控制能分离这种取舍，拒crossing/同界/共享ID，不按结果继续扫策略。

- 全64fresh10不能按惯例替换48模型：同固定过滤/LR主Score.711325→.705869，mention净+2TP3FP但关联-1TP+9FP。相同权重跨runtime推理byteexact排除了该控制的解码差异，不保证torch训练轨迹等价；保留版本差异并以实际目标指标取舍。nested64小增仍未追回，停止当前规模/epoch扫描。

### 较大 GLiNER checkpoint 的来源与变量边界

large-v2（9d0533ed）与medium-v2均明确NuNER/C4自动标注来源，许可Apache2；large-v2.1改为pile-mistral且卡缺完整谱系，GLiNER2-large精确训练配方映射不明，因此本轮不采用后二者。large-v2官方API参数459,495,168，encoder12层768→24层1024，head512→768；tokenizer_config和spm均与现有实际SHA相同，max_len384/max_width12/markerV0未变。loader按config建projection并严格加载，显式BCE无需换算法，尚须真实权重CPU加载及单步确认。

同NuNER不等于只改容量：上游config还记max_neg_type_ratio1→2、warmup5000→.1；完整checkpoint训练/选择清单未公开。后续即使固定赛事train48/10ep/loss/lr/decoder，也只能研究较大公开checkpoint替换，不做纯容量因果结论。ModelScope两准确ID实际404、Qwen正控制200，平台全局搜索不完整，只能说未找到可核镜像；不拿同名v2.1代替。FP32参数/梯度/两Adam矩理论7.352GB，加activation未知，encoder算量比例约3.56×；先实测最长4×384，不能拿medium17分钟承诺large同成本。
- large实际单步只1.308秒却峰值10.603GB，不能从一次首步与medium1.270秒接近就断言整轮同成本：首步冷启动、不同runtime、训练中平均文本长和checkpoint IO都混入时间。记录实测及假设分开的预算范围，不把算量比当真实耗时。CPU严格加载与训练窗对齐已通过，只证明兼容，不证明较大公开checkpoint能加分。

### GLiNER函数名focal不代表实际默认训练用了focal

原论文§2.2式3明确sum BCE。gliner0.2.22 `TrainingArguments` 和 `SpanModel.loss` 上层默认alpha=-1、gamma=0、negative=1、sum，覆盖底层 `focal_loss_with_logits` 自身的.25/2；不要据函数名或独立helper默认断言本路线遗漏原版focal。正确意图为Σvalid[-y logσ(z)-(1-y)log(1-σ(z))]。我们显式BCE除以Nvalid，且用稳定BCEWithLogits绕过H30错位；上游sum和其包含masked/padding分母的mean尺度都不同。缩放会影响梯度裁剪/Adam数值与不同batch权重，不能说整套原版训练等价复现。

train48有3558正/1705153负，数量约1:479.25，但medium10终点训练probe仅159正/48个首正窗R.97484，online全训练3558正R.84907（过程模型而非冻结终点）。按已有probabilitysum算原始BCE标量logit梯度绝对量Σpos(1-p)=630.527、Σneg p=626.815，近1:1；此非参数梯度或按batchmean加权梯度，仍足以说明不能从数量比例直接推出负梯度压倒。没有支持“正例塌缩”的现成证据，也没有跑过GLiNER focal对照。root明确本轮不启动focal，gamma2只是可选新假设，不进入当前计划；所有来源/训练统计见gliner-loss-research.json。

## GLiNER-BioMed 接口候选（2026-10-23，仅下载/CPU预检授权）

固定 `Ihor/gliner-biomed-large-v1.0@f7977121` 公开权重1.912GB，来源独审未发现其为benchmark few-shot成品，仍保留teacher四示范来源未披露边界。它是1024维head/24层DeBERTa，已有完整 `encoder_config` 和保存的128003词表（ENT128001/SEP128002），不能原样使用旧legacy loader：重复传encoder_config会报错，无条件添加FLERT会改变词表和类token索引。独立CPU入口保留新格式配置，strictload且weights_only；原large训练入口不动。SPM官方SHA与现有模型一致，可核验复用；新增tokenizer JSON/映射按准确revision保存。max_len2048不意味着本次扩大输入，仍只验既有384预算窗口；训练数据、当前BCE和在跑参数都不变。来源/限制/预算见[预检计划](evidence/round2/gliner-biomed-interface-plan.json)，尚无新GPU或泛化分数。

## 2026-10-23 large-v2 完整管线与启动器兼容

固定训练48的large-v2纯span union .769904优于medium48 .757484，完整固定Sap/rules总分却.720152低于medium .721504。不要把更强跨度识别或模型容量当成完整任务成功的替代指标；这次checkpoint替换已关闭，不迁移原dev或围绕同fold调参。来源/head/上游recipe也不同，因此不能把差异只归因参数数量。实际终点、两独立审查、真实model/state恢复位置见gliner-large-terminal-results.json。

远端系统 `python3` 实为3.10.12，新的外部launcher使用3.11新增 `hashlib.file_digest` 会在模型前退出。已改现有项目使用的流式 `sha256.update`，保留原job/原wrapper SHA/trace及实际解释器CPU hash自检。模型入口和decoder未改变；失败只计启动器问题，不计模型负结果。今后必须依据实际执行解释器版本检查标准库API，而非本地Python版本。外存gliner-biomed-nested/attempt1-pythonhash/与actual-wrapper-hash-probe.json；selftest在成功重启后补做，时序如实记录。

### BioMed领域预训练的固定监督对照

同nested32/1150窗、原BCE/固定10轮后，BioMed .654424仅略超medium32 .653745；新增实体同42TP、FP由27降16，但关系macro略低。公开BioMed零样本 .652014 到本监督 .654424 的增幅也有限。当前证据支持“此固定实现改善实体精度”，不支持用模型领域名称推断能大幅改善患者归属，也不支持仅根据训练probe R=1选择更多轮次。实际模型/状态、预算与完整分项在[终点证据](evidence/round2/gliner-biomed-supervised-results.json)，原dev没有新评估。

### 全本体类型预适配暂不投入

训练48已有OOF聚合显示，908无重叠漏项仅54能由指定HPO name/EXACT同ID直接覆盖，原Sap高置信覆盖40，合计60；未见训练词面不等于本体已有词面。孤立短语HPO多类预适配重复已做PhenoTagger；全HPO合成上下文跨度目标虽不同于353同HPO替换，但对多数未知表达迁移缺少直接证据。唯一CPU exact gate控制已否定：虽补40TP，但伴58实体FP、22关系FP，pooled Score下降.0008865、三折两降，保留旧政策并结案。现有覆盖缺口不应只按已知TP修接口，必须统计同时引入的误报与关系传播。细节与来源见[短机制结论](experiments/ontology-miss-mechanism.md)，不增加GPU或原dev实验。

### BioMed的focal配置提供新依据，但不是BCE塌缩证据

此前通用GLiNER上层默认BCE的结论仍成立；新BioMed固定配置另有.75/2/sum，不能混为同一个默认。实际本路线未跑过focal。固定配方可同时增加正类相对权重、压低容易span贡献，但训练probe已高，计数1:502不证明负梯度支配。CPU数学合同通过后仍只是一条待检验学习目标假设；保持mean而非上游sum以免再改变分母政策，不拆参数网格。跨A800/A30的相同权重推理复现不足以当作训练对照，`.100`如运行应fresh两臂，来源/成本见[提案](experiments/focal-loss-proposal.md)。
