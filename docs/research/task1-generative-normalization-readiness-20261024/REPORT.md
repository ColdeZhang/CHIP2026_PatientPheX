# Task1 生成式规范化：CPU 查重与训练侧准备

2026-10-24。**唯一值得进一步准备的具体机制是：固定已有 mention 的原文坐标与局部上下文，监督生成全公开 HPO canonical name(s)／NIL，再精确查表。** “逐段/逐窗生成 exact quotes 再归一化”已经真实训评，不能作为未尝试机制重开。这里仅给研究提案；没有训练、模型/tokenizer调用、下载、新预测/评分、远端调用或 held/original-dev gold 读取。历史分数引用既有摘要，训练侧新统计只读 train32。

## 实际机制对照

| 机制 | 真实输入、目标和供给 | 已有事实与本次决定 |
|---|---|---|
| 固定 mention→公开名称生成 | 明确原跨度及上下文；名称来自完整19,119公开specific HPO，允许compound和独立NIL/NO状态；可生成候选Top-K之外的名称 | 所查源码与历史终点中未找到这个**监督生成目标**的完成运行。不是“LLM规范化从未做过”；旧Qwen已根据char8+Sap8候选零样本生成HP ID。具体新变量是全词表名称的监督生成，无需自由编造HP编号 |
| 逐passage/window→exact quotes→规范化 | 全原文窗口，生成正表型字串，回定位，再exact别名或候选链接 | **已做完**：旧Qwen3-8B train48、305窗、2epochs、78updates，F0与F1各fresh训练。F0固定Sap筛后实体控制 .711353522→.714060535；F1 .644796274→.643814605。直接非重叠并集两折负，跨折扩展关闭。不能因逐passage代替打包窗口而宣称是新技术路线 |

H59的Top8 CE是固定跨度、Sap新ID、±180字符上下文、BiomedBERT独立候选二分类；正式 .642033887→.637749023，严格嵌套CE入树也已负。它与新名称生成的候选空间、输出表示、seq2seq训练目标不同，但都是规范化；“加上下文”本身不是新变量。

当前 `canonical_names` 是**Task2逐患者完整全文→标准名称集合**，目标来自association，不接受待规范化的给定mention；它不证明Task1固定mention头已运行。当前 `grounded_generation` 是逐患者全文→[unit_id,quote,occurrence]，训练通过nearest-patient弱锚点获得quote；不是逐段所有实体抽取。GENRE的单实体自回归名称检索范式更接近本提案，其来源已在现有canonical准备报告核过；本次不新下载其非商用权重/代码。

旧逐窗实验的具体局限也保留：quote-only目标合并同词面的不同occurrence，后处理将字符串定位至所有出现并用否定启发式过滤；F1一次恢复后仍有1窗口非法，不能写成所有模型输出都合法。旧HPO步骤使用基础Qwen（没有抽取adapter），限制char8+Sap8候选并允许NIL/compound，不是名称全词表监督。其F1过滤新增60跨度在坐标层29TP/31FP，加入ID后20TP/43FP；不能把检索关系小涨当实体收益。

## train32 实际目标库存

冻结 train32 `38c353ea…`、HPO2026-06-23 `a5092cbd…`；复用 strict.py `31d380f5…` 的精确原坐标与完整compound校验。新 `census.py` 仅本地CPU，完整结果 `census.json`，实际约0.81秒。

- 32篇、2,362条实体注释；2,275正、87 NO。2,361个唯一原文occurrence，1条同跨度重复，不是删除难例。
- 可监督2,353行：**2,222 HPO + 44 NIL + 87 NO**。NO独立于NIL；其中2条NO的原HPO在允许分支外，仍保留NO监督，因为NO目标不依赖概念编号。8条分支外**正**注释显式排除，不变成NIL或负类。
- 16个compound行、最多5个名称；2,246正HPO原子、783个独立训练HPO。所有嵌套/交叉/词内跨度可按原字符坐标表达，不使用GLiNER的词边界限制，不能据此推定模型就会学好。
- 1,239种casefold词面中39种有不同监督目标，34种同时出现正与NO；所以输入须指定occurrence上下文，不能只按词面缓存标签。同一passage有5处正/NO同字串歧义，进一步说明quote-only列表无法表示完整occurrence标签。
- 拟每侧256原字符且保留整个mention，实际输入context最长581字符、总1,045,101字符，mention最长69、目标JSON最长117字符。**这些是字符数，不是token预算、GPU时长或已通过的native模板合同。** 2epochs、有效batch8会有590个更新，不能套用旧305窗/78步的542秒当保证。
- 2,222个HPO跨度中1,199被公开name/EXACT别名覆盖全部目标，1,023不被覆盖。它只说明监督映射不全是公开exact复述；**不是当前强池有1,023个错误**，也不是可提升上界。没有读取held错例来选择这些目标。
- 逐passage库存为1,122非空段，416段含正注释、44段含NO。这个数量仅说明可构建监督，不恢复已关闭旧quote方案。

本次目标原件 `train-targets.jsonl` 保留原pmc/offset/length/text、passage、context source offset、精确mention边界、目标和原注释索引；`excluded.json`保8条正例排除原因。不把完整训练正文入Git。

## 一个具体后继提案及真实未解决项

建议候选：**Qwen3-8B固定mention名称规范化，fresh LoRA两轮，epoch2唯一**。输入原跨度+同passage局部上下文，不注入旧预测ID、Sap候选或gold名称。输出用可审计的三分支对象：HPO→公开canonical名称列表；NIL→空names；NO→空names。compound完整保留，NIL来自44条真实监督，NO来自87条独立监督；不能把模型未知名称或无匹配自动叫NIL。

复用已执行 `train_lora.py` 的 messages+answer、完整chat prefix检查、prompt mask、EOS监督、越cap报错、LoRA r16/alpha32/dropout.05、BF16、lr1e-4、batch1/accum8、2epochs和完整epoch状态保存。旧实际8B基座8,190,735,360+LoRA43,646,976<10B；这是历史库存，未来新实例仍核真实参数。当前Qwen3原生训练入口已加入真实发布tokenizer身份、选定supervised logits数学合同与raw/EOS保存，可以复用这些合同，而不是无视新运行时差异直接启动旧脚本。没有现成“GPT外部API规范化”实际训练入口的证据，不臆造服务调用能力。

名称约束可借用 `canonical_names/schema_contract.py` 的发布token trie和beam-prefix无状态回调；**不能直接原样套用**：需要三分支状态、NIL/NO与HPO列表互斥、HPO非空、compound无重复，并对实际Qwen3 tokenizer重新编码。该源码目前说明production generate接线未验证；CPU trie成功不等于真实GPU generate完成。训练必须监督与推断语法相同的原生IDs，完整2353行prefix/mask和最大形状待后续具体准备，不在本次伪称已测。

主端点可固定为当前强Task1的**正mention原跨度不变，只替换其ID集合**；原NO对象直通，模型NO/非法/触顶输出保留原ID，HPO/NIL接受后精确查表，关系全部固定。这样先判断规范化是否有用，避免同时引入新跨度与删除器。单独用gold跨度验证名称准确率只能当诊断，不能当比赛Score；实际强池输入含边界误报/非实体，本2353行只有NO而没有这些困难负例，分布差异仍在。

**当前最关键的缺口不是再找模型，而是当前强池的训练侧可修复ID供给。** 后续若root选此提案，先沿现成严格OOF祖先核当前train32选中跨度：固定坐标正确但ID错、NIL/compound错各有多少，区分跨度误报；不得用本线F1诊断23TP/81FP逐条挑训练目标。若供给很少则关闭提案；即使生成目标新，也不足以证明值得GPU。此处未执行该新OOF错误诊断、未建新分支/训练，也不重新打开旧逐窗quote配方。

源、历史实际指纹与读取范围见 `sources.json`；结论限于列出的现有源码/记录，不声称全局不存在未发现实验。GLiNER负结案及独审763a5c81另有不可变终态，本准备不改变其预测、阈值或结论。
