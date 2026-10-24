# CRF 序列标注工作台账

实验 `E-20261023-009`，分支 `round2/sequence-crf`，初始基点 `00f7da8`。共享主分支已正常合并至 `0f8901e`，merge `773d12e` 已推送；同步前后本方案 Git tree 字节一致。新任务还应读取 `git show main:HANDOFF.MD` / `main:AGENTS.md`，再看本页。

## 目标与可信最佳结果

- 原 train64/dev16、指定 HPO 2026-06-23、公共 evaluator v2 all/text，**Score ≥ .8** 才申请主线程验收合并。本轮不上传官方。
- 本路线原政策 dev **.711324992**（独审PASS）：原CRF+Sap/LR基线保留，增加固定train48 GLiNER终点、train64 Sap筛选跨度，再沿既有LR；源码`4adf9da`，预测`gliner-dev-transfer/assembled/full-lr.jsonl`。对应仅增实体不改关系控制.706111684；独立train64 LR重拟合、三输出/全部指标逐字节复现，格式/HPO零错警。前一独审最佳.700635947及完整证据继续保留。
- 跨路线Pheno实体-only组合候选 **.712586161** 已独审PASS；固定nested48候选.712146627也已独审PASS。两项都只有温和增益，mention分项有下降，均未达.8。
- 新跨路线候选 **.728794595** 已独审PASS：全train64 source-HGB+公开Pheno新实体，原best aa107790 SFT关联对象完全保留；较原best .715759605 +.013034990，只提升实体两项。详见台账17；新fit/全概率/预测与指标独立复现（1112ced2），仍未达.8。
- 新双公开模型候选 **.732257765** 已独审PASS（fc98e9ca）：同完整SFT关联、全64双来源HGB+训练OOF预选cut.3；固定.4对照.731445835也提高。仅实体变化、仍未.8；真实295637B模型已保存，见台账20。
- 当前跨路线固定嵌套候选 **.737585917** 已独审PASS（9c068d49）：在root已审e9 .736526555上仅补38实体（14TP24FP），原368关系完全不变；men小降/doc升，仍未.8。恢复见[实体嵌套终点](evidence/round2/selector-nested-dev-results.json)。
- 内四折规则关系优于 LR，原 dev 相反；LR/rules 都评过原 dev，该选择使用过 dev，不能声称盲验证。尚未达到 .8。

## 正在执行与恢复入口

**固定候选逆序及old-or-reverse并集已获一次推断授权，准备启动。** 原80/SFT32的固定逆序+CID/code_map同步，80完整双逆byteexact，587语义候选不变；78条变/2条不变，总250691/max5691token，无截断，old80耗时代理282.5秒。历史未做等价控制；root最终预定义逆序alone诊断+old-or-reverse并集主臂，不运行intersection；两fullpred共同freeze后唯一F1，600+30秒，不原dev。[计划与成本](experiments/reverse-json-order.md)。

**关系JSON单检索示例已负结案，独立终审PASS（b1a178ca）。** 唯一同A800/SFT32/80提示，CPU输入与实际token/模型/运行环境均通过；80完整响应、587候选全部有效，无fallback/tokenlimit，runner304.593秒/wrapper313.157秒，GPU已00:12:04释放。完整fullpred先freeze再唯一F1评：Score **.660942003→.645783102**，micro−.044148462/macro−.016487140，净−27TP/+5FP；实体完全不变、0错警。按预设keep失败关闭，不扫示例/提示、不原dev。传输与本地解析依赖失败均留档，同源环境恢复未增加推断；当前无活动GPU。[结果及全部恢复](evidence/round2/retrieved-json-example-results.json)、[过程](experiments/retrieved-json-example.md)。

**当前实体控制均已完成，无活动作业。** 同资源77 HGB固定.4基线独审3c7c12e5：Score.648061812；binary final.650311382仅+.002249570（men升/doc降）。唯一当前nested F1固定veto负：.728144787→.720623999，删除152=71TP81FP，men/doc都降，关系不变；按预设关闭，不扫cut/不迁原dev。95列/train48 base与附加binary32不同资源如实披露；新veto独审a842969e已PASS。[对照结果](evidence/round2/entity-slot-legacy-baseline-results.json)、[veto结果/恢复](evidence/round2/entity-slot-selected-veto-results.json)。

**实体二元固定终点完成、独审PASS。** 严格train32候选链/HGB77 gate≥.1，637提示/2430候选；fresh9B独立LoRA、固定2ep160calls，A100训练1460.803秒、峰27.035GB。公共base/唯一epoch2同F1的398提示/1589候选，推断230.765秒，双完整预测先freeze后唯一评：men **.503219→.701740**、doc **.604003→.762191**、Score **.561134423→.650311382**；全部437旧关系及非entity字段保持，0错警。显著优于公开base，但**公开base比较不是当前97列best的直接比较；后补同77 HGB对照仅+.00225**，不升级原dev。训练resub gate偏差与候选覆盖限制保留。新增终点独审f8f50357通过，GPU2 23:43:03.639Z已释放，无本路线活动GPU；同F1 HGB及单次veto已完成，见上；不重GPU/扫阈值。[结果/全部恢复](evidence/round2/entity-slot-formal-results.json)、[过程与边界](experiments/entity-slot-feasibility.md)。

**公开pair路线最后旧池控制已负结案。** 原Bioformer旧pool244794行、旧字段/顺序不变，只加同3均值；4fit64.88秒、九cut1.11秒，统一选.3，实体目标 **.780192596→.777836002**（−.002356594），三fold负；mention净−21TP/+16FP、doc净−4TP/−3FP。关系/非实体对象保留，无新NN/原dev，关闭本配方不扫。独立终审35d9e67e通过：244794旧字段/3均值全核、fold0唯一重fit/预测exact、36decode曲线exact、format0。当前无活动进程。[完整结果/恢复](evidence/round2/phenobert-old-pool-results.json)、[方法](experiments/phenobert-old-pool-control.md)。

**公开 PhenoBERT pair 控制已完成，未升级。** 固定top3/non-Sap全部旧候选/字段，三类原子均值新增，严格四fold HGB160共163.19秒；固定.3主臂pooled **.725155643**，比匹配top3+.001688347，但低保留Bioformer.3的.725655389。九点训练校准3.19秒仍选.3，实体目标.779193104低保留.780192596；3折升/1折降。固定原边界仅ID臂10对象变更，Score **.725763139**（+.000107750），doc略降、fold0负/两折零/仅fold3正；不自动迁原dev。全部预测先共同freeze、关系/非实体保留、0错警；整体独审85050d25通过：188349 held旧字段及三列全核，唯一fold0独立fit概率/预测byteexact；其余三fold同引擎复用。公开cache189.65秒、GPU已释放，当前无活动进程。恢复见[完整结果](evidence/round2/phenobert-pair-results.json)、[方法与记录](experiments/phenobert-pair-control.md)；原准备/启动指针保留历史，禁止重复已完成fit。

1. **共指控制全部完成**：fold0 72窗严格定位新增18mentions；原规则/旧alias/offset-only三路输出与原baseline逐字节相同，Score仍.711353522。独立重放10产物一致，18项语义阅读17支持/1输入身份未决，未按复核结果过滤。见 [共指索引](evidence/round2/coreference-fold0.json) 与 [独立审查](evidence/round2/coreference-independent-review.json)。
2. **同adapter SFT小幅提升**：仅changed29重推，完整相同29行复用旧响应且实际权重/decoder/runtime核验一致；固定947实体，.674333431→.686852935，净TP+4/FP-1，仍低于rules .711353522。实际作业已结束，GPU2复查释放，无待恢复GPU任务。见 [SFT结果与完整恢复入口](evidence/round2/coreference-sft-results.json)。
3. **GLiNER训练内控制结案**：nested32→16固定fresh2ep/10ep的纯span非重叠组合为.700614/.702135；固定10ep+Sap全链.644762→.653745，69新增=42TP/27FP，四项均升，独审PASS。见[2ep](evidence/round2/gliner-supervised-results.json)、[10ep](evidence/round2/gliner-long10ep-results.json)、[固定链接](evidence/round2/gliner-link-results.json)。
4. **outer48→fold0同政策已完成**：A30 fresh10/4360steps耗996.64秒、峰值6.028GB；零样本/终点预测先冻结。固定原947实体+Sap筛选新增26项（16TP/10FP），关联7TP/4FP，Score **.711353522→.721504431**、四项均升，纯span/来源/完整输出重放独审均PASS。只保留final模型+latest，实际余477.22GB。详见[终点结果/完整恢复](evidence/round2/gliner-outer0-results.json)，外存gliner-outer0-a30/、gliner-link-outer0/。
5. **原dev迁移两臂已完成**：明确span48原文/标签是linker64子集且dev隔离；不重训64、不扫阈值。1103原始跨度经固定Sap等处理新增32实体（22TP/10FP），entity-only **.706111684**；同LR新增关联12TP/9FP、未移除旧关系，full-LR **.711324992**。两臂17:46:38Z统一冻结、零错警，独审PASS。[结果及全部恢复入口](evidence/round2/gliner-dev-transfer-results.json)；GPU1于17:45:11Z交检索、GPU0于17:46:11Z释放，无活动GPU。等待根预先固定其他有效候选组合，不自行扫顺序/阈值。
6. **固定GLiNER→Pheno组合已结案**：只追加12实体（4TP/8FP），保旧关系臂 **.712586161**；同LR臂 **.712070726**（新增2TP/5FP）。两臂独审PASS；mention分项略降，只有小总分收益，不扫组合顺序/阈值。[完整结果](evidence/round2/gliner-pheno-joint-dev.json)。
7. **全train64固定10已完成、未改善**：fresh64/2372窗/5930steps耗1300.262秒，峰6.021GB，最终model e7985761、真实Adam/RNG state8026c3a8再哈希通过，仅保epoch10+latest。原48同runtime候选byteexact。64主fullLR **.705869445**（较48下降.005456），entity-only.705271539；预先固定nested次臂.706234066，仍较差。所有输出18:18:59Z统一freeze后评，0错警；模型跨度净+2TP/+3FP但关系-1TP/+9FP，不能因全量训练就替换旧模型。GPU于18:18:39Z fresh核空交本体，实余217.22GB，本次任务已结案，后续large-v2任务见第9项。新终点独审PASS、三预测/全部指标独立复现；[完整结果与恢复入口](evidence/round2/gliner-full64-results.json)。
8. **固定嵌套实体控制完成**：nested57字节复现2d24bb99；同proper containment/HPO不交政策，outer48追加31（12TP/19FP），.721504→.723440；原dev48追加35（10TP/25FP），.711325→.712147，mention分下降而doc分升，所有旧关系不变、0错警，三路独立重放PASS。只保留温和收益，不扫规则；64主终点原政策及预声明nested次臂已统一freeze/评，均未超过48。[结果](evidence/round2/gliner-nested-control-results.json)。

9. **large-v2 固定终点完成、不升级**：fresh train48/1744窗/459,495,168参数、固定10ep4360steps耗2279.35秒，峰13.34GiB；工作区中断未重启远端。base/final/raw/Sap均exit0，19:26:36Z完整输出先freeze，格式0错警。固定原947实体+Sap/rules新增24项，Score **.720152028**，较原.711354升.008799但低于medium48 .721504431（-.001352403）；mention/micro较medium升，document/macro降。纯span非重叠93TP96FP、union .769904，不能拿纯span收益代替完整Score。GPU0 19:26:26Z fresh空，实际余203.93GB。保留medium，不扫参/不自行迁移原dev；终点CPU重放及实际谱系独审均PASS；root关闭本次替换、不迁移原dev。[完整结果/源码/全部恢复入口](evidence/round2/gliner-large-terminal-results.json)，外存gliner-large-v2/terminal-summary.json；原训练与最后链接managed均已结束。

10. **GLiNER-BioMed唯一nested16零样本完成**：公开f7977121实际477,981,696参数，来源/CPU/接口均独审通过；594窗2085跨度推理24.93秒、峰2.139GB。固定Sap32/.65/.92/原rules新增58实体（32TP/26FP），关联8TP/10FP，Score **.644761594→.652014251**，四项升，但略低同层medium32训练10ep .653745027；原dev未触。裸span F1 .453801高于已存medium-zero .277295，未经Sap非重叠union却.548780（118TP/966FP），不可直接叠原池。两阶段exit0，完整输出先freeze、格式0错警；新全链CPU重放/实际谱系独审均PASS 4b925364。GPU0 19:30:52Z fresh空释放，无活动GPU。第一次launcher因Python3.10缺file_digest在模型前退出，保留trace并修流式SHA，非模型负结果。新作业proc_b66bb6f7已完成；[完整结果/全部恢复与失败证据](evidence/round2/gliner-biomed-nested-results.json)，外存gliner-biomed-nested/。root现授权一次独立native-loader监督适配，见第11项，不更改本零样本终点。

11. **BioMed监督32×10已完成、小幅改善**：fresh公开基座/原1150窗/477,981,696参数，固定2880steps耗1660.16秒（27.67min）、峰13.78GiB；实际原生loader/显式BCE及唯一终点不变。训练/final推断/原监督Sap三阶段exit0，20:06:42Z完整freeze后统一评分，0错警。固定新增58实体42TP16FP、关联8TP10FP，Score **.654424302**，较同层medium32 **+.000679275**、较BioMed公开零样本+.002410051；mention/doc升而macro略降，不能夸大微小总分收益。原dev未触，不自动扩大训练。模型aed0d582及真实latest-state f4e01084均再次全SHA核对；20:06:37Z fresh核GPU0空释放，实余197.85GB，managed proc_64f58b79已退出，无本路线活动GPU。[完整结果/实际命令/恢复与成本](evidence/round2/gliner-biomed-supervised-results.json)，外存gliner-biomed-supervised/terminal-summary.json；新终点独审PASS（cd5afba8），完整预测/指标字节复现。原[计划](evidence/round2/gliner-biomed-supervised-plan.json)、[就绪证据](evidence/round2/gliner-biomed-supervised-readiness.json)、[启动指针](evidence/round2/gliner-biomed-supervised-launch.json)均保留，禁止重复开跑已完成作业。

12. **本体覆盖诊断及CPU exact控制已结案、负结果**：原train48三套OOF的908无重叠FN仅54被HPO name/EXACT正确覆盖，高置信Sap40、并集60，不优先全HPO短语预适配。只解除`c.exact`跳过，旧三对照byteexact、三新预测统一freeze后评：pooled **.667647648→.666761145**，等权平均.669729202→.668894975，一升两降；新增实体40TP58FP、移除1TP3FP，新增关系6TP22FP、移除1TP。保留旧政策，不扫阈值/不触原dev/不占GPU；独立复核PASS（6e49d0c8），六预测byteexact/全部新指标一致。[短机制结论](experiments/ontology-miss-mechanism.md)、[完整结果与复现入口](evidence/round2/exact-gate-control-results.json)。

13. **同A30固定focal三臂完成、均不升级**：原1150窗、两fresh medium32固定10ep/2880steps；BCE训练685.10秒、focal693.25秒，各峰6.030GB。完整训练子进程共1400.51秒<1800硬cap，六GPU阶段均exit0。评分前另冻唯一解析score cut .6011759516702939（实际loss梯度/解析零点CPU通过），其余Sap .65/.92与原装配不变；原两臂保留，三预测20:57:04Z统一freeze后唯一nested16评分。BCE **.655315611**，focal **.653881260**（-.001434351），解析cut **.655256389**（-.000059222）；后者实体两项小升/关系两项降。三臂均0错警、独立重装/指标/来源预算和解析过滤PASS（cdbd1f23）。只关闭这套固定配方，不泛化所有focal；未触原dev/不扫阈值。20:55:34Z fresh核GPU1空，实余446.93GB，managed proc_483790e9已完整退出，无本路线活动GPU。[完整结果/模型状态与复现入口](evidence/round2/focal-pair-results.json)，外存focal-loss-control/；原[启动](evidence/round2/focal-pair-launch.json)、[配对计划](evidence/round2/focal-pair-plan.json)与[解析CPU臂](evidence/round2/focal-analytic-endpoint-plan.json)均保留。


14. **公开新来源选择器准备（CPU/源码，不训练）**：严格OOF本体/Sap/CRF HGB已能删旧实体、替换冲突边界/ID，四折source/confidence和严格CE入树均已完成，不能称新机制。当前未有GLiNER/publicPheno完整同折池；最小新控制为旧source-HGB±公开Pheno候选/真实来源分数，复用三train32 OOF与秒级CPU训练入口。固定公开模型可无标签缓存train64后按篇跨折复用，赛事别名/CRF/Sap-LR/任务NER和二层标签仍按折；待bio helper的64cache与schema，尚未训练/推断/评分。[短判断与唯一提案](experiments/public-source-selector-readiness.md)、[11快照/12配置哈希证据](evidence/round2/selector-readiness.json)。

15. **唯一public source-HGB控制完成、独审PASS**：原source baseline `59bf4fa3`/预测`8ec02c6c`（.710359060）复用；真实公开64缓存6064项来源核验后，新pool训练+887/外层+257，六新特征、HGB160/.4/三train32 OOF不变，CPU13.40秒<120cap。两fullpred21:14:44Z共同freeze后唯一outer0评，新Score **.732901229**（+.022542168），四项升、0错警。实体增61TP38FP/删12TP39FP，public-only增23TP13FP、原池重选增38TP25FP；关系净+14TP/+7FP。支持新来源+特征整体方法，未触原dev/不宣称.8；一次新fit独立复现PASS（8107cc27）。[完整结果与恢复](evidence/round2/public-selector-results.json)、[固定计划](evidence/round2/public-selector-plan.json)，外存 `public-selector-control/`。无本路线活动GPU。

16. **固定public selector四fold已完成、独审PASS**：source-HGB同法outer1/2/3仅改reference白名单，9套旧train32资源SHA/隔离齐；三次CPU共47.06秒<120cap，无新NN。三新fullpred21:18:45Z共同freeze后一次评分，f1 .663220→.720292、f2 .721467→.738304、f3 .680343→.703019；f2关系两项降仍保留。加复用f0四折Score全升，pooled **.692527322→.722651022**、mean .693847251→.723628971；pooled四项升、实体净+281TP/-51FP、关系+61TP/+23FP。无原dev。outer0独审8107cc27已PASS；三新fit/全概率及预测/全部指标/谱系独审PASS（c9428a9e）。[完整结果](evidence/round2/public-selector-followup-results.json)、[计划](evidence/round2/public-selector-followup-plan.json)，外存 `public-selector-followup/`。

17. **固定原dev迁移已完成、独审PASS**：全train64/4组旧train48 OOF，旧60182例+公开1142=61324/4092正，85+6=91自然列，唯一CPU16.709秒。两新/两旧预测21:24:07Z共同freeze后一次v2，0错警。A新实体+原规则 **.659239099**（匹配旧.639287+.019952）；B同新实体+aa107790全部关联原样保留 **.728794595**（比原best .715760+.013035），men .782359/doc .807393，关系micro .694656/macro .630770完全相同。相对best实体增58TP52FP/删23TP62FP；仍未.8、不扫参数。独立新fit/全概率/两预测/全部指标与关联保留复核PASS（1112ced2）。[完整结果](evidence/round2/public-selector-dev-results.json)、[固定计划](evidence/round2/public-selector-dev-plan.json)，外存 `public-selector-dev-transfer/`。


18. **独立Bioformer四fold控制完成、独审PASS**：PubMed .95六字段不变，增加独立Bioformer .95来源/六字段（真实分数不混）；四次CPU共51.494秒<120cap，无新NN。4新+4旧完整预测21:31:35Z共同freeze后唯一评分，pooled **.722651022→.724556074**（+.001905），四汇总分项升；foldΔ[-.000862,+.000012,+.007617,+.001178]，首折下降保留。实体增56TP51FP/删37TP63FP，关联增19TP32FP/删15TP30FP；0错警。新选择输出无Bioformer-only键，增量107项来自已有pool重选；这是键来源统计，不能据此单独归因特征。原dev未触，四新fit/全概率/预测/分数/实际cache谱系独审PASS（0efe738e）；root关闭负的lowgate，不做联合，转预设OOF cut校准。[完整结果/恢复](evidence/round2/public-bioformer-selector-results.json)、[计划](evidence/round2/public-bioformer-selector-plan.json)，外存public-bioformer-selector-control/。


19. **统一OOF cut校准完成、独审PASS**：固定九点.1… .9/原greedy，36完整decode21:35:13Z先freeze后读取train64实体标签，.4四fullpred逐字节复现。总2.125秒CPU，按pooled(menF1+docF1)/2唯一选 **.3**：.777993968@.4→.780192596@.3，实体men .763831/doc .796554、4315条（+161）；mention+70TP91FP/doc+41TP48FP，fold目标Δ[+.003036,+.007263,-.004463,+.001727]。这是全train64选择后的**校准拟合指标**，不是新盲验证Score；无关系选型、无原dev。独立36预测/全曲线/OOF来源/选点复核PASS（234064fe）；后继仅固定.4/.3同SFT关系共同迁移，不扩网格。[结果](evidence/round2/public-bioformer-cut-results.json)、[计划](evidence/round2/public-bioformer-cut-plan.json)，外存public-bioformer-cut-calibration/。


20. **双公开模型原dev两cut已完成、独审PASS**：全train64/4×48 OOF、61781例4147正/97自然列，CPU fit14.755秒、三阶段exit0；两新+旧16cd/model/prob21:40:00Z共同freeze后唯一v2，0错警。固定.4 **.731445835**（较旧+.002651）、OOF预选.3 **.732257765**（+.003463；比.4+.000812）。.3 men .786730/doc .816874，较.4 men下降/doc上升；全部aa107关系TP273/FP60/FN180及micro/macro完全相同。相对旧best实体净+14TP/+10FP，未.8，不追加cut。模型selector.pkl295637B SHA db2fc288已实际保存，Python3.12.14/sklearn1.8.0，独立重fit概率/预测/模型byteexact，实际模型重载16篇全部13144概率exact；根负责另备份。旧16cd31成员备份059d680a保留。[完整结果/恢复](evidence/round2/public-bioformer-dev-results.json)、[计划](evidence/round2/public-bioformer-dev-plan.json)，外存public-bioformer-dev-transfer/；无活动GPU/新训练任务。源码与终点检查点`8b830ec`已推送，当前只补独审结案。


## 已完成的近期结论

- 关系 LoRA：204 条严格嵌套 OOF 提示独立审查通过，2epoch/52steps 已完成。同 A800 基座 `.623068` → adapter `.674333`，减少过度归属但低于同实体规则 `.711354`；固定集合投票也未超规则。原 dev 未新增测试。
- 本体校准池的 24 个新概念/26 支持跨度经现有 adapter 判断，保留原实体及规则关系，仅增 14 项关系（4TP/10FP），fold0 `.712942`。边际精度较低，尚不推广。
- 训练锚点诊断独立复算通过：967 个 SFT 候选内正关系中，363 项实际局部证据无目标患者 marker；全局 anchors 仍可见，不能称不可回答率。完整全文 LLM 共指此前没有实际运行。
- 共指首轮训练输入 pilot 因子串定位与身份混淆失败；同 3 篇仅修订一次为唯一原文 context＋完整词边界，严格保留 4 项新增且逐条身份检查通过，召回保守。已按此协议复验fold0，新增18mentions但规则零变化；同adapter SFT有小增益但未超规则，不继续按结果调提示。
- 神经 NER、HPO 重排、零样本关系、thinking 小样本、O 编号锚点、外部表格等正负结果与限制均保留在实验报告。O 编号四折不稳定、表格新增关系均为本地 FP，未升级默认。未使用包含本地 dev 的赛事公开成品权重。

## 证据与下一步

- 新增[当前最佳端到端资源清单](experiments/best-pipeline-resource-inventory.md)：只读确认实际HGB/CRF/Sap及关系模块恢复入口，不跑test、不新增训练。F0/F1 cut.3 OOF实体支持集已交检索，明确全train64选cut的开发使用；现已补候选池→已存HGB的薄入口并完成无评分字节回放，原文候选生成与跨目录CRF模型路径适配仍未整合。

- [实验过程](experiments/E-20261023-009.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。完整产物根目录：`/workspace/patientphex-runs/round2-20261023/sequence_crf/`；权重/预测/原始响应/完整日志不入 Git。
- [SFT 固定结果与恢复索引](evidence/round2/relation-sft-fold0-results.json)、[新关系池结果](evidence/round2/new-ontology-relation-results.json)、[训练锚点诊断](evidence/round2/patient-anchor-audit.json)。A800 现有 base 和本路线 epoch2 adapter 可只读复用；旧 180 概念/29 提示弱池未运行。
- 共指完成后保留全部控制，包括负结果；如无可靠收益，停止当前原型，不以格式通过替代身份或分数验证。后续再协调检索新跨度与本体候选组合，避免混淆本次单变量对照。
- 历史基线口径：旧 baseline-fold0 未拆复合关系标签 `.613929`，修复后 `.603398`，实体不变；fold1–3 已使用修复版。不要将四折旧关系基线写成同一未修补模型。

### 已存HGB推断入口（无新增实验质量分）

`predict_existing_selector.py` 只load模型+vectorizer，接受任意输入/合并候选/base/source路径与cut，可选原关联原样装配；无fit/gold读取。原16篇13144概率和完整cut.3预测分别复现c032ca68/6401a24e，实际约1.99秒；16项合同含概率毒化后实际重算、任意ID/空池、fit及annotation-IO陷阱通过。独立增量审PASS（7c3b8401），全16实际重算及16合同独立重放一致。[使用与边界](experiments/load-existing-selector.md)、[实际命令/所有证据](evidence/round2/selector-load-only.json)。未跑test100/官方；不把新HGB实体自动改作Task2候选。

### 固定WIS解码控制已完成：零变化

只改旧Bioformer四fold/cut.3非重叠选择，精确p-.3效用与固定tie；80组独立穷举/边界合同通过。实际3.31秒，8fullpred先freeze，新旧64篇全部byteexact、实体/关系TPFP全Δ0，pooled仍.725655389；exact .3候选0。当前池此效用下greedy已同最优，关闭该固定控制，不扫效用、不迁移原dev；无fit/GPU。旧baseline首测指标按同SHA复用，独立新增审PASS（4802573a）。[结果/恢复](evidence/round2/weighted-interval-results.json)、[实验](experiments/weighted-interval-control.md)。

### 旧HGB池严格嵌套追加已完成：温和正收益已独审

旧GLiNER纯边界/合法HPO/否定政策复用，上游明确换HGB .3/原排序；2.63秒CPU，pooled .725655389→.727975656（+.002320267），195新增86TP109FP，全部旧实体/关联保留。三fold正、f3−.001194855，完整碰撞/拒绝证据外存，0错警；独审PASS f5482b85；唯一原dev迁移已完成 .736526555→.737585917（+.001059362），38实体14TP24FP、men降/doc升，368关系不变，0错警，新增端点独审PASS（9c068d49）。仍无fit/GPU。[结果/恢复](evidence/round2/selector-nested-results.json)。

- 2026-10-23 动态微批探针已115.957秒PASS/卡释放：五组真实更新、native双列exact/梯度通过，峰26.786GB；预计更新阶段23.5–23.8分钟但尚无正式训练。入口 evidence/round2/entity-slot-microbatch-probe-results.json，actual独审待回。

- 2026-10-23 根已授权唯一fresh实体槽train32×2：637提示/160calls，2200+30训练cap，base/final同398F1输入固定.5，600+30推断cap；计划 evidence/round2/entity-slot-formal-plan.json，尚未启动时不称完训。

- 实际23:13:53.928Z已fresh启动 `proc_3fc5df9d-531b-4280-9257-4d2f809d196c`，A100GPU2；主cap至23:50:33.928Z+30秒grace。恢复入口外部 `entity-slot-formal32-v1/train-job.json`。本地controller后补重入/未来infer计划闭环，live训练/source/plan不变。

- 正式入口/CPU独审 `eca2f72e` PASS，live源与plan不变；未来infer使用带重入和计划闭环守卫的controller。实际训练/唯一epoch2质量仍待终点，不以入口通过代替完训。
