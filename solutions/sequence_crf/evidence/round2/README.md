# 第二轮证据索引

保存小型指标、配置、源码/产物哈希、复现命令和少量错误分析。全量预测、模型和原始日志保存在台账指向的仓库外目录；此目录不存凭据或完整中间日志。当前证据包括四折校准、关系对照、语义新增跨度、GPU小型manifest、数据缓存顺序核验及后续消融。


- `relation-features-inner4.json`：两种上下文LR、4内折分项、来源SHA及原dev复验前的固定选型。
- `ner-negation-hpo-ablation.json`：五标签与纯HPO预适应对照，小型manifest/指标；完整预测与模型留仓库外。
- `inner4-crf-calibration.json`：保留旧fold0并追加修复版，关系训练标签代码口径显式区分。

- `independent-review-summary.json`：独立CPU重训/缓存装配范围、完整报告SHA与位置、源码/预测SHA及选择限制。
- `biolink-large-source.json` / `biolinklarge-fold0.json`：通用底座来源、完整权重SHA、NER负结果与basename哈希覆盖补充证据。

- `id-anchors-inner4.json`：官方编号来源、四折锚点计数、原始/内部派生输入SHA、源码和输出SHA，以及两种关系规则混合正负结果；不升级默认。

- `thinking-paired-pilot.json`：同GPU/runtime配对12提示，条件诊断/有效率/预算，不是比赛Score。
- `table-policy-train.json` / `tables-fold0.json`：训练输入侧预冻结政策与负结果，源表/解析器/原始输入SHA；Task1不变。

- `relation-sft-data.json`：嵌套OOF候选与训练标签覆盖统计、来源SHA；完整样本/嵌套重放命令外存。

- `relation-sft-independent-review.json`：3套CRF重训/增强装配和204提示反事实重放、全部监督目标、实际远端训练绑定；明确开发验证和未重跑Sap的范围。

- `relation-sft-training.json`：完整2epoch/52step训练统计、固定epoch2所有产物SHA、独立数据审核来源和磁盘预算；尚不表示评测收益。

- `relation-sft-fold0-results.json`：配对58提示完整结果、所有固定集合负结果、作业恢复与响应/预测SHA；原dev最佳不变，无官方提交。

- `new-ontology-relation-pool.json` / `new-ontology-relation-results.json`：同折校准新候选、10提示固定adapter、仅追加关系的小幅开发收益与较低边际精度；不推广。
- `patient-anchor-audit.json`：仅train48锚点密度、gold同概念跨度的乐观距离代理和实际204提示marker覆盖，全文聚合外存；不是模型分数。

- `coreference-plan.json` / `coreference-pilot-failure.json` / `coreference-context-pilot.json`：输入侧预选3篇、原型定位/身份失败与唯一context协议复验，不是gold共指准确率。
- `coreference-fold0.json`：正式72窗、原文定位、冻结规则三控制和SFT58排队入口；当前三rules预测byteexact，阶段二结果待回收。

- `coreference-independent-review.json`：72窗输入/18新增定位及三规则/no-op独立字节复现，17身份支持/1未决；规则零收益，SFT变更29短批另评。

- `coreference-sft-results.json`：共指只改mentions同adapter控制，29新＋29严格缓存逐行来源；fold0 .674333→.686853，仍低于rules，原dev不动。

- `coreference-sft-independent-review.json`：实际29缓存＋29新响应最终独立核验、6产物byteexact、两分数一致。

- `training-loss-diagnosis.json`：仅outertrain48三nested OOF实体/关系损失分解，用于下一路线选择，不是新dev分。

- `training-loss-independent-review.json`：三nested全部计数/分层/marker丢失独立复算。GLiNER候选结构设计见P-20261023-GLINER.md，当前无新评测分数。

- `gliner-span-diagnostic.json`：冻结三篇训练候选的纯跨度负结果，无HPO/官方分数。
- `gliner-supervised-preparation.json`：官方标签对齐、上游loss问题/显式BCE检查、A800来源与固定2epoch计划。

- `gliner-supervised-results.json`：固定nested32→16零样本/2epoch纯跨度配对、非重叠增量与误差、冻结UTC/预测SHA和训练状态恢复入口；不是比赛Score。

- `gliner-long10ep-plan.json` / `gliner-long10ep-results.json`：同A800 fresh10固定预算开发对照、逐轮训练诊断/最终纯跨度增量、独审与GPU释放/保留检查点。

- `gliner-link-plan.json` / `gliner-link-results.json`：固定epoch10＋同nested32 Sap过滤全链控制，69新增/四项得分/冻结SHA，独审已通过。

- `gliner-outer0-plan.json`：outer48固定10ep正式A30作业、源/模型/1744窗/预算/真实smoke和恢复入口；运行中不代表新分数。
- `gliner-outer0-results.json`：同A30训练/零样本/唯一终点来源、实际保留检查点、纯跨度与固定Sap完整增量；fold0 .711354→.721504，原dev未在本控制评测。
- `gliner-dev-transfer-plan.json`：原dev单次迁移两臂、span48/linker64分立来源、旧LR零新增字节复现；未在计划阶段评分。
- `gliner-dev-transfer-results.json`：原dev两臂统一冻结，entity-only .706112 / full-LR .711325，32实体/21关系增量、来源与GPU释放证据。

- `gliner-pheno-joint-dev.json`：固定跨路线追加两臂、全部增量与独审PASS；实体-only .712586，关系重算较差，停止扫描。
- `gliner-full64-plan.json`：2372窗/5930步固定规模控制、准备独审/实际smoke、同runtime48对照与45分钟训练硬cap；终点前不表示新分数。

- `gliner-nested-control-plan.json`：只允许不同HPO的proper containment，原nested预测字节复现；单次outer48实体-only控制，既有GPU64政策不变。

- `gliner-nested-transfer-plan.json` / `gliner-nested-control-results.json`：outer31与原dev35追加，关系保留；原dev总分微升但mention下降，完整负分项与双freeze，独审排队；64次臂预先固定。

- `gliner-full64-results.json`：全64唯一终点主.705869/nested.706234，未超过48；同runtime旧模型字节复用、训练/真实状态SHA、三臂统一freeze、存储和GPU释放，终点与对照复用独审PASS。

- `gliner-large-feasibility.json` / `gliner-large-smoke-plan.json`：同NuNER large-v2确切来源与上游配方差异，ModelScope查询/HF回退、静态兼容与仅训练侧单步授权；尚无新模型分数。

- `gliner-large-smoke-results.json`：459.495M实际严格加载、1744窗对齐和最长4×384单步1.308秒/10.603GB、显式释放；正式10ep未运行，没有新分数。

- `gliner-large-formal-plan.json`：授权fresh train48×10/4360步，75min训练硬cap，先等本体交卡；同原947实体池/Sap/rules，唯一终点，不碰原dev。

- `gliner-loss-research.json`：论文/实际上层默认BCE与底层focal默认区分，旧训练侧计数不支持positive塌缩；本轮不启动focal，不修改large固定训练。

- `gliner-large-terminal-results.json`：459M固定48终点与完整来源、真实model/state恢复、两独审PASS；.720152略低medium48，关闭替换、不迁移原dev。
- `gliner-biomed-interface-results.json` / `gliner-biomed-nested-plan.json` / `gliner-biomed-nested-results.json`：公开生医基座来源/实际CPU与唯一nested零样本固定全链.652014，失败启动器证据与GPU释放，不能虚构medium-zero完整Score。
- `gliner-biomed-supervised-plan.json`：独立native-loader训练32×10固定适配，CPU1150窗/单步资源条件、2400秒训练硬cap；计划不代表训练结果。

- [BioMed监督32固定10终点](gliner-biomed-supervised-results.json)：.654424，较medium32+.000679，58实体42TP16FP；实际模型/状态/命令/成本完整，新终点独审PASS（cd5afba8）。

- [本体漏项覆盖](ontology-miss-coverage.json)、[唯一exact控制计划](exact-gate-control-plan.json)与[结果](exact-gate-control-results.json)：既有train48三嵌套诊断及固定CPU对照；解除exact过滤pooled下降.0008865、两折退步，六预测与新指标独审PASS（6e49d0c8），保旧政策，不扩GPU/原dev。

- [固定focal提案](focal-loss-proposal.json)与[纯CPU合同](focal-loss-cpu.json)：BioMed配置.75/2来源、实际BCE训练清单、10组数学/梯度/mask通过与同环境成本；无新GPU或质量结论。

- [focal同环境配对计划](focal-pair-plan.json)：A800存储不足后选择.100 fresh BCE/focal；原1150窗/10ep，shared1800s与TERM30，CPU/单步条件及同固定Sap装配。

- [focal三臂固定终点](focal-pair-results.json)：同A30 fresh BCE .655316、focal .653881、评分前唯一解析cut .655256；两focal均不升级。三fullpred、真实model/state、1400.51秒共享训练预算、GPU释放与独审cdbd1f23完整索引。原.5仅候选cut，Sap .65/.92保持；无原dev/阈值扫描。

- [选择器复用条件](selector-readiness.json)：11不可变源快照、12既有配置来源核对；已有全池OOF树允许删换，缺新公开来源完整缓存。唯一建议source-HGB±publicPheno，无新训练/推断/金标评分；固定公开前向跨折可共享，任务学习资源仍逐折。
# 最新有界CPU准备

- [public-selector-plan.json](public-selector-plan.json)、[public-selector-results.json](public-selector-results.json)：原严格OOF source-HGB加公开Pheno真实源/分数；CPU13.40秒，outer0 .710359→.732901，四项升，旧基线复用/新来源与旧删换损益齐；独立新fit复现PASS，原dev未动。
- [public-selector-followup-plan.json](public-selector-followup-plan.json)、[四折结果](public-selector-followup-results.json)：三新CPU共47.06秒，统一freeze后评分，四折Score均升，pooled .692527→.722651；fold2关系两项降如实保留。outer0独审PASS，三新fit/全概率/指标/谱系独审PASS（c9428a9e）。
- [public-selector-dev-plan.json](public-selector-dev-plan.json)、[原dev两臂结果](public-selector-dev-results.json)：全64 CPU16.71秒/91列，A匹配规则.639287→.659239；B保原SFT关联 .715760→.728795，只提升实体指标。两新/两旧共同freeze后一次评，0错警，独审中、未.8。


原dev public-selector 两臂独审已完成：`validation/public-selector-dev-transfer-review/review.json` SHA `1112ced231cc43f7769482ff189fde4e980dd895fb91d5807d9244d256b157c4`。独立一次新全64 fit与全部概率/A/B预测逐字节一致，v2/格式全同；B .7287945949、原aa107关联完整保留。旧对照未重训/重评，不改变原dev已用于比较的限制。

- [独立Bioformer四fold控制](public-bioformer-selector-results.json)：固定双模型六字段，pooled .724556（+.001905），一折负；51.494秒CPU，待唯一独立复核，无原dev/GPU。
