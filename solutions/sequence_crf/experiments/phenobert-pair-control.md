# 指定 HPO 名称的公开三类 pair 特征

当前阶段：全部固定CPU控制完成，质量独审通过，整体不升级；当前原dev最佳仍.737585917，不新增原dev运行。公开135000缓存沿独审0eef457c复用，实际唯一四fit和预定九cut结果如下。

## 固定方法与边界

复用已审 top3/non-Sap prepared manifest `9ea7d683`：747,178 次候选出现、四个 outer train48→16，每个训练侧仍为三组 train32→16 OOF。候选键、顺序及旧字段完整保留；不混本体正在进行的两个 Sap 新特征，不引入新的 ID/跨度。旧 HGB160、树参数、seed20261023、标签和非重叠 greedy decoder 原样。全部非 entities 对象沿旧 cut.3 模板保留，包括原 .4 来源关联，不重新分配患者。

公开模型是 `EclipseCN/PhenoBERT@195d5df2ad9c6265e1de523f139694c7646eee29` 的 pair-only 原生模型，108,312,579 参数/201 tensors，权重 `353fc588…`；实际加载/原生接口独审 `5a8a7185` 已通过。只用原文候选 `text` 和指定 `hp.obo` 当前 preferred name；赛事标签不进入该模型。原生 lower、ASCII punctuation→空格、显式中间 `[SEP]` 和逐词 WordPiece 沿官方实现，wrapper 再加 CLS/尾 SEP，不能换成现代 tokenizer 默认 pair API。完整原生输入 tuple 相同才共享 forward。

三列固定为 `[Not Match, Related, Match]`。`Related` 来源是 query 表型→祖先/父级目标，不能解释为对称距离、完整 HPO 后验或患者关系概率。新增 `phenobert_not_match/related/match` 三个标量；compound 按 `;` 拆唯一原子 ID，分别与其 preferred name 配对，再取三列算术均值。这只是多个原子分数的均值，不是 compound 全正确概率。NIL/不在当前合法分支的 ID 不发请求，三列全 0 为未编码哨兵；真实 softmax 三列须有限、落在 [0,1] 且和为 1。混合合法/非法 compound 报错，不静默部分编码；当前实际池只有纯 NIL 未编码。

这和已负的赛事 span/context CE 不同：这里不以本赛训练标签拟合跨编码器、不使用患者或上下文；原 CE 是任务内二元监督。它也不同于 Sap 独立向量 cosine，以及 PhenoTagger 的全 HPO 分类头 posterior：该模型联合编码具体短语与具体名称，输出固定语义类别。公开模型未给完整权重训练清单，沿已有来源审查边界，不将来源审查夸大为本赛质量证据。

## 已完成的 CPU 证据

- 旧 top3 源码/计划 16 个快照保存在外部 `sequence_crf/phenobert-pair-control/snapshots/`，实际标签、decoder、池检查、旧模块装载等 13 个函数 AST 完全不变；新 main 仅加缓存绑定、三列以及匹配 top3 对照检查。该准备阶段未执行 fit；后续唯一正式fit见下文。
- 27 项纯函数合同通过：原子类顺序、去重均值、顺序不变、零哨兵、真实 softmax 约束、缺失/重复响应、非当前名称、字段覆盖拒绝、显式 native representation 复用及 ontology 分支。
- 全 747,178 次出现与 biomed 的 137,784 个 `(原文,atomic ID)` 请求逐项 join 通过，12.63 秒；369 compound、6,500 NIL，无缺失 pair、无混合非法 compound。完整 forward tuple 去重后 135,000；原生长度 median14/p95=22/max48，不截断。只读取输入/候选，不访问 prepared base 的标签/关联。
- `preparation-plan.json` 保留 `execution_authorized=false`，cache/scores 尚为空；不能把准备计划当作已执行命令。实际原生推断及准备的小源码逐字保存在 `phenobert_pair_control/native_source/`，其余官方源码按固定 commit/blob 重取。

## 已授权的固定执行方案，按依赖顺序进行

biomed 的 512 实际无标签输入 pilot：8×64 native FP32，forward3.18秒（首批冷2.584秒、暖批0.062–0.1255秒），总9.56秒、峰约716MB，CPU/GPU前两 pair 最大 logit 差 1.43e-6；不是质量测试。按 135,000 tuple、两倍最慢暖 batch 加30秒估约559.8秒，建议一次公共缓存 cap600秒+TERM15。缓存冻结后四个 HGB fit/infer 共 cap300秒：旧同池实际143.50秒，root另据当前Sap同池前三折171.41秒保留读缓存/检查余量；新只有3列，预算为估计而非完训保证；九 cut 解码/实体校准另 cap30秒。不重训任何神经模型。

后继固定 .3 完整四fold预测统一 freeze，再一次 v2/format；旧 top3 对照及已有曲线直接复用。随后只用预定九点 [.1,.2,…,.9]，以 pooled `(menF1+docF1)/2` 选统一 cut，tie 距 .4 最近再较高；完整36预测先冻结再读本轮校准标签，不按关系分数选点。匹配 top3 已校准最佳 .4 objective=.777516517；保留当前 Bioformer .3 的 .780192596 作实际旧最佳参照。全 train64 标签参与校准，最大值不称独立泛化；不扫描额外阈值、特征或模型，不自动迁移原 dev。

小索引：[准备证据](../evidence/round2/phenobert-pair-preparation.json)。外部根 `/workspace/patientphex-runs/round2-20261023/sequence_crf/phenobert-pair-control/`，完整请求、模型和批量响应不入 Git。

## 在首次fit/质量前补充的固定边界ID臂

root预注册一个30秒CPU解码控制：完整保留原Bioformer cut.3的实体边界、数量、顺序和其他字段，只在每个原边界的完整top3候选中按本次新HGB概率argmax选identifier。原ID必须exact存在，真并列先原ID、再identifier稳定序；不加cut、不加/删/挪跨度、不重分关联。8项区别性合同通过，方案v2 bf1469ad/正式plan8963ca27已在任何任务fit或新评分前冻结。四个主臂、四个固定边界臂以及匹配旧top3/原Bioformer完整对照共同freeze后各新臂只评一次；该臂不参与九cut选择。

公共缓存实测189.654秒（forward176.622）<600，135000键完整、归一/finite、6500NIL不请求模型，GPU已释放。批量缓存及2110batch来源独审0eef457c沿用，不重复NN。完整指针见[正式准备](../evidence/round2/phenobert-pair-launch.json)。


## 唯一终点（源码4424d5fd，已推送）

四fold实际fit/infer分别41.27/39.48/41.20/41.24秒，共163.194秒<300；固定边界ID解码1.556秒<30。主臂、固定边界臂、两组旧对照共同freeze `e8bb76ad` 后才读取新质量标签，四fold/pooled全部0错警；关联及其他非实体对象完全不变。没有再训公开模型，也没有原dev操作。

| 控制 | pooled Score | 对照与解释 |
|---|---:|---|
| 旧top3 / .3 | .723467295 | 匹配同候选池对照，复用旧完整报告 |
| 同池+三pair均值 / .3 | .725155643 | +.001688347；四foldΔ +.003355/+.002249/−.001536/+.002571；仍低未扩top3的Bioformer基底 |
| 原Bioformer / .3 | .725655389 | 当前保留的训练侧非嵌套基底，复用旧报告 |
| 固定其边界，仅新HGB换ID | .725763139 | +.000107750；men微升/doc微降，fold0负/1、2零/3正 |

主臂相对旧top3增加44TP/68FP、去除34TP/93FP，净10TP/−25FP。固定边界臂保持4315实体对象，只有10个identifier改变（各fold2/2/0/6）；按评分原子单元为加4TP/5FP、去2TP/8FP，不能把这两个分母混为“9次修改”。这显示公共pair特征有一定ID/旧池排序信号，但没有稳定超出现有强基底；仅+0.000108不足以据此重复原dev比较。

预先固定九cut在所有36预测freeze `a8428dcd` 后一次计算，3.191秒；统一选.3，pooled实体目标 **.779193104**，较匹配top3已校准最佳.777516517 +.001676587，但较保留Bioformer.3 .780192596 **−.000999492**。该目标由全train64标签选择，是校准拟合值，不是独立泛化或比赛Score。旧calibrator字节沿用，继承的`original_fixed03_negative_unchanged`字段不用于判断相对哪个基底的正负；以上实际数值明确区分同池改善与未超过现有强基底。

全部8新预测/4315固定边界决策、81合同、36cut/计数/选择经独立CPU复放通过（质量审`296db624`）；其范围不含重新训练全部四个选择器。合并独审85050d25已通过：188349 heldout旧字段及三列均值全部核对，fold0一次独立fit40.41秒、全部概率/pred byteexact；其余三fold同引擎复用，未机械重四fit。公共cache/模型沿既有独审，不重复NN。

若未来采用pair模型，已核必要神经参数9,475,880,164加108,312,579=**9,584,192,743**，仅是候选资源预算；本次并未将pair加入当前best，Qwen未使用bundle模块的计数边界仍按库存说明保留。完整预测/概率留外部，Git仅小结果索引和本次必要源码：[结果/源码/计划/预算/恢复](../evidence/round2/phenobert-pair-results.json)。
