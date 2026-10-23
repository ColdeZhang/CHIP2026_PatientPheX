# 指定 HPO 名称的公开三类 pair 特征

当前阶段：CPU/source 准备完成，公共短吞吐已由 biomed 测量；尚无本控制的批量缓存、HGB 拟合或质量结果，原 dev 不运行。目标是检验现有相同边界多个 HPO ID 之间，真实公开 pair 语义分数能否帮助选择器排序，而不是再扩大跨度或重复调整门槛。

## 固定方法与边界

复用已审 top3/non-Sap prepared manifest `9ea7d683`：747,178 次候选出现、四个 outer train48→16，每个训练侧仍为三组 train32→16 OOF。候选键、顺序及旧字段完整保留；不混本体正在进行的两个 Sap 新特征，不引入新的 ID/跨度。旧 HGB160、树参数、seed20261023、标签和非重叠 greedy decoder 原样。全部非 entities 对象沿旧 cut.3 模板保留，包括原 .4 来源关联，不重新分配患者。

公开模型是 `EclipseCN/PhenoBERT@195d5df2ad9c6265e1de523f139694c7646eee29` 的 pair-only 原生模型，108,312,579 参数/201 tensors，权重 `353fc588…`；实际加载/原生接口独审 `5a8a7185` 已通过。只用原文候选 `text` 和指定 `hp.obo` 当前 preferred name；赛事标签不进入该模型。原生 lower、ASCII punctuation→空格、显式中间 `[SEP]` 和逐词 WordPiece 沿官方实现，wrapper 再加 CLS/尾 SEP，不能换成现代 tokenizer 默认 pair API。完整原生输入 tuple 相同才共享 forward。

三列固定为 `[Not Match, Related, Match]`。`Related` 来源是 query 表型→祖先/父级目标，不能解释为对称距离、完整 HPO 后验或患者关系概率。新增 `phenobert_not_match/related/match` 三个标量；compound 按 `;` 拆唯一原子 ID，分别与其 preferred name 配对，再取三列算术均值。这只是多个原子分数的均值，不是 compound 全正确概率。NIL/不在当前合法分支的 ID 不发请求，三列全 0 为未编码哨兵；真实 softmax 三列须有限、落在 [0,1] 且和为 1。混合合法/非法 compound 报错，不静默部分编码；当前实际池只有纯 NIL 未编码。

这和已负的赛事 span/context CE 不同：这里不以本赛训练标签拟合跨编码器、不使用患者或上下文；原 CE 是任务内二元监督。它也不同于 Sap 独立向量 cosine，以及 PhenoTagger 的全 HPO 分类头 posterior：该模型联合编码具体短语与具体名称，输出固定语义类别。公开模型未给完整权重训练清单，沿已有来源审查边界，不将来源审查夸大为本赛质量证据。

## 已完成的 CPU 证据

- 旧 top3 源码/计划 16 个快照保存在外部 `sequence_crf/phenobert-pair-control/snapshots/`，实际标签、decoder、池检查、旧模块装载等 13 个函数 AST 完全不变；新 main 仅加缓存绑定、三列以及匹配 top3 对照检查。新入口尚未执行 fit。
- 27 项纯函数合同通过：原子类顺序、去重均值、顺序不变、零哨兵、真实 softmax 约束、缺失/重复响应、非当前名称、字段覆盖拒绝、显式 native representation 复用及 ontology 分支。
- 全 747,178 次出现与 biomed 的 137,784 个 `(原文,atomic ID)` 请求逐项 join 通过，12.63 秒；369 compound、6,500 NIL，无缺失 pair、无混合非法 compound。完整 forward tuple 去重后 135,000；原生长度 median14/p95=22/max48，不截断。只读取输入/候选，不访问 prepared base 的标签/关联。
- `preparation-plan.json` 保留 `execution_authorized=false`，cache/scores 尚为空；不能把准备计划当作已执行命令。实际原生推断及准备的小源码逐字保存在 `phenobert_pair_control/native_source/`，其余官方源码按固定 commit/blob 重取。

## 已授权的固定执行方案，按依赖顺序进行

biomed 的 512 实际无标签输入 pilot：8×64 native FP32，forward3.18秒（首批冷2.584秒、暖批0.062–0.1255秒），总9.56秒、峰约716MB，CPU/GPU前两 pair 最大 logit 差 1.43e-6；不是质量测试。按 135,000 tuple、两倍最慢暖 batch 加30秒估约559.8秒，建议一次公共缓存 cap600秒+TERM15。缓存冻结后四个 HGB fit/infer 共 cap300秒：旧同池实际143.50秒，root另据当前Sap同池前三折171.41秒保留读缓存/检查余量；新只有3列，预算为估计而非完训保证；九 cut 解码/实体校准另 cap30秒。不重训任何神经模型。

后继固定 .3 完整四fold预测统一 freeze，再一次 v2/format；旧 top3 对照及已有曲线直接复用。随后只用预定九点 [.1,.2,…,.9]，以 pooled `(menF1+docF1)/2` 选统一 cut，tie 距 .4 最近再较高；完整36预测先冻结再读本轮校准标签，不按关系分数选点。匹配 top3 已校准最佳 .4 objective=.777516517；保留当前 Bioformer .3 的 .780192596 作实际旧最佳参照。全 train64 标签参与校准，最大值不称独立泛化；不扫描额外阈值、特征或模型，不自动迁移原 dev。

小索引：[准备证据](../evidence/round2/phenobert-pair-preparation.json)。外部根 `/workspace/patientphex-runs/round2-20261023/sequence_crf/phenobert-pair-control/`，完整请求、模型和批量响应不入 Git。
