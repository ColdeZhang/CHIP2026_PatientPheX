# 本体规则工作台账

本路线 `round2/ontology-rules`，worktree `/workspace/patientphex-worktrees/round2-ontology-rules`；只修改 `solutions/ontology_rules/`。恢复先读 main 的 AGENTS/HANDOFF/总台账并核实际文件/Git。

## 目标与可信基线

- 原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text，目标 Score≥.8；未达标，不合并实验实现。最高901已按用户单次授权官方接收，其余上传暂停。
- 本路线原dev固定 BioBERT迁移 **.738493753**（旧.737585917），独审7336b280；[证据](evidence/round2/public-phenotagger-biobert-dev-results.json)。主线程当前组合最佳以根台账为准。
- train64强BioBERT四fold **.730993616**，不能混作原dev指标。每个独立模型≤10B，允许集成，旧全链总上限已撤销。

## 当前执行（2026-10-24 08:21 UTC）

- 官方最高901的B预测已07:54:13真实accepted（回执8d39aba4，额度5→4），不查询成绩；现恢复本地冲0.8，其他上传暂停。
- **A800 indexed全文正式运行中**：08:07:31唯一 `proc_93f50e85`/PID65024，epoch1；原124×2/32updates/7170监督，整体8400+30，绝对截止 `1792837648.873858`。每epoch必须完整state跨机.100持久ACK后继续；恢复入口 `research/qwen35-fulltext32-indexed-formal-v1/controller`，源[formal_indexed](qwen35_sft32_full_report_training_readiness/formal_indexed/README.md)。原fullhead正式版从未启动，独立保留。
- **A100 H5固定20步已负结案**：唯一训练/完整F1与两质量已完成，主 .728697536469→.727938862709（−.000758674），同源诊断 .658357066464→.661787702194；不迁F0/dev、不调cut。原H5格式guard在held NN前失败保留，独立同权重恢复成功；两质量08:31:07已在原600截止内完成，GPU08:29:47释放交sequence。[结果](public_h5_adaptation_readiness/formal_train/RESULT.md)。
- 两实际工程已通过：head284.122秒/12真实反向，peakallocated32.693GB、reserved35.123GB，独审e35dba61；PubMed69.501秒/原201变量/三Adam更新，独审d2734bc0。均不代表质量提升。H5旧F1实际vec101（full64才103），原a465/8cb基线身份不变。
- 最近已核远端 `ed8a9c7`；两方案源码/启动/失败恢复及报告指针均已保存，原完整权重、数组和训练产物外存。以下为阶段历史，实际恢复以本节与新namespace为准。

## 已完成阶段：唯一最高方案901的官方B预测

- 用户最新明确授权只提交当前最高分方法，覆盖此前“暂不上传”的阶段约束；root统一协调提交与回执，不查询官方成绩。
- 冻结方法901原dev Score .754361398199；B输入878fe248/2060389B，100文献/244患者/3802段，公共合同2c169066。Task1沿train64规则/Sap64/CRF、三public和固定103特征/LGB803b，不fit、不调阈值、不混day1 train80。
- 原dev纯重建18ac gzip及内容byteexact（1.862476s），Bioformer中间候选值/类型/字段顺序也exact；Task1沿原crf_confidence:null，不加入confidence。源码f80669fb。
- 本owner只追加三public原native候选及103特征pool；sequence提供共同底座，retrieval固定LGB .3与关系组装。原公共H5/TF2.12环境在A100 mount476f完整，06:28 fresh GPU2空；root安排先三public后patient。唯一CPU census57.819秒完成，100doc/3794有效段/446799候选/362779unique/347634fresh、6100dictionary；CPU42d9。已冻三模型顺序总2400+30/单绝对deadline，源plan6fc113a8；唯一proc_f3f4ef06/PID199194于06:34:37.589启动，原native与threshold不变。PubMed已338.457+5.186秒rc0，100doc/8929canonical，manifest5ef5e971/canonicalfbcadc36已交retrieval；Bioformer也271.310+5.384秒rc0、100doc/8726，manifest4d1ea0d/canonicala38a4766已交retrieval；BioBERT也334.467+5.384秒rc0、100doc/8723，manifest6e920d5f/canonicalf66e1f45。三源整体970.229908秒rc0、06:50:47.914198 GPU2实际释放并已交sequence；Sap完整b0013d9f已恢复；Task1 pure池11.792190秒rc0、100doc/93710候选，final4838f658/5263866B、Bioformer中间53998556均已交retrieval固定803b预测；manifest3f158274。
- 全文formal8400训练已完成源码部署，但无campaign、无epoch job、无GPU启动；因官方优先暂停，保留私有目录与源码，仅在root确认本次提交接收后恢复。事前Score-only附录0714ffe3另存，旧policy e9a7不改：同强基线.6765463547，保留只按Score严格升，macro完整报告不加额外硬门，双升取Score高者；两fold复验保留，旧结案不重开。[暂停收据](qwen35_sft32_full_report_training_readiness/formal_train/controller/paused-before-GPU-official-priority.json)。

## 历史：两折全文输入对照与训练CPU可行性准备

- F1旧 **.660942003 → 全文单独 .676546355 / 事前主臂旧∪全文 .669554799**；主micro与macro均增，新增19TP26FP。80strict/0弃答，376.475652秒rc0；独审原d655/metadata8d19，303内容成员包190297B/d7cb2131与两个小审查附录已在.100完整SHA/fsync。
- F0旧 **.695443948 → 全文单独 .696603975 / 主旧∪全文 .695988393**，按事前Score升且macro不降门都true；但主micro **.690076336→.682352941** 下降、macro **.639022564→.648923742** 上升，仅新增6TP19FP，Score只+.000544446。58strict/0弃答，277.261910秒rc0；05:58:08.483853 GPU2实际释放。统一独审760cc08d通过，240内容+members共241成员包170467B/0e14e398已在.100逐SHA/fsync。原模型/评分不重跑。
- 各折原1189/947实体、全部非关系UTF8字节保持；各折两个完整预测共冻后新2v2/旧0。这里只是已知F1/F0开发对照，不能称原dev改善或.8达标；全文alone仍保留事前诊断身份。[实验](experiments/E-20261024-SFT32-FULL-REPORT.md)、[F1指针](qwen35_sft32_full_report_readiness/full80/result-pointer.json)、[F0指针](qwen35_sft32_full_report_readiness/f0/full58/result-pointer.json)。
- root新授权纯CPU四源控制：取sequence准确old∪reverse∪patient基线F1 .663841820660/F0 .709556743085，预定primary三方∪全文、diagnostic reverse∪patient∪全文（去old）。两折四完整pred共冻后各1v2，旧三方不重评；每臂须两折Score严格升且macro不降，若都过按两折合计增量选择。已知全文结果后的开发性质，不自动迁原dev，未来35条public-new支持不得丢失。已唯一0.423940秒完成，新4v2/旧0，共冻21c1e4a5：主F1 .665274114/F0 .701747140、诊断F1 .666488411/F0 .702172965；两臂均F0 Score/macro下降，selected=null，关闭迁移且不再扫组合。summary93770641；统一独审85c55eca通过，39内容件107814B/fb246a76已在.100逐SHA/fsync持久。[结案](four_source_fulltext_control/RESULT.md)。
- Qwen35原124/F2+F3/32doc全文训练仅盘点/CPU可行性准备：未找到该exact实际训练终点；旧SFT32局部两轮1149.460秒，原402594 tokens/max5523/监督7170。新124完整输入只改user证据、答案/其他字段保留，原文完整，255markers中254插入+1按固定重叠政策skip；native实际22.406947秒PASS：1009181总tokens/max13549唯一idx98，token比2.506697、平方长度比6.047086，工程cap13824；原encoded7fa9与全部7170监督IDs不变。同Torch selectedCE实际2.329901秒PASS，loss按声明容差、梯度最大差0；两轮2881.348秒只线性代理，正式GPU训练仍未授权。root现已授权唯一A800 GPU2训练压力探针420+TERM30，源165e9bfc/plan32f089df/entryadf673dd已冻结，实际15baseSHA/空卡通过；唯一proc_6439c798已288.173982秒rc0结束，12F/B+1Adam/三族更新/base冻结/finite全PASS；最长42.930650秒，峰allocated32.689GB/reserved59.115GB，06:16:04.859空卡释放。48件SHA/zip8ad38b1e已回收，统一工程审b39274ed通过，121内容+members共122成员包1302209B/48c74e22已.100逐SHA/fsync持久；新实际两轮计算代理4503–6395秒，另需每epoch约520MB跨机持久预算，root已授权fresh124×2/32step整体8400+30含逐epoch全state持久ACK，随后全80fresh1200+30；runner86d1297c/原69项CPU+controller48ade338/7项HTTP持久fixture已冻；源checkpoint后部署，当前无GPU运行，fresh base原r16与selected CE，先8微批非零LR诊断更新建Adam，再3累积+最长idx98；沿H51压力口径不重放warmup，状态退出即弃，不视为正式训练。来源/CPU独审dabb18f6通过。[完整训练可行性记录](experiments/E-20261024-SFT32-FULL-REPORT-TRAINING.md)。[实际CPU](qwen35_sft32_full_report_training_readiness/ACTUAL_CPU.md)。

## 最近：R1 F1完整负结案，F0扩展不启动

- 准确公开R1固定F1的80行已完成，78新+2精确缓存；**旧 .660942003 → R1单独 .605215986 / 唯一并集 .634402907**。两臂均未满足事前关系micro+macro升且macro不降；不迁原dev、不扩codec/预算。实体及非关联字节完全保持。[实验与分项](experiments/E-20261024-R1-F1.md)、[完整清单](deepseek_r1_readiness/full80-preparation/terminal-manifest.json) d8e5a992。
- 70行合法、10整行弃答（含2触顶），478/587 CID有答案；缺CID仅弃答。3446.521秒/rc0，05:09:32.952 A30 GPU1空卡；原monitor及唯一finish controller均结束，R1本身无运行中GPU/训练/传输，新SFT32探针另列。完整终点独审05d60bdb通过；65内容项＋members.json共66归档成员，405598字节包0873a63e已在.100全SHA/成员核验并fsync，不重模型/评分。
- F0已完成58真实编码、九资产私网copy及6028→3427最长工程（89.507秒，strict/normalized均8/8），统一独审fb9edc97。后继条件在F1质量前冻结；因union负而[关闭完整58](deepseek_r1_readiness/f0-preparation/final58-preparation/closed-before-start.json)，pending仍禁用，无F0质量。A800 GPU2预留已解除。

## 持久资源与接手

- `R=/workspace/patientphex-runs/round2-20261023`；完整R1源码/响应/预测在 `R/research/deepseek-r1-0528-qwen3-8b-feasibility/`，大文件不入Git。
- .100 mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，自有root `/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness`；完整模型9文件16,388,603,517B在 `models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6`，全SHA/header/399 tensors/8.19073536B通过。原full80-v1响应仍持久保留。
- A800副本 `/dev/shm/patientphex-r1-f0-1045-20261024/` 是易失缓存，不能替代.100持久源。所有原源/失败/接口纠正保留；旧AutoTokenizer GPU记录不能冒称发布输入质量。
- 当前新机制：已有SFT32原80提示只追加完整允许article，CPU11.767秒通过，max14390；主线程已授权input cap14400/gen1600与A800最长idx19单探针240+30，须先查上下文≥15990、真实缓存SHA/空卡。最长及完整80/两质量均已完成，实际结果见上方当前项。环境重连后先查原intent/job，不能重复start。

## 最近结案与索引

- 强BioBERT+NCR四fold **.730993616→.731229952**，仅2/4非降<事前3/4而关闭；mention净−7TP/−17FP、doc−5TP/−10FP，关系exact。[结案](experiments/E-20261024-PUBLIC-NCR-ADDITION.md)，独审de325a2c。
- NCR65件关键预测/指标/小HGB/source持久archive在.100 `research/ncr-strong-biobert-control-backup/20261024-terminal-v1/ncr-terminal-v1.tar.gz`，660803B SHA bb9aa427，全成员核验。
- 最近已核Git远端 **b4539bf40d004f99470620d06e04146766754a2a**（push+lsremote exact）。本次结案提交后另核，不为台账自身hash递归提交。
- 历史：[截至05:11台账](worklog/WORKLOG-through-20261024-0511.md)、[截至04:08](worklog/WORKLOG-through-20261024-0408.md)、[截至03:36](worklog/WORKLOG-through-20261024-0336.md)、[实验](experiments/)、[笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。

- 后继待办（官方B关键依赖已交后执行）：root新授权F0/F1训练内纯覆盖审计，比较当前LGB Task1文档HPO与oldpool∪public2合法关系提示空间；先查等价历史，只聚合来源/≤.95/真实关系FN供给，不读原dev gold、不新NN/fit、不改本次B。549低分/534边界FP分组后置。head_index_probe由head_projection_feasibility helper独立准备仅CPU/源码，owner明确清单统一Git，不碰旧formal、不GPU；工程probe须官方接收后root另决。

- head_index_probe独立18件ready2026d27e已SHA核齐：新probe143c2a3c/planbf7170e2，CPU0.156748秒的9训练+9同hidden局部+11拒绝检查通过，source41项通过；无部署/GPU，原formal不变。root事前后继边界：官方accepted后仅一个420+30工程控制；未过保留失败、不放宽数值门/连开诊断，继续原已验证formal；通过才另冻新正式版本与绑定，不能把理论head FLOPs节省称全训练实际提速。

- 训练内关系候选空间审计唯一0.716438秒rc0：Task1缺口F0仅1个doc-HPO/6对/覆盖0FN，F1仅2个/9对/覆盖2FN；三者均BioBERT-only原生供给但同时有semantic候选，非独立贡献。old/public2实际提示空间等价全过、原关系252/63/127与366/105/154精确复核，新评分/模型0。建议不专门开GPU扩展；完整输入先冻、literal与canonical分列。[结案建议](relation_candidate_space_followup/RESULT.md)。

- 后继参考FN分解一次0.055596秒rc0：F0 55合法无已解析正例＋47无HPO空间＋16非HPO/文本＋9支持拒；F1 86＋45＋7＋16；集合未解释0，原FN127/154不变。支持优先既有关系训练、另审候选识别适配，不据25正确被gate拒就放宽gate。[小诊断](relation_candidate_space_followup/fn_partition/RESULT.md)。

- 公开原生H5监督适配只读盘点：native接口可行，但原上游训练入口不先load H5、旧19120头builder与17266列/None末列不同，不能直接套。复用历史train64类覆盖，不重读标签；NO词面冲突、canonical多列与固定LGB分布迁移须另冻合同。[可行性](public_h5_adaptation_readiness/README.md)。无数据准备/训练/新GPU。head_index准备另获独审102f5148无阻断，仍未部署/GPU，原formal不改。

- root新授权公开PubMedH5 CPU准备：唯一原生H5/train32 F2+F3；源码/标签合同已冻结，公开缓存负例不混train48谱系。canonical多原列用概率质量监督，真实多canonical/NIL/缺列显式不支持，实际token32冲突整组排除；只CPU，未fit/GPU。候选主端点为替换该源后固定F1强103列LGB，与完整16旧端点比较，held不筛。

- PubMed train32 CPU完整成功：native14.164822秒/整体18.536317秒rc0，1658原native实际输入0截断/max21；冲突与unsupported保护后1252 unique=811正+27NO+237边界负+177普通负，31正query多原等价列、单轮64共20步。稳定logits loss同TF CPU0.329863秒5数值+5拒绝及Dense接口通过，未加载原模型/fit/GPU/heldgold。主比较已冻F1强Task1 .728697536469、同源诊断 .658357066464分别比自身旧端点；完整预算仍待真实反向工程。源码/小清单及全产物双地指针见[CPU完成](public_h5_adaptation_readiness/train32_prepare/README.md)。

- PubMed原H5唯一GPU工程入口已准备（未部署/未GPU）：独立gpu_probe命名空间，plan186b82ff，真实首64/次64/tail36、freshAdam5e−6三更新，原softmax/共享Dense合同与差异先存、原unused pooler2例外，其余encoder/head梯度与Adam状态核；总240+TERM30。15项本地静态/已有数据检查通过，7件119502B包c1c06368；activation/job不存在，仍等官方accepted及root调度，不自动20步或held评分。[工程入口](public_h5_adaptation_readiness/gpu_probe/README.md)。

- root已给PubMed短探针条件授权：收到真实官方accepted回执后，fresh空A100 GPU2可直接唯一240+30，无需再次批准；尚未收到回执/未建activation/未GPU。仍不自动20步或held质量。原plan186b82ff不改，另存[条件授权](public_h5_adaptation_readiness/gpu_probe/conditional-authorization.json)。

- 07:54:13官方901 B已真实accepted（8d39aba4，额度5→4，不查询成绩）；按root授权恢复两独立工程probe。fresh A100 GPU2=0MiB、A800 GPU2=1MiB/noapps，盘余54.0GB/1.51GB。H5原240+30启动准备；head原420+30新增复用stress的controller，仅绑定真实accepted及收齐head统计，不改plan/数值门/模型源；均无正式训练或质量授权。

- 两唯一GPU工程实际启动：A100 H5 proc_352c3ab0/PID256648，07:56:59，240+30；A800 head-index proc_20d3881d/PID47321，07:57:45，420+30，15基座文件全SHA及空卡通过。仅监控原作业，未知不重启；全产物在research/public-h5-train32-probe-readiness与research/qwen35-head-index-probe-v1，actual-start小指针已保存。

- head唯一工程已284.121738秒rc0/12次真实反向通过，A80008:02:29释放；peakallocated32.6926GB与旧近似，reserved35.1231GB较旧59.1153GB低，不能宣称大幅提速。root授权独立formal_indexed原124×2/32calls/8400+30；新helper原字节复用、旧formal保留，逐epoch完整状态持久ACK契约不变。H569.501秒工程正，root授权独立formal_train20步+F1/600+30；向量维数更正F1实际101，其余模型比较身份不变。两正式尚未启动。

- indexed正式epoch1已08:07:31 A800 GPU2唯一proc_93f50e85/PID65024，plan635493e4；完整8400原deadline已开始，原state恢复/每epoch .100持久ACK保持。head53件实际终审e35dba61已PASS。H5纯CPU旧F1 pool/primary8cb/diagnostic24d双字节精确复现（原LGB只apply一次，无fit/quality）；overlay缺json的首故障已保存、修复只补namespace并复用已成功primary。正式H5各独立源已冻结，等本次push后fresh启动600。

- H5工程原27成员zip已远端完整逐SHA重开；26小件实际本地，8MB概率数组原A100持久定位保留（非本地全恢复），不重算。H5正式部署write_file UNKNOWN实核只已有2分片、无start/job；恢复仅补余4、原包不变。paired质量入口提前修正旧library metrics缺gold_sha256，用既有F1元数据665a绑定，deadline每调用前硬核；尚无新评分。

- A10008:10:01明确LUNA_OFFLINE，H5正式尚无GPU/start，源/分片保留，owner最早08:13:06唯一库存后恢复；sequence同mount CPU未知启动也不重复。H5工程/正式源联合独审d2734bc0通过（真实full20/held质量待后继）；head实际/新正式源联合e35dba61通过。

- 08:17 A100再次明确LUNA_OFFLINE，H5正式依旧无start/job/600时钟，下一唯一库存≥08:20:08.743648；单条原包恢复入口已准备，只校验/补缺且不触模型。Qwen原93f50仍formal_training。已准备新adapter F1全80fresh入口，复用原加载/generate/parser AST，14400/1600不变；实际2epoch持久ACK绑定前模板不能执行，旧idx19缓存完全移除。[入口](qwen35_sft32_full_report_training_readiness/formal_indexed/f1_inference/README.md)。
