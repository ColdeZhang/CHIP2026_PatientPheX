# 本体规则工作台账

本路线 `round2/ontology-rules`，worktree `/workspace/patientphex-worktrees/round2-ontology-rules`；只修改 `solutions/ontology_rules/`。恢复先读 main 的 AGENTS/HANDOFF/总台账并核实际文件/Git。

## 目标与可信基线

- 原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text，目标 Score≥.8；未达标，不合并实验实现、不上传官方。
- 本路线原dev固定 BioBERT迁移 **.738493753**（旧.737585917），独审7336b280；[证据](evidence/round2/public-phenotagger-biobert-dev-results.json)。主线程当前组合最佳以根台账为准。
- train64强BioBERT四fold **.730993616**，不能混作原dev指标。每个独立模型≤10B，允许集成，旧全链总上限已撤销。

## 当前最高优先级：唯一最高方案901的官方B预测

- 用户最新明确授权只提交当前最高分方法，覆盖此前“暂不上传”的阶段约束；root统一协调提交与回执，不查询官方成绩。
- 冻结方法901原dev Score .754361398199；B输入878fe248/2060389B，100文献/244患者/3802段，公共合同2c169066。Task1沿train64规则/Sap64/CRF、三public和固定103特征/LGB803b，不fit、不调阈值、不混day1 train80。
- 原dev纯重建18ac gzip及内容byteexact（1.862476s），Bioformer中间候选值/类型/字段顺序也exact；Task1沿原crf_confidence:null，不加入confidence。源码f80669fb。
- 本owner只追加三public原native候选及103特征pool；sequence提供共同底座，retrieval固定LGB .3与关系组装。原公共H5/TF2.12环境在A100 mount476f完整，06:28 fresh GPU2空；root安排先三public后patient。唯一CPU census57.819秒完成，100doc/3794有效段/446799候选/362779unique/347634fresh、6100dictionary；CPU42d9。已冻三模型顺序总2400+30/单绝对deadline，源plan6fc113a8；唯一proc_f3f4ef06/PID199194于06:34:37.589启动，原native与threshold不变。
- 全文formal8400训练已完成源码部署，但无campaign、无epoch job、无GPU启动；因官方优先暂停，保留私有目录与源码，仅在root确认本次提交接收后恢复。[暂停收据](qwen35_sft32_full_report_training_readiness/formal_train/controller/paused-before-GPU-official-priority.json)。

## 当前：两折全文输入对照结案；更强缓存对照与训练CPU可行性准备

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
