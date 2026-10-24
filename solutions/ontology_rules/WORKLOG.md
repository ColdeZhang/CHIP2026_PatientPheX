# 本体规则工作台账

本路线 `round2/ontology-rules`，worktree `/workspace/patientphex-worktrees/round2-ontology-rules`；只修改 `solutions/ontology_rules/`。恢复先读 main 的 AGENTS/HANDOFF/总台账并核实际文件/Git。

## 目标与可信基线

- 原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text，目标 Score≥.8；未达标，不合并实验实现、不上传官方。
- 本路线原dev固定 BioBERT迁移 **.738493753**（旧.737585917），独审7336b280；[证据](evidence/round2/public-phenotagger-biobert-dev-results.json)。主线程当前组合最佳以根台账为准。
- train64强BioBERT四fold **.730993616**，不能混作原dev指标。每个独立模型≤10B，允许集成，旧全链总上限已撤销。

## 当前：F1全文对照正结案；固定F0复核准备

- 固定F1开发对照：旧 **.660942003 → 全文单独 .676546355 / 事前主臂旧∪全文 .669554799**。主关系micro **.686600221→.694092827**、macro **.542228853→.569187435**，主retain=true；新增19TP+26FP。全文单独虽更高，仍是预定诊断端点，不事后改主选条件。1189实体/全部非关系原字节保持；这不是原dev达标，也不能外推0.8。
- 原唯一完整80：79fresh＋idx19精确缓存1，80/80严格完整JSON与自然EOS、0整行弃答。wrapper **376.475652秒/rc0**、fresh生成347.081604秒；05:39:04.322610 A800_GPU2实际used1/free81156/无进程。原proc_cce6e47e已完成，不能重启。
- binder绑定240原始逐行证据后，两个完整pred共同freeze f0ccf03c，再各唯一v2、旧0；summary d2eddae4。独立统一实际终审d655/metadata修订8d19通过。303内容成员包d7cb2131（190297B）已在.100全成员SHA/fsync；8d19及scope66c5两小附录单独持久，不重打包。root已明确授权下述F0复核，仍不自动迁原dev/官方。
- 外部 `F=R/research/qwen35-sft32-full-report-feasibility/full80-v1`；[实验](experiments/E-20261024-SFT32-FULL-REPORT.md)、[关键结果指针](qwen35_sft32_full_report_readiness/full80/result-pointer.json)、[指标](qwen35_sft32_full_report_readiness/full80/quality-summary.json)。恢复只读原F1终点与job，不再start/评分。
- root已授权F0：原58/415CID/47PID/1166pairs、947实体，原SFT48 adapter8e32；同全文后缀。当前本地构造58全保，609passages/158markers/0skip、训练48与F0文献无交叉；原A800旧48SHM和.100原件实际SHA全等，不需搬运。native CPU实际7.478秒PASS（wrapper8.124s rc0），195355→505090tokens，max16245唯一idx30，工程cap16384；58行Auto/native/template IDs与双decode全核、不截断。原唯一最长probe proc_50f66d5a已PASS结束37.909523s rc0：16245→130，含prefill9.632724s，allocated22.753GB/reserved24.157GB；05:51:02.874170实际卡释放。全13终点SHA已核，原作业不重启。既定probe条件实际满足，完整58 source1100ee4c/planed429817已冻结，1200+30、idx30缓存1+57fresh，13件6小片实际SHA/fsync通过，唯一proc_76d0ce4c/PID20214/childPGID20224已05:53:31.239376启动，TERM06:13:31.239376+30秒；只读full58-v1/operations.py inspect，不重复start。两endpoint模板e48a97c1绑定实际probe及正式源且仍disabled，gold/质量未读。详见[事前执行政策](qwen35_sft32_full_report_readiness/f0/execution-policy.json)，两完整端点共冻各1v2/旧0，主Score升且macro不降。

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
- 最近已核Git远端 **98d450bec26b6750d2b5b6b50c375aee1600b3cc**（push+lsremote exact）。本次结案提交后另核，不为台账自身hash递归提交。
- 历史：[截至05:11台账](worklog/WORKLOG-through-20261024-0511.md)、[截至04:08](worklog/WORKLOG-through-20261024-0408.md)、[截至03:36](worklog/WORKLOG-through-20261024-0336.md)、[实验](experiments/)、[笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。
