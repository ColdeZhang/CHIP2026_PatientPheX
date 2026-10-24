# 本体规则工作台账

本路线 `round2/ontology-rules`，worktree `/workspace/patientphex-worktrees/round2-ontology-rules`；只修改 `solutions/ontology_rules/`。新任务先读当前 main 的 AGENTS、HANDOFF.MD 与总台账，再核真实作业和 Git。

## 目标与可信基线

- 原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text，Score ≥ .8；目前未达标，不合并实验实现、不上传官方。
- 本路线原dev固定 BioBERT 迁移 **.738493753**，旧 .737585917；预测先冻、独审7336b280、0错警。[结果/恢复](evidence/round2/public-phenotagger-biobert-dev-results.json)。主线程组合最佳以根台账为准。
- train64强 BioBERT 四fold基线 **.730993616**，与原dev分数不同口径。每个独立模型≤10B，允许集成；旧全链上限已纠正。

## 当前：准确 R1 发布输入接口通过，质量未知

- 准确模型 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`，MS revision `5ee5fa4832e658ab84298806de06ed2a167834bd`，MIT；9资产16,388,603,517B全SHA、399 BF16 tensors / 8,190,735,360参数实际通过，唯一下载1862.200秒完整结束。[资产](deepseek_r1_readiness/assets/complete.json)。
- TF5 AutoTokenizer曾重建发布ByteLevel，原80 IDs全不同；旧1024/4096运行仅是该错误接口的工程记录。已用发布JSON generic逐行核80 raw/generic/template IDs、组件/特殊token/往返；新first2014、最长5689。原config不改，独立派生配置明确映射固定vLLM scale1。[纠正](deepseek_r1_readiness/ERRATA-TOKENIZER-20261024.md)、[当前报告](deepseek_r1_readiness/published-interface-v1/REPORT.md)。
- 正确first2014→746，自然EOS；原围栏使strict/sparse拒绝，旧结果不改。新固定codec只整段围栏/唯一整数P码表示转换，CPU5正23负及既有first746通过，8原键/2别名、0补键。
- 唯一新longest5689→2197，79.013秒/27.806 token/s，自然EOS，旧strict/sparse/新normalized均通过8/8，无需转换；wrapper92.494秒/rc0。04:04:37.441 GPU1实际0MiB/无目标进程。该短探针已结束；后继正式78新推断单独记在下方，无gold/质量结果。[终点清单](deepseek_r1_readiness/published-interface-v1/codec-longest-v1/terminal-manifest.json) SHA02986fde。

## 恢复位置与下一步

- 本地 `R/research/deepseek-r1-0528-qwen3-8b-feasibility`，`R=/workspace/patientphex-runs/round2-20261023`；大权重、提示、raw生成IDs/text在仓库外。
- .100 mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，自有 `/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/`；model在 `models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6`，新CPU/两次GPU在 `published-interface-v1/`。
- 已结束短探针job `proc_d63b7015-88a0-4da5-981e-cda0ee2ee160` 已观察running=false，实际终点在 `published-interface-v1/codec-longest-v1/gpu/recovered/`；只读恢复脚本 `poll_original.py`，不得重发start。
- 统一终审09612e0f通过；主线程已授权下一阶段同F1固定80（2缓存+78新）、逐row fresh同seed、10800+TERM30。[正式作业](deepseek_r1_readiness/full80-preparation/README.md) `proc_12e6dfa0-4f21-4f0a-a2d2-4ad35a5ab248` 已04:12:06.254启动，主截止07:12:06.254+TERM30；plan4a88bd33原留。CPU同入口0.084秒通过，原只读monitor遇RPC退出，已由 `full80-preparation/monitor_recovery.py`（local72050）接原job，不得重发start。固定standalone与唯一oldSFT32∪R1两臂，共同冻结后各唯一v2；失败行弃答不修补。

- 并行[F0准备](deepseek_r1_readiness/f0-preparation/README.md)：原58提示CPU已6.605秒通过，唯一最长index30/6028；.100→A800私有SHM九资产copy已04:55:02完整SHA/fsync，1797.002秒，原source止停。仅实际最长一条240+30已获工程授权，原plan609febf2在total/free混淆gate退出、0NN，原件保留；独立单字段修复plan a3745b9a已PASS：6028→3427／43.414tok/s、wrapper89.507秒，05:01:02已释放。新增后继条件：F1完整old∪R1通过原retain才启动6000+30/缓存idx30+57fresh；当前仍未触发，无F0质量。

## 最近结案与索引

- 强 BioBERT+NCR固定四fold **.730993616→.731229952**，仅2/4非降<事前3/4，关闭当前配方；mention净−7TP/−17FP、doc−5TP/−10FP，关系exact、0错警，不迁原dev/扫cut。60.879秒/rc0，独审de325a2c。[结案](experiments/E-20261024-PUBLIC-NCR-ADDITION.md)。
- NCR全cache408272键、两工程错误与CPU-only恢复均原留；65件关键预测/metrics/小HGB/source在.100自有 `research/ncr-strong-biobert-control-backup/20261024-terminal-v1/ncr-terminal-v1.tar.gz`，660803B，SHA bb9aa427，fsync/全成员通过，不含gold/大权重/大特征。
- 最近已核远端 **441de59b86c5adb36a85916799bdfcfc82cad92e**，push+lsremote exact；本次终点保存后另核，不为台账自身hash递归提交。
- 完整过程：[截至03:36旧台账](worklog/WORKLOG-through-20261024-0336.md)、[截至04:08旧台账](worklog/WORKLOG-through-20261024-0408.md)、[实验](experiments/)、[笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。
