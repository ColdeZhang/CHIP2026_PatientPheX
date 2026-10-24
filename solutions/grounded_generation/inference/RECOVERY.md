# 原作业与后继恢复

当前真实作业和绝对截止以 `../evidence/formal-started.json` 为准，恢复时先查外部 `formal-v1/started.json`、`start-intent.json` 和 watcher 已存终态；已有intent绝不再start。原source/plan/部署已冻结，watcher只有readonly观察。

工作目录为本worktree，`B=/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1`。这些命令列出依赖顺序，不能越过实际成功门：

1. 原formal结束后一次 `training/formal_collect.py --directory B/formal-v1`，全包逐SHA，两个完整durability ACK/model/optimizer/scheduler/RNG必须成功。失败保留原证据，不自动重训。
2. `inference/bind.py`只绑定实际epoch2与原native39输入；检查点保存/推送后 `inference/remote.py pack --plan .../infer-plan.json --directory B/infer-F1-v1`，再deploy/唯一start。独立1200+TERM30，实际GPU/磁盘门；已有intent只observe。
3. complete39后 `inference/collect.py --directory B/infer-F1-v1` 回收所有原IDs/raw/end状态。`inference/bind_flow.py extraction`生成实际提取计划，推送后用 `python -m solutions.grounded_generation.inference.extract_quotes --plan .../extraction-plan.json`，输出B/F1-quotes-v1；无NN重生/补答案。
4. `normalization/held_prepare.py --extraction-manifest B/F1-quotes-v1/manifest.json --output-dir B/F1-normalization-preparation`仅同32/public严格映射与完整必要query集合。若query=0不启动GPU；否则实际DocRE30释放.3之后，用 `held_bind.py --preparation-manifest ... --release-evidence 已存真实release文件` 绑定独立180+TERM30，源/计划推送后 `held_remote.py pack/deploy/start --directory B/normalization-held-F1-v1`；冻结公共41099bank不重编码，超过预写成本界限先报root，不截集合。
5. 必要query完整终态以 `held_collect.py --directory B/normalization-held-F1-v1` 回收原scores；`inference/bind_flow.py normalization-finalize` 绑定，推送后 `normalization/held_finalize.py --plan .../held-finalize-plan.json` 调原政策一次，输出B/F1-normalized-v1。
6. `inference/bind_flow.py quality` 只绑定八件真实完整ref，推送后 `quality/assemble.py --plan .../quality/plan.json`，两完整预测共同冻结。最后 `quality/score.py --plan .../quality/plan.json` 独立30+TERM1、两个唯一v2、旧0，主/诊断最高且严格超过N才候选；不自动F0/原dev/官网。

以上`...`须采用本方案目录的实际路径，`B/`是本文标记、不能原样传shell。所有阶段遇UNKNOWN先库存原件、续原包；保留原时钟和失败，不借新namespace静默重模型。各阶段partial结果不冒充完整结果。

当前39推断已唯一启动，真实PID/时钟见`../evidence/inference-started.json`；先查该记录和外部`infer-F1-v1/watch/`，现有watcher活着时不要再起。若已明确watcher退出且原模型未终态，`python inference/watch.py --directory B/infer-F1-v1`只接管读取，永不start模型；remote driver和1200秒时钟独立于本地watch。
