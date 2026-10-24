# 本体规则工作台账

本路线 `round2/ontology-rules`，worktree `/workspace/patientphex-worktrees/round2-ontology-rules`，仅修改 `solutions/ontology_rules/`。新任务先读当前 main 的 AGENTS、HANDOFF.MD 与总台账；此文件是当前索引，历史状态以时间及对应终点证据为准。

## 目标与可信基线

- 用户验收：原 train64/dev16 文献隔离、公共 evaluator v2 默认 all/text，Score ≥ .8；目前未达标，不合并实验实现、不上传官方。
- 本路线最新原dev固定 BioBERT 迁移 **.738493753**（旧 .737585917，Δ+.000907835）；完整预测先冻结，统一独审7336b280通过，0错警。[结果/恢复](evidence/round2/public-phenotagger-biobert-dev-results.json)。主线程组合的其他最高分以根台账为准，不混成本路线新实测。
- train64内强 BioBERT 四fold基线 **.730993616**，与原dev分数不同口径。[实验](experiments/E-20261023-PUBLIC-PHENO-BIOBERT.md)。

## 当前：准确 R1 资产与短格式成本探针

- 模型 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`，ModelScope revision `5ee5fa4832e658ab84298806de06ed2a167834bd`，MIT；实际头399 BF16 tensors / 8,190,735,360参数。允许多模型集成，每个独立模型≤10B，不能再套旧全链上限。
- **资产已完整通过**：.100 mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，原managed `proc_1c596785-307e-4471-8541-c982943da8bb` 03:36:58结束，1862.200秒、9/9全SHA、16,388,603,517B、399实际头/EOF通过；running=false。未重试/延期，原03:40:49+TERM30预算保留。[完整收据](deepseek_r1_readiness/assets/complete.json)。
- 外部目录 `R/research/deepseek-r1-0528-qwen3-8b-feasibility`，其中 `R=/workspace/patientphex-runs/round2-20261023`。恢复先核 `assets/intent.json`、`download-job.json`、`monitor-state.json`；只读入口 `assets/poll_original.py`。UNKNOWN先核原job/实际文件，不能再发start。
- 原生80纯输入CPU完成，三固定row输入2171/3425/6002 token；首BatchEncoding错误与修复原留。YaRN CPU对HF4.51参考64值bitexact，配置不改。[完整CPU清单](deepseek_r1_readiness/cpu-terminal-manifest.json)。
- **唯一GPU探针已结束并释放**：.100 GPU1，managed `proc_2eb8109a-602c-4714-956c-1aa4db401a35`；wrapper46.141秒/rc0，加载6.714秒/399实际参数完整。first2171输入→1024新token，33.219秒/30.826 token/s，无EOS，首token cap停止余2条；峰allocated19.496GB。03:39:03.804目标卡0MiB且无进程。仅当前预算不够，无gold/80质量，不关闭模型质量方向。[实际摘要](deepseek_r1_readiness/gpu-probe/terminal-summary.json)。
- **后继单条4096诊断正在运行**：主线程据原30.826 token/s授权同first/index0/2171输入fresh生成，max4096、总240+TERM30（225s软停），不续旧KV/RNG、不余2条/全80/gold。独立新源9c6020a2、plan918da954；proc_3150d10e-0826-45f1-adfe-557563152adb于03:43:18.247启动，主截止03:47:18.247+TERM30，freshGPU1为0MiB/free24164；再次cap/wall即止。[冻结计划](deepseek_r1_readiness/gpu-probe-single4096/plan.json)。
- 原生FP32 rotary同device比较及CPU/GPU 8eps容差在启动前冻结，原CPU证据保留；source实际SHA/compile通过。执行恢复入口为外部 `gpu-probe/{plan.json,start-intent.json,job.json}` 与 `gpu-probe/recovered/wrapper-result.json`，实际生成结果在 `recovered/`；只有真实产生后才作为已运行证据。[研究/过程/边界](deepseek_r1_readiness/REPORT.md)。

## 最近结案

- **强 BioBERT + NCR 固定四fold未满足保留条件**：.730993616→.731229952（+.000236335），仅2/4折非降<事前3/4；mention净−7TP/−17FP、doc−5TP/−10FP，关系exact、0错警。60.879秒/rc0，统一终审de325a2c通过。关闭当前配方，不原dev/扫cut；不泛化所有NCR用法。[完整结案](experiments/E-20261024-PUBLIC-NCR-ADDITION.md)。
- NCR资产、原生CPU/GPU接口、408272键全cache均完整核验；两工程异常和末TF32未测保留，CPU-only fullSHA/5341跨度绑定恢复可用。旧强制去B方案在fit前已superseded。[缓存恢复](public_neuralcr_readiness/full-train64-cache-attempt2/README.md)、[接口终点](public_neuralcr_readiness/gpu-cost-probe-attempt2/REPORT.md)。
- NCR 65件关键预测/metrics/四小模型/source manifests已实际持久备份至.100自有 `research/ncr-strong-biobert-control-backup/20261024-terminal-v1/ncr-terminal-v1.tar.gz`：660803B，SHA `bb9aa427b3fc28d81bc86e9103c818d5ecc7ee5fcd56b270d3a54dd7041adf02`，fsync/全65成员SHA通过；无gold/NN权重/大特征。

## 下一步与备份

1. 已冻结完整资产/CPU/GPU合一清单 `d172a788`，统一增量复核8fa2f637通过；下载已完，当前GPU仅后继4096单条原job。
2. 按主线程新授权执行唯一首条4096诊断；已有1024阶段已独审8fa2f637通过。完整80质量仍未授权。
3. 完整原始生成IDs及资产位于.100持久自有namespace，文件SHA/恢复见[路线报告](deepseek_r1_readiness/REPORT.md)和[合一清单](deepseek_r1_readiness/terminal-manifest.json)。

最近已核远端：`1c8703f`，push与lsremote exact（R1实际运行job/空卡检查）。大文件在外部/持久服务器，不入Git。更早各公开模型、链接、NER与关系正负实验见[原台账完整历史](worklog/WORKLOG-through-20261024-0336.md)、[实验目录](experiments/)、[笔记](NOTES.md)与[证据索引](evidence/round2/README.md)。
