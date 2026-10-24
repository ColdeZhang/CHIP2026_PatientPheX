# 非思考双种子共识台账

## 当前状态

- 固定主配方 `B ∪ (S1 ∩ S2)` 内部F1 **.686338981→.693653304**（+.007314）、内部F0 **.711507580→.721969518**（+.010462）均正；每折 micro/macro 同升，实体保持各自强B。原dev最佳901及官方成绩未更新。
- F1完整160已终审PASS `e5437e12…`，原替换型多数诊断负保留。F0完整116已807.028秒rc0并于11:48:37.791245Z释放 .100 GPU1，全部EOS/strict，375成员原包完整恢复。F0唯一主端点冻结后仅1次v2，独立终审/小证据备份正在收尾。
- F0预测`3ec75365…`保947实体/全部B315（池外39），固定新增21=9TP12FP，0删除；分项和可复现入口见[实验002](experiments/E-20261024-SAMPLE-002.md)。完整原始两seed已交root固定融合，不按逐项正确性选择。
- 条件full64 CPU准备已完成[说明](full64_readiness/README.md)：原normal64短提示不能冒称全文，新完整64每轮512091 tokens/max13230，原full64适配器1a703b已实核，无需重训。仅CPU，无原devgold/NN/quality；是否运行由root后续决定。
- 分支`explore/sampled-consensus`；最新已核远端`8f07116`（完整F0冻结、唯一评分前），后续结果提交正常续推。全量产物在`R/research/nonthinking-sampled-consensus-native-20261024/`及`R/research/nonthinking-sampled-consensus-F0-20261024/`，Git仅源码/短记录/manifest。

## 执行与恢复

| 控制 | 已完成与决定 | 证据 |
|---|---|---|
| [SAMPLE-001](experiments/E-20261024-SAMPLE-001.md) | A800同native4f48、80×2；主正、替换多数负；完整终审PASS、两个包全读回 | `evidence/result.json`、外部F1 terminal-manifest.json；终审e5437e12 |
| [SAMPLE-002](experiments/E-20261024-SAMPLE-002.md) | A30同native4f48、58×2、每seed最长首行属于正式116；完整主正；无新训练 | `evidence/F0-result.json`、外部F0 terminal-manifest.json；终审待收 |
| full64条件准备 | 全祖先排原dev，纯输入完整文本构造+真实CPU token合同；首脚本命名遮蔽stdlib已独立恢复 | `evidence/full64-readiness.json`；原始失败与恢复都保留 |

F0原作业 `proc_972436a9-26f2-4af2-91d7-87c81710dc58` / PID2734284 已完成，不重启。原TERM12:10:10.631670Z/KILL12:10:40.631670Z保留；`monitor_once.py`仅用于必要原终态恢复，当前已持有真实release。CPU装配`f0/assemble.py`、终态核验`f0/verify_terminal.py`、唯一质量`f0/quality.py`，实际命令见外部assembly-intent/quality-command。所有新GPU、原dev质量或官方动作均需root给新任务，不能从readiness自动启动。

## 关键边界

F1候选祖先训练32、F0复用候选训练48，均排目标折；生成模型始终同F2+F3训练32。F0换A30但native源码/版本/参数不变，不声称跨硬件逐位等价。保护强B的固定追加与替换型投票不同；两个正结果支持本配方，不证明全部采样/联合方案有效。失败、零收益和单独的条件部署准备保持可追溯。
