# 文档级关系工作台账

- **当前状态：三组完整训练内 F1 控制已结案，固定零阈值的直接并集/候选内替换均未超过对照；没有上线新方案。** 不扩 F0/full64/dev，不扫描阈值或中间轮次。未来学习融合是新实验，须严格 train-only inner OOF；不能把本组负结果泛化成所有 DocRE 融合无效。
- 分支 `explore/document-relations`，worktree `/workspace/patientphex-worktrees/explore-document-relations`，起点 main `0c0672d`；只维护本方案目录。用户原固定dev≥.8的总目标未改变，本线未达到。
- 当前无GPU作业。最后唯一job `proc_3973bb9b-954c-4b6a-afff-3e71eb2f1baa` / PID1199462，30ep/240Adam+全部F1完成，639.318447秒rc0；10:29:54.550211实际GPU0回4MiB/noapps。root已将下一顺位交span四折补全，其后grounded必要Sap查询。
- 最近已核远端检查点 `7fae42925973b6c8bfa9c45acd43a2478ddace38`；本次结案提交将继续push。原 `evidence/long-resume.json` 是启动前历史记录，不能据它再start。

## 结果与决定

固定 F2+F3 train32→F1 16文献/39患者，公共v2/all/text，同1189实体；每run两完整预测共冻后各1v2，三run新6/旧0，无原dev质量/官方操作。

| 配置 | 并集Score | 候选内替换Score | 并集micF1/macF1 | 状态 |
|---|---:|---:|---:|---|
| 未标记span，8轮 | .645655645 | .534242627 | .628272/.539411 | 负 |
| 显式边界marker，8轮 | .663128103 | .558854734 | .667880/.569694 | 相对首版改善，仍负 |
| 同marker，固定30轮 | .669661581 | .631836310 | .679213/.584494 | 仍负，关闭本组 |

固定 **normal-fulltext N** 为 .676546355（mic .695279/mac .595968），不是所有历史F1流水线最高；历史 public-new LGB 完整组合 .686338981，已只读核同1189实体，但其支持gate训练48排F1/含F0，资源不同。30轮并集对两者分别−.006884773/−.016677400；不称新best。详见[比较](evidence/comparison.json)、[完整实验](experiments/E-20261024-DOCRE-001.md)。

30轮新增237条中73TP/164FP；候选内替换另丢88TP/62FP。候选覆盖469/520，不仅是候选缺漏。候选AP依次 .44811→.54738→.65608（30轮AUROC .88979，多患者AP .62049），有排序信号，但默认直接融合未过总Score门。完整所有候选logits已保留，不以AP代替赛事分数。

## 机制、隔离与恢复

- 独立实现ATLOP式全提及logsumexp、mention attention均值及pair交集context，完整重叠窗口覆盖全文；每物理提及只归一个完整窗口。共有attention为0时仅由两端entity表示分类，不冒称跨窗全局传播。
- 三public固定实体缓存提供候选，所有学习来源排F1，未把旧OOF但包含F1的模型用作特征。二元BCE/零logit阈值，不伪称多类ATLoss复现或人工evidence监督。
- marker后继只改已有词表`*`边界/起始marker表示，必要重分窗已披露；30轮只延长固定训练与相同比例LR日程，全部fresh公共base。参数110,277,377；源/资产SHA、CPU合同及命令见各evidence。
- 外部根 **D**=`/workspace/patientphex-runs/round2-20261023/stage8-document-relations`。核心结果：D/quality-v1-entry-recovery、quality-marker-v2、quality-marker-long-v3；原始全部2874分数分别D/collected/formal-v1、formal-marker-v2、formal-marker-long-v3/native/held-scores.jsonl。
- 30轮分数SHA `2933713266fcbaa433397a3f30350ca1e0473aa1d4c85f252a241e9c1b86520f`；完整checkpoint1,318,873,119B/SHA `67317bdd67d5fae26846806844edf59c5dc3ea4f2478f9a137fd9c8d1abe3142` 在`.3 research/stage8-document-relations/formal-marker-long-v3/native/checkpoint.pt`。模型/optimizer/scheduler/RNG持久；205模型张量重开精确，未逐tensor复核optimizer。完整12件回收身份见[long-runtime](evidence/long-runtime.json)。
- 融合候选支持：D/fusion-support/held-candidate-support.jsonl，1069概念/2180原提及，含原边界/text/三public来源，与logit stable key直接join；SHA f59270c6…。原输入patient元数据提供全部患者/提及。
- 两早期run的95件源码/预测/证据已从服务器实际回读逐SHA，见[two-control-backup](evidence/two-control-backup.json)。30轮补包另存，不覆盖旧包；大权重留远端，不进Git。

## 已处理的入口问题

1. 首次CPU编码误用1秒同步wrapper，wrapper状态丢失但原子进程已完整完成；恢复原encoded未重跑，后继均用managed start_process。精确证据见cpu-transport-recovery.json。
2. 首质量入口假设N预测含patient字段，在gold/冻结/v2前退出；固定按association IDs并与encoded输入核同39患者，保原失败源/空目录，只成功新2v2。后两run复用修正，不重复失误。
3. 一次备份验证早于异步下载子进程完成而见FileNotFoundError；等待原下载完成后实际逐SHA核准，没有重新下载，记录在two-control-backup.json。

## 同槽完成的协调任务

root另分配GP F0增量只读终审，已完成：`/workspace/patientphex-runs/round2-20261023/validation/span-oof-F0-final-review/review.json`，SHA `a3f855a0067d980087c0c968de4b871c38c0a6077ebca2add6c7d8122a2cfa69`。实际排F0/源链、四完整score、三fresh8ep与旧GP23的1735 raw交集exact、102→104接口、先冻1fit1v2/关系不变通过；质量 .740572140440→.739506259147负。该审未读/hash gold、未model/fit/v2/remote、未改owner/main，也不授full64。root和span已收到。
