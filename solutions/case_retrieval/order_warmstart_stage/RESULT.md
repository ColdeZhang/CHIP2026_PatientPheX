# 共同旧权重起点的新两轮训练：完整负结果

2026-10-24 UTC，两臂完整训练、四次持久全状态导出和各80条推断结束；唯一F1开发对照总分均未提高，关闭本配置，不替换旧adapter、不迁F0/原dev、不扫描阈值或更多epoch。原dev≥0.8目标仍未达到。独立终点审核 `d253f625` 通过；既有入口与恢复审核保留。

## 同预算对照与结果

两臂都从严格train32旧SFT epoch2权重 `4f4815b6` 开始，新建相同AdamW、seed和两轮LR。控制O/O；增强R/O只在新第一轮把候选和对应CID/target共同逆序，第二轮回到原顺序。每轮124行且样本ID顺序相同，每臂248曝光、32 optimizer calls（含初始LR为0的调用）、14340监督tokens。完整tokens805188与805190；病例、患者、概念、标签语义不变。旧optimizer不继承；新stage自己的epoch1→2恢复完整Adam/scheduler/RNG。未另做跨进程数值等价实验，不能宣称bitwise连续训练。

| 同一F1开发16篇、1189实体 | Score | 关系TP/FP/FN | micro F1 | macro F1 |
|---|---:|---|---:|---:|
| 缓存旧2ep基线 | .660942003 | 310/73/210 | .686600221 | .542228853 |
| 旧权重＋新stage O/O | .658421563 | 299/86/221 | .660773481 | .557973834 |
| 旧权重＋新stage R/O | .645528610 | 287/70/233 | .654503991 | .512671513 |

O/O相对旧基线Score−.002520440，正确关系少11、误报多13；macro虽升.015745，micro下降.025827，总分仍负。这说明本次新增训练阶段没有收益，不能将下降单独归因于额外epoch或直接断言过拟合，因为新阶段也重置了优化器、RNG和LR。

R/O相比匹配O/O再降.012892953，少16误报、也少12正确关系，macro下降.045302。这支持关闭本次固定顺序增强配方，不能推广为所有顺序增强无效。此前固定逆序**推断**并集在原dev的正结果仍然有效；训练排列和测试时排列是不同变量，本次没有重评或替换旧推断并集。

两臂都是80/80完整响应、587/587合法CID、0未解；1189实体及所有非关系字段原样保留，因此不是格式失败导致的质量负结果。双完整预测于02:25:55.340Z共同冻结 `e35f9c3a` 后，每个新臂只运行一次既定公共v2 all/text；旧指标直接复用，无新raw质量臂、union或阈值搜索。这是反复观察的F1开发折，不是独立泛化、原dev或官方分数。

## 成本、故障与资源终点

共享原3900秒＋TERM30预算，开始01:24:11.579Z、原截止02:29:11.579Z。全部神经计算和四次导出在02:25:21.124Z完成，实际3669.545秒。O/O两轮epoch合1123.041秒；R/O合1124.733秒；各自原80推断280.585/279.659秒。四份完整adapter＋Adam/scheduler/RNG及源/manifest共2,079,864,798B，实际传输233.852秒，均在.100逐文件size/SHA、fsync与回读确认。完整控制耗时还包含加载、校验、网络及恢复，不能拿epoch合计替代总预算。

初次O/O第二轮恢复因`epoch-001-export-ack.json`被旧`glob('epoch-*')`误收，在base加载前退出、没有新更新。仅更改controller两处ack basename，旧trainer、数据、epoch1全状态和原绝对截止均不变；旧失败源码/回执和实际3例CPU反例保留，恢复增量独审 `baa8be3b` 通过。原epoch1位于`formal/`，后续三份checkpoint在`formal-ack-recovery/`，最终评测计划显式列出这条历史，没有把旧产物伪装成新目录生成。

末次推断 `proc_8c15b053` 的实际managed状态为不再运行，MCP没有提供远端模型exit code；本地controller和endpoint均rc0。02:25:04.530Z真实GPU2空卡收据 `856903f5` 已交sequence用于其独立患者优先实验；这是当时的释放事实，不代表当前仍空闲。本路线不再使用GPU、不删除旧或新checkpoint。

## 接手与证据

- [结果索引](../evidence/round2/order-warmstart-paired-result.json)：指标、差值、所有4份持久checkpoint locator/size/SHA及预算。
- [唯一质量summary](formal-ack-recovery/paired-evaluation/summary.json)：SHA `32690b52`；[共同freeze](formal-ack-recovery/paired-evaluation/paired-predictions-freeze.json)；[完整历史评测计划](formal-ack-recovery/paired-evaluation/run-plan.json)。
- [最终状态](formal/execution-state.json)：原controller已被恢复入口替代且两者都已结束。禁止重启任何controller或评测入口；原始状态快照不变。
- 外部完整产物：`/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry/formal-ack-recovery/paired-evaluation/`。O/O完整pred `c45c3920`、R/O `f1d9e571`；原80raw分别`c398d637`/`a5849b49`。原始预测、响应、日志与权重留仓库外，Git只保存源、小manifest和结果。
- .100持久mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，相对目录 `case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/{original,reverse-first}/epoch-00{1,2}`。A800 SHM不是唯一备份；未清理任何新checkpoint。

独立终点审核以101个小源/收据快照复核4 fullstate ACK与已有持久回读、epoch fingerprint、双raw覆盖、实体对象与共同freeze/唯一v2来源；未再次读gold、运行parser/v2/NN或重新下载2.08GB权重。见[终审](terminal-review/review.json)。随后双raw/fullpred及小manifest共31个业务成员453,591B另存.100自有58,944B归档，02:31:28.789Z实际服务器逐成员size/SHA与fsync通过，SHA `2dd8bfc0`；[备份收据](terminal-backup/receipt.json)、[最终结案](formal-ack-recovery/terminal-closeout.json)。较早terminal-result快照中的“review pending”保留当时事实，不表示当前未审。
