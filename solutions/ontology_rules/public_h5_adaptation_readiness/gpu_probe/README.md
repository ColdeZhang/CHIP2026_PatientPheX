# 公开 PubMed 原生 H5：唯一 GPU 工程探针准备

**只完成本地源码、固定批次与部署包准备；未部署、未连接远端、未启动GPU。** 正式执行须等官方最高方案被接收，并由root按原排期启用。没有完整20步训练或held质量授权，不修改官方B、head-index、Qwen全文formal或原native文件。

数据继续原1252种train32输入 `9590645d`，目标/负例/primary/diagnostic合同不变。原CPU统一审 `2e71b793` 只证明准备，不能代替本次全encoder反向。

## 固定实际批次

seed20261024，按 `SHA256('20261024|epoch1|'+token_sha)` 排序，完整顺序和164个实际query SHA预先固定。

| 探针阶段 | 真实批次 | None目标 | 多等价原列目标 |
|---|---:|---:|---:|
| 同原权重推断合同＋第1次更新 | 原单轮首64 | 26 | 0 |
| 保留已建Adam后的第2次更新 | 原单轮第2个64 | 21 | 2 |
| 保留Adam后的第3次更新 | 原单轮最后36 | 18 | 2 |

跳过中间17个完整批次；尾批取样真实，但这里只是三次工程更新，不能称完成一轮。每行仍3×32，目标来自原准备，不读取held输入/标签。

## 最小工程合同

- 原TF2.12/Transformers4.30.1环境，唯一空A100 GPU2 UUID `2c22a952`、memory growth、float32；沿原TF32默认并记录实际值。启动前重新核UUID/空卡/输出盘与全部资产SHA，绝不把CPU fallback当通过。
- 原H5 `d58af55a` strict topology load；先NaN填充，再逐201变量与H5及既有覆盖SHA完全一致，122759794参数不改。只读原权重。
- 同一真实首64，原functional softmax与同图hidden→**同Dense kernel/bias**→softmax比较。先保存完整两数组及差异，再判 `rtol=1e−5, atol=1e−7, rowL1≤1e−4`、归一与argmax一致。原推断头/softmax不替换。
- 实际 `training=True` 全encoder/head前后向，逐行masked-logsumexp目标、batch均值；每步在Adam前保存与原逐行参考的差异、None/等价列损失、全部梯度有限性。行loss容差 `rtol=1e−5, atol=2e−6`。所有NaN/Inf可安全记入JSON，失败前不会因序列化丢证据。
- fresh Adam `lr=5e−6, β1=.9, β2=.999, ε=1e−7`，无warmup/裁剪/重置；三次实际更新。原输出不使用BERT pooler，准确两变量允许无梯度且必须不更新；其余199变量必须有有限梯度。核embedding、首/末encoder层及head实际更新，Adam399个状态变量在第一步后保持同一对象且全有限。
- 记录原TF allocator current/peak及逐阶段物理卡used/free观察，不把allocated冒称全部物理占用。首步含trace、第二fullbatch给稳态、tail含新shape trace；forward/backward计时含全梯度同步，Adam计时含全模型finite同步，重开hash审计另记。
- 前后核原H5及所有指定旧public文件字节/SHA未改。**不保存适配H5/optimizer checkpoint**，进程退出即丢弃三步状态；未来正式训练必须fresh原H5。

## 预算与入口

整体240秒＋TERM30，15秒控制收尾预留；fresh资产/空卡检查也消耗同一窗口。只终止自有子进程组。完整stdout/stderr、contract数组及阶段收据在新私有namespace，退出后记录真实卡释放。等待官方接收的时间不计作运行，也不自动启动。

`operations.py prepare` 仅生成本地包；已经执行，119502B、SHA `c1c06368`，7个部署成员重开相等。15项静态/已有数据检查通过，含批次来源、真实tail、None/等价列覆盖、NaN收据保存及远端命令语法。没有重新运行CPU数值合同或任何模型。控制接口 `deploy/start` 在MCP初始化前要求另存的真实activation与准确planSHA；当前只有全false模板，activation文件不存在。

若root以后启用，仅按 `deploy → start → status → collect` 进入这一唯一作业；未知启动结果先恢复原intent/job，不能重复start。代码/计划和所有比较容差已冻结，失败保留证据，不在此入口放宽门槛或扩多配方。完整20步＋原F1推断总预算仍需root根据实际三步成本/内存/加载与保存开销决定；本准备没有质量结论。

[计划](plan.json)、[批次](sampling.json)、[静态检查](static-checks.json)、[本地包指针](ready.json)。

## 接收后实际启动

官方accepted凭据8d39aba4已核；本独立唯一作业已启动，具体ID/时间见[实际启动](actual-start.json)。以上准备阶段记录保留；新实际结果另存，不重写事前门槛。当前还没有完整工程终点或质量结论。
