# 同文同概念的患者多标签对比目标

目标是减少正确概念的错误患者归属。沿已执行geometry模型，唯一变量是原patient bag BCE改为concept×patient分组MLCCE：每概念负患者logsumexp(0,z)＋正患者logsumexp(0,−z)，平均全部概念。多个正患者全部保留；不做单患者softmax。masked bag不造负标签；只有masked的概念为零并保留在全C均值。

这是患者目标的分组与自然尺度变化，不称等尺度hard-negative消融。原全局GP损失和1:1、全篇图/全锚点几何/50参数新LR组、fresh24轮768updates/76warmup、双>0及固定规范化全部不变。原32/87与16/39完整数据复用，无新tokenize/probe/参数。

执行入口：`contracts.py --encoded EXISTING --output REPORT`（CPU无权重）；`driver.py --mode formal --plan plan.json --source SOURCE --encoded EXISTING --output NEW_RUN`。唯一900+TERM30覆盖全部训练/推断/规范化。原global/model/normalize实际代码引用见evidence/reused-actual-source.json，部署权重206张量CPUexact重开，不是full resume。

固定两完整端点先共冻后2v2/old0；强B .686338981272为主目标，同时只读报告geometry .676794409655和无几何24 .669185282495。强目标不胜即关闭，不扫分组/权重/轮数/阈值。原dev/官网不操作。

产物根：`/workspace/patientphex-runs/round2-20261023/research/patient-competition-loss-v1`。

已完整负结案：主.675529387933 < B.686338981272；独立new-only.564954220539。完整源/CPU/预测冻结/末权重证据见evidence与实验记录。
