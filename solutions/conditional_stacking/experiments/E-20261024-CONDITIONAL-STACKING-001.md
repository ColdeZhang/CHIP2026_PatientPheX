# E-20261024-CONDITIONAL-STACKING-001

固定条件化 simplex 融合的唯一 CPU 控制。准备报告、公开成熟依据与全部祖先见 [REPORT](../REPORT.md)，原 prepared `7d53ce36` 保持不变；实际 source `14fca616`、授权绑定 `39d3c63d` 启动前已推送。

## 方法与执行

按 [plan.json](../plan.json) 原数学：三源二值专家、6参数softmax，x=(1,ln给定患者数,S1 xor S2)，按文献均值NLL求和＋.5θ²；全F1 rawunion561、364正/197负，未按B1筛选。C只作F0固定交集补充。唯一阈值>=.5，float64/L-BFGS-B/原参数，线程1；timeout60+5没有触发。没有重试或扫描。

精确启动命令在 [execution.json](../execution.json)；实际命令为 `python solutions/conditional_stacking/driver.py --activation solutions/conditional_stacking/execution.json`。全部运行产物位于 `/workspace/patientphex-runs/round2-20261023/research/conditional-stacking-v1/`。

整个控制 15:18:28.810645→15:18:29.812980 UTC。唯一fit .012403秒、9迭代/11函数及梯度调用，目标8.647475316→8.424861561；停止原因是固定ftol相对下降门，success=true、全有限，并非所有梯度已低于gtol。原系数和梯度见 [theta](../evidence/theta.json)、[convergence](../evidence/convergence.json)。参数与重载预测一致；15:18:29.289611参数冻结后才构造F0。

## 唯一F0结果

完整16文献/47患者/947实体、343关系，全部315条B0关系及非关系对象保留。15:18:29.627035先完整冻结，再于.752569完成唯一公共v2；1new/0old、F1v2=0、GPU=0。

| 指标 | 旧B4 | 固定条件融合 | Δ |
|---|---:|---:|---:|
| Score | .7281956289113009 | .7283349399548672 | +.0001393110435663 |
| menF1 | .724105461393597 | 相同 | 0 |
| docF1 | .7285714285714285 | 相同 | 0 |
| micF1 | .7340720221606648 | 相同 | 0 |
| macF1 | .726033603519513 | .7265908476937785 | +.0005572441742655 |

相对B4新增1=0TP/1FP，删除1=0TP/1FP，均在同一9患者文献，患者不同。新增bits010概率.517196096，删除bits101概率.482803904；两者disagreement都0。F1患者数最高7，因此这是患者数范围外推后的S权重过半，不能把分数变化归功于新增采样分歧信号。

形式上通过预先规定的严格正增益门；实际收益极小，没有新增正确关系，不足以证明普遍改进。未更改配方、未补做消融或其它阈值评分，未执行原dev/官方。完整预测SHA `478daa8840536117a84db5af73bef6b9dd674168317aa2da07be059d3f807427`，124397B。下一步仅由root决定，不自动扩大。
