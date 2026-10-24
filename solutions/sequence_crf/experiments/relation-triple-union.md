# 固定两折三方关系并集

在patient-first与逆序各自已有增量后，root另行授权唯一组合oldSFT∪reverse∪patient-first；这是事后提出的新控制，不能称原始预注册。先固定F1/F0所有输入SHA和门槛，再两fold完整pred共同freeze后各唯一公共v2，旧old∪reverse/rules指标复用。无训练、新GPU、第四臂、阈值扫描或旧预测重算，实体/非关联不变。

成功标准提前固定为两fold Score各严格高于对应old∪reverse且macro均不降，不要求micro单独升；强rules仍并列。

| 折 | 三方Score | 旧∪reverse | Δmicro | Δmacro | 新TP/FP |
|---|---:|---:|---:|---:|---:|
| F1 | .663841821 | .662472447 | −.007934237 | +.013411732 | 8 / 27 |
| F0 | .709556743 | .704687440 | +.005050889 | +.014426323 | 7 / 8 |

两fold按门槛通过，但收益小；F1微观下降、F0仍低rules .711353522（−.001796779），不掩盖限制。F0 patient来源为独立23缓存＋35新生成的有限弃答恢复，缺O7P的8对未人为补答案；原strict失败未评分并保留。

首次CPU0.114597秒在新pred/gold/评分前被列表顺序guard拒绝。实际F1 20处/F0 11处只有phenotype列表顺序差、集合差0；独立恢复只将该guard改成逐患者集合比较，原union、源输入、保留门槛不变。恢复0.364852秒，总.479448秒，保留原source/plan/trace；两foldcommonfreeze2c15823d后仅两新v2，旧0次。独审3444fe0a PASS，无重fit/predict/evaluate。

[完整短源/计划/结果](../research_readonly/relation-triple-union-control/README.md)；[终审](../evidence/round2/relation-triple-union-review.json)。全pred外部R/research/relation-triple-union-control/order-guard-recovery/assembled，F1ce7363ab/F0c46f4662。root据此授权full64的无goldCPU和最长240+30更新probe，正式训练仍待真实资源后统一放行；当前未迁dev/官方，未达.8。
