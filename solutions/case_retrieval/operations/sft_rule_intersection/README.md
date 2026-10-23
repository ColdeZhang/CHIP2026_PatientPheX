# 固定9B SFT与rules交集

root授权的一次F0 CPU开发控制，≤30秒。旧Qwen3-8B交集已负结案；本项使用不同的Qwen3.5-9B训练48结果63aefe68与同rules b44c10f1，主要比较本9B的.695443948，另列rules .711353522。

只在literal-v2原子集合上保留双方支持的关系，SFT所有非关系字段和患者条目元数据保持，非HPO原文/分号文本不规范化。8项纯函数契约覆盖空集、复合HPO部分交集及普通原文；完整预测freeze后唯一评测，保留标准为micro+macro之和提高且macro不降。无新训练/GPU/原dev、无新门控或阈值搜索，F0已用于开发。

运行：`timeout --signal=TERM --kill-after=5s 30s python solutions/case_retrieval/operations/sft_rule_intersection/run.py --plan solutions/case_retrieval/operations/sft_rule_intersection/plan.json`。数据位置及SHA见plan，输出在仓库外；源码/计划先提交推送再运行。

实际终点：运行前source/plan提交b3a32e4并push/核验，8合成契约通过。唯一0.184秒CPU输出261关系，较旧SFT删15；Score .695443948→.702672183（+.007228235），micro+.013048664、macro+.015864277，225TP36FP154FN。满足自身基线预定保留标准，仍低同实体rules .711353522。16篇/47患者/947实体及全部SFT非关系对象不变，格式0错警；独立增量审查待回收，无原dev迁移。完整来源与外部产物见../../evidence/round2/sft-rule-intersection-result.json。

## Fixed F1 transfer (closed)

The pre-pushed F1 source/plan commit `4cfe7aa18fef81fa31e4e087e230129505dcb3eb` applied the same literal intersection only to the 383 old SFT relations: 320 survived the 483-rule set. All 65 previously supported public additions were preserved unchanged, giving 385 relations instead of 448. Complete prediction freeze preceded the sole all/text v2 evaluation. CPU elapsed 0.205 s; outer timeout returned 0; format errors/warnings 0.

Score fell from 0.682128804 to **0.661533824** (−0.020594980); micro F1 −0.037913794 and macro F1 −0.044466127. The predeclared criterion failed. Despite the F0 gain, these development-fold results do **not** support original-dev transfer; close this fixed policy without new gates or threshold selection. F0 and F1 have both been used repeatedly during development; the existing support cut was developed on pooled64. No original-dev was read and no official submission was made.

Small evidence: `../../evidence/round2/sft-rule-intersection-F1-result.json`. External summary and full frozen artifacts: `/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/sft-rule-intersection-F1/`. Source/input-only rule replay and the actual F1 invocation receipt: `/workspace/patientphex-runs/round2-20261023/research/sft-rule-intersection-F1-preparation/`.
