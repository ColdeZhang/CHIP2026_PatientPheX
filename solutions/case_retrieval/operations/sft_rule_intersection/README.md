# 固定9B SFT与rules交集

root授权的一次F0 CPU开发控制，≤30秒。旧Qwen3-8B交集已负结案；本项使用不同的Qwen3.5-9B训练48结果63aefe68与同rules b44c10f1，主要比较本9B的.695443948，另列rules .711353522。

只在literal-v2原子集合上保留双方支持的关系，SFT所有非关系字段和患者条目元数据保持，非HPO原文/分号文本不规范化。8项纯函数契约覆盖空集、复合HPO部分交集及普通原文；完整预测freeze后唯一评测，保留标准为micro+macro之和提高且macro不降。无新训练/GPU/原dev、无新门控或阈值搜索，F0已用于开发。

运行：`timeout --signal=TERM --kill-after=5s 30s python solutions/case_retrieval/operations/sft_rule_intersection/run.py --plan solutions/case_retrieval/operations/sft_rule_intersection/plan.json`。数据位置及SHA见plan，输出在仓库外；源码/计划先提交推送再运行。

实际终点：运行前source/plan提交b3a32e4并push/核验，8合成契约通过。唯一0.184秒CPU输出261关系，较旧SFT删15；Score .695443948→.702672183（+.007228235），micro+.013048664、macro+.015864277，225TP36FP154FN。满足自身基线预定保留标准，仍低同实体rules .711353522。16篇/47患者/947实体及全部SFT非关系对象不变，格式0错警；独立增量审查待回收，无原dev迁移。完整来源与外部产物见../../evidence/round2/sft-rule-intersection-result.json。
