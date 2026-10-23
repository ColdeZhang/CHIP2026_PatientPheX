# 固定9B SFT与rules交集

root授权的一次F0 CPU开发控制，≤30秒。旧Qwen3-8B交集已负结案；本项使用不同的Qwen3.5-9B训练48结果63aefe68与同rules b44c10f1，主要比较本9B的.695443948，另列rules .711353522。

只在literal-v2原子集合上保留双方支持的关系，SFT所有非关系字段和患者条目元数据保持，非HPO原文/分号文本不规范化。8项纯函数契约覆盖空集、复合HPO部分交集及普通原文；完整预测freeze后唯一评测，保留标准为micro+macro之和提高且macro不降。无新训练/GPU/原dev、无新门控或阈值搜索，F0已用于开发。

运行：`timeout --signal=TERM --kill-after=5s 30s python solutions/case_retrieval/operations/sft_rule_intersection/run.py --plan solutions/case_retrieval/operations/sft_rule_intersection/plan.json`。数据位置及SHA见plan，输出在仓库外；源码/计划先提交推送再运行。
