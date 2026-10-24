# MedGemma 关系控制：当前结论索引

短输入与闭集全文两个固定配方均已完成、均负；没有活动GPU或后继训练。强对照是旧SFT32在反复观察的内部F1上的Score .660942003。

| 已冻结控制 | 单独Med | 预先唯一旧SFT32并集 | 决定 |
|---|---:|---:|---|
| 原短输入fresh32×2ep | .603701549 | .644663430 | 负结案，不迁F0/dev |
| train+infer追加允许全文fresh32×2ep | .592418621 | .627966889 | 负结案，不迁F0/dev |

旧短完整结果在`RESULT-short-input-frozen.md`，全文完整结果在`fullcontext-preparation/RESULT-fullcontext-frozen.md`；各自source/plan、终点manifest与全状态目录分开，不改写旧结论。全文新增关联被大量FP主导：并集新增193=28TP165FP；80/587CID格式与实体对象完全有效，非关联行bytes不变。

两配置都保留每epoch完整模型+Adam/RNG的.100持久ACK/回读。短raw/fullpred归档b66fd723，全文归档07b2ca5a；对应`short-result-backup/actual-persistent.json`与`fullcontext-preparation/result-backup/actual-persistent.json`。完整raw与预测不入Git。

当前不支持替换旧9B基线或迁移这些Med配方到F0/原dev/官方；未测试的其他基座/schema不受此固定负结果一概否定。实时总目标与新任务以本路线WORKLOG和主仓总台账为准。
