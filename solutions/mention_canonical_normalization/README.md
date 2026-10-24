# 固定 mention 的标准名称生成

目标：以现有Qwen3-8B监督生成公开canonical名称／原文NIL／NO，检验固定跨度池的规范化与新增实体供给。旧逐窗quote抽取已实际训评，不重做；旧GLiNER配方完整负结案保持。

CPU合同及真实GPU14FB/Adam/三状态约束探针已完整通过。原2400秒训练成本门关闭；root按实测3654秒批准新formal-recovery-v1的4800+30完整训练预算，不变更数据/配方，不重探针。训练全部train32的2353目标；F1输入固定强8cb的跨度∪GLiNER接受886实体的跨度，去重1416行。数据/端点决定见[plan.json](plan.json)，状态见[WORKLOG](WORKLOG.md)。历史CPU报告的“只替旧ID”建议被root固定并集/替换双端点明确取代。
