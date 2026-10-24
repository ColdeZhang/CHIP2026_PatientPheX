# 原JSON关系候选固定逆序：仅CPU可行性

这是共享9B/SFT32原80提示的新输入控制，不使用刚关闭的检索示例提示。检索负责人确认此前没有同80的候选逆序/一致性组合；旧530条件枚举保持原输入和前序，不等价。机制假设仅为自回归多答案可能受前面输出/候选顺序影响，未把该假设当质量证据。

唯一固定规则：每条完整候选列表逆序，并将原CID序列依次分配给逆序后的候选，同步code_map。仅改每个JSON对象顶层code值，保留所有其他原字节；system、患者/全局anchors、所有候选证据及HPO意义不变。双逆后完整row及user字节还原。结构化JSON边界定位，原文中的C000、C010及嵌套字面code文本不会被替换。无候选新增、移除、换词、ICL或随机排列。

实际CPU1.303秒：80条/16篇/587概念，78条变序、2条单候选不变；6个空/单/双/奇数/8/11概念与嵌套文字合同通过。真实固定tokenizer旧80长度精确复现；新总250691（旧250696），9行token数有微小变化，最大5691，预留1600后7291，0截断。每行CID双向映射及完整渲染SHA外存。

原80同A800耗282.502秒；按input+旧output token代理282.496秒，只是预算估计。后续若root授权，可用原SFT32公开base/adapter与原runner单次推断，预算最多600+30；不改训练/decoder。尚未GPU、未读新标签、未产预测/分数。

历史未做原hard与逆序输出的一致性组合。如以后预先授权，一个固定正关系交集可只保留两次都支持的患者-HPO，但可能损失召回；当前只提案，未生成该臂或按F1结果选规则。原hard实体和所有非association字段必须保持。所有比较均是反复开发的训练内F1，不是盲泛化或原dev。

[冻结计划](../evidence/round2/reverse-json-order-plan.json)、[实际CPU/源/所有产物SHA](../evidence/round2/reverse-json-order-readiness.json)。本方案源码 `reverse_json_order/prepare.py`，外部 `research/reverse-json-candidate-order/`。

后续root已授权唯一同A800/SFT32逆序80推断600+30。最终预定义主臂为旧hard OR 逆序完整关系并集（改善候选内FN），逆序alone仅顺序依赖对照；不运行前述intersection提议。字面v2键：只有完整HPO compound拆原子，NIL、大小写、非HPO和含分号原文保持原字符串；旧关联列表/对象字段原样保留，仅append新集合差。合同覆盖compound/NIL/字面分号/重复/患者与非association边界。两个新fullpred统一freeze后各一次v2，keep=union Score严格升且macro不降，不改组合/门槛；已评原.660942基线仅复用。
