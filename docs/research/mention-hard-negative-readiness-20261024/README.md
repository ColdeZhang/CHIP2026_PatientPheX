# 固定片段规范化：新增拒绝监督准备

原报告和census保留研究时点原字节；当前2353行正式训练保持不变。Root另批准独立分支 `explore/mention-hard-negatives`（起点23b053e6）的CPU准备，负责人 `/root/schema_extraction`，方案目录 `solutions/mention_hard_negatives`。

完整保留2353原目标，每文追加 `ceil((HPO行数+NIL行数)/4)` 条与所有注释零重叠的OOF候选；按probability降序、similarity降序、offset升序、length升序固定选择，不足取全部并披露。1:4是在成本内适度增加拒绝监督的首个固定配方，不称最优，不来自held结果，也不搜索比例。正span错HPO、NIL、分支外正注释和边界重叠不当负例。新增空数组目标只表示标注意义负例，仍有漏标风险。

新变量仅训练支持集；三状态prompt、上下文、语法、模型和训练入口复用原线。先实际CPU核tokens/形状/完整祖先和更新数；尚无新GPU训练或质量。F1仍为原1416固定跨度池，两完整端点共同冻结后再各一次v2。GPU预算与资源待root根据实测成本及当前结果安排；旧2353实验及其结果独立保留。
