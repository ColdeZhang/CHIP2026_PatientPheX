# 原 dev16 的一次固定支持替换

状态：源/计划已先76ba77a发布，唯一控制已完成并由5c043574统一终审通过；当前best .7542169692819372，见RESULT.md。原冻结plan.json字节不改，不重启。

起点是预测 `0e2c26f8…`，Score `0.748583439566388`：16 篇、36 位患者、1061 个实体、391 条关系。旧 public-new 使用原有 42 条提示、276 个新增文档概念，留下 145 条正关系缓存，其中旧支持筛选保留 35 条。库存已逐集合确认：这 35 条恰好等于旧 supported368 减 baseline333，也恰好等于当前391与原 raw145 的交集。其余356条包含原333条和旧逆序候选增加的23条，全部保护。

唯一变化是用当前已冻结 full64 LGB103/nested 实体输出的文档概念集合筛选同一 raw145。删除原35中不再被支持的键，接受的旧键保持原位置，新增被支持键按固定排序追加；其余356条的值和顺序不变。实体及所有非 `association` 字段的原始 UTF-8 字节保持不变。复用原响应及其解析键，不重跑候选、SFT、LGB、阈值或解码；不修改 sequence 的 a179∪patient 预测。

先完成整份16篇预测及格式验证，写入 `complete-freeze.json`，之后才核验并读取 gold，调用一次标准 `evaluate_report(..., 'all', 'text')`。标准报告固有的固定敏感性结果保留，主判据只使用默认 Score。旧最佳指标原样复用，不重评；新 Score 严格大于 `0.748583439566388` 才成为候选最佳，四项分数和计数无论正负都报告。目标0.8单独记录，不改变保留判据。

full64 LGB/SFT 的训练清单均排除原dev16；但历史方法选择已使用原dev，本次属于开发集迁移，不能称盲测。实体支持沿用 pooled64 选定的 cut .3，LGB、BioBERT候选与nested解码共同构成支持栈，不能将效果全部归因于分类器替换。前置F0/F1控制分别增加 Score `0.005790432342576257` 和 `0.004210176873942872`，两个fold的macro也增加。

执行入口（主线程完成发布并通知后，仅一次）：

```bash
python solutions/case_retrieval/public_new_lgb_support/full64/run_once.py \
  --plan solutions/case_retrieval/public_new_lgb_support/full64/plan.json
```

输出固定到仓库外 `research/public-new-lgb-support-control-20261024/full64-actual-v1`，30秒加1秒TERM宽限。目录必须不存在；失败保留现场，不重试或更换策略。既有两fold装配和两份报告实际耗时0.265秒，本次无需加载模型。

入口索引在 `entry-pointer.json`；完整来源、当前391与原35的集合证据在 `source-inventory.json`；固定路径和SHA在 `plan.json`。旧 `metrics.json` 没有预测SHA字段，其归属通过已发布的 `result.json`、预测冻结及终审连接，不虚构该字段。准备期间gold仅复制既有计划中的路径/SHA并核文件大小，未打开或重新计算gold SHA。

源检查已通过：唯一一次合成替换检查覆盖CRLF、非关系字节、保护顺序及非法键拒绝；静态检查发现一个标准报告调用，无模型fit/predict/NN调用。详见 `static-contract.json`。准备过程没有产生新支持集合或预测结果。
