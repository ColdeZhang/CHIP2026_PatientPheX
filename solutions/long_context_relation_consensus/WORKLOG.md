# 工作台账

当前：方案与源码冻结，尚未读取实际相交计数/新gold或运行质量。

- 目标：检验全文关系模型固定正票与既成G/S/Q/C一致时是否提高B4。
- 唯一公式、输入SHA与门：[plan-F1.json](plan-F1.json)。父线负并集不变；0模型/0fit/0remote，CPU≤120秒。
- 运行：`python solutions/long_context_relation_consensus/run.py contract`，合成通过及源推送后唯一 `run`。
- 产物：仓库外 `research/long-context-relation-consensus-v1/F1-v1`。有新增完整freeze后1直接v2；无新增byteexact继承0。
- 开始基点 main f960ae41；历史与不盲边界见[NOTES](NOTES.md)。
