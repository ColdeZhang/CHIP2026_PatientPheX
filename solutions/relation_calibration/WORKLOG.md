# 工作台账

- 当前：**唯一 F1 元训练完成；等待 F0 三原始源完整，尚未读 F0 gold / 尚无 F0 评分。**
- 起点：main `a89f225d33a4facbd896e6c2d61dbef481d6c5ff`；分支 `explore/relation-calibration`。
- 来源：train32/F2F3 两生成器、CRF/Sap候选、grounded训练与严格词典均排 F0；公共41099规范化 bank 不含训练别名。实核小型元数据并引用三个已完独审，见 `evidence/ancestry.json`。
- 执行：源码/计划 `4b6e732b9a57ad1fb8a811f760b603876ee1ae4c` 已先推送并核远端；11:42:59 UTC 唯一 fit 结束，外1.27154秒/拟合.003523秒/rc0，11迭代、无warning、重开pickle概率exact。
- 输入：F1 16篇39患者；grounded282、sample交集383、Qwen3416，完整并集561；364正197负。特征共冻后才读F1训练标签，无 B1 限制/特征。
- 参数：C1/lbfgs/默认L2、截距、class_weight None、max_iter1000、seed20261024。系数 `[1.2821716714, 2.4158995729, .9640994501]`，截距 `−2.1711055903`；cut固定.5。
- 结果范围：这是元训练模式统计，无F1 v2/无held质量。固定模型接受sample交集存在，或grounded与Qwen同时存在；未据结果调整规则。八模式JSON应用与sklearn选择完全一致。
- 证据：`evidence/fit-result.json`；可直接恢复的小参数在 `evidence/model.json`、模式统计在 `evidence/patterns.json`。全量标签/特征/pickle/日志在 `R/research/relation-calibration-20261024/`。
- 下一步：按 `F0-plan.template.json` 绑定三完整F0输出及祖先 → 保B0全量装配 → root和固定2of3共同冻结 → 各唯一v2。F0 gold在共同冻结通知前禁止读取或哈希；必须严格胜B0和固定2of3。

主线程现已委托本owner接管两个F0方案的联合冻结与各唯一评价，无需再等待逐阶段批准。`joint_quality.py` 强制两完整预测/计划SHA及原947实体/315关系保护，先格式验证与共同冻结，再独立读gold；LR必须同时严格胜B0和固定2of3。固定2of3分支已追加同joint门（`0a432ef`），未改变其算法。当前仅sampled F0已完整，S1/S2全strict、交集280，原F1窄接口实际可用；另两source仍待完整核。F1拟合源码、参数与计划不修改；若实际F0出现文本/纯compound/明确弃答，再另建兼容层而不丢行或重fit。
