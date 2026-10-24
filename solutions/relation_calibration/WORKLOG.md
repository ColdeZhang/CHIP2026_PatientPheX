# 工作台账

- 当前：来源核查通过，源码和唯一拟合计划待推送后执行。未 fit、未读 F1/F0 gold、无新质量。
- 起点：main `a89f225d33a4facbd896e6c2d61dbef481d6c5ff`；独立分支 `explore/relation-calibration`。
- 核查：train32/F2F3 两生成器、CRF/Sap候选、grounded训练与严格词典均排 F0；公共41099规范化 bank 不含训练别名。复用既有三个完整端点独审，实核相关小型来源元数据。见 `evidence/provenance.json`。
- 纯输入预检：F1 16篇39患者；grounded282、sample交集383、Qwen3416，完整并集561；无 B1 行限制或特征。
- 配方：sklearn1.8.0 LR，C1/lbfgs/默认L2/intercept/class_weight None/max_iter1000/random_state20261024；3binary、cut.5、唯一 fit。
- 下一步：提交推送核验 → 唯一 CPU fit → 持久参数/模式统计 → 等三个F0纯输出 → 和主线程固定2of3共同冻结 → 各唯一v2。F0 gold在共同冻结通知前禁止读取或哈希。
