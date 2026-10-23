# 本体规则工作台账

实验 `E-20261023-007`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：已复核 fold0 基线 .671522；CPU第一阶段14组无改进；第二阶段语义补充四内折均正增益，固定 dev16 最好 **.630237**（语义补充加文内缩写；原 .612597），仍未达0.8。首日本地 0.612597136，官方 0.603338039，两者不可混比。
- **记录入口**：[实验过程](experiments/E-20261023-007.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/`，完整日志、预测、模型不得入 Git。
- **下一步**：回收broad门控结果；Qwen模型就绪后在专属10.5.2.100 GPU0做零示例患者证据判断，先2窗口smoke再inner fold0。与CRF协调不同候选来源组合。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

运行状态：两批GPU语义候选作业已结束，GPU0复查无进程并交还CRF；当前无本路线GPU运行。关键产物 `/workspace/patientphex-runs/round2-20261023/ontology_rules/dev16-sapbert/augmented.jsonl`，来源与四折结果见[正式证据](evidence/round2/semantic-frozen-results.json)。
