# 本体规则工作台账

实验 `E-20261023-007`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：已复核 fold0 基线 .671522；CPU第一阶段14组完成，最好增强 .668950，未超原基线 .671522；GPU语义候选作业运行中。首日本地 0.612597136，官方 0.603338039，两者不可混比。
- **记录入口**：[实验过程](experiments/E-20261023-007.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/`，完整日志、预测、模型不得入 Git。
- **下一步**：在内验证比较 exact / related / morph 候选与患者关联学习，入选后扩大至四折并评原 dev16。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

运行恢复：10.5.2.3 `ontology-round2/fold0-sapbert.log` 与 manifest；managed `proc_2b5dc67e-9dcd-4cb1-ad05-a896cf74577a`，完成后交还GPU0。配置/命令见 [GPU 作业](evidence/round2/gpu-job.json)。
