# 本体规则工作台账

实验 `E-20261023-007`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：固定 dev16 最好 **.646244333**（4折 OOF 候选→二阶段树校准；原 .612597）；仍未达到0.8。语义+缩写四内折均正增益，dev .630237。初版校准与禁用auto固定160对照 .636026 都只用train64，后者为负结果；嵌套train48验证正在准备。
- **记录入口**：[实验过程](experiments/E-20261023-007.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/`，完整日志、预测、模型不得入 Git。
- **下一步**：先验证外层fold0 train48内三子折候选校准，避免把第二层训练拟合当OOF。Qwen关系22窗口与2窗口smoke已冻结；.100GPU0被他人占用，root正在分配.4.7A800 GPU2及迁移共享权重，runtime尚未就绪，未启动LLM。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

运行状态：两批GPU语义候选作业已结束，GPU0复查无进程并交还CRF；当前无本路线GPU运行。关键产物 `/workspace/patientphex-runs/round2-20261023/ontology_rules/dev16-sapbert/augmented.jsonl`，来源与四折结果见[正式证据](evidence/round2/semantic-frozen-results.json)。

溯源：冻结SapBERT候选生成代码为 `016bb77/0084d33` 的 `gpu_candidates.py`（SHA dc5cc0…），后续 `0673f85` 为装配证据检查点；两者不混称。独立验证代理复评和逐字节重建4折/dev均通过，报告在外部 validation/ontology-sapbert-review.json。

最近已核验备份：`bd1ba48` 已推送本方案分支；此后的固定160与嵌套验证准备作为下一检查点。
