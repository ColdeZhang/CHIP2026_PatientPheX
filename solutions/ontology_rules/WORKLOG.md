# 本体规则工作台账

实验 `E-20261023-007`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：固定 dev16 最好 **.646244333**（4折 OOF 候选→二阶段树校准；原 .612597）；仍未达到0.8。语义+缩写四内折均正增益，dev .630237。初版校准与禁用auto固定160对照 .636026 都只用train64，后者为负结果；嵌套train48验证已完成：auto .703426/off .705023，与abbr .704790持平。
- **记录入口**：[实验过程](experiments/E-20261023-007.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/`，完整日志、预测、模型不得入 Git。
- **下一步**：CE1/3epoch、先验修正和固定融合均无有意义收益，停止此替换、不跑dev。嵌套CRF候选+树fold0 .710359略胜同池树.705023，仍低于完整CRF+Sap/rules .711354，准备另一外折复验。A800私有runtime已矩阵/全Qwen推理实测通过，卡由root交序列关系SFT。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

运行状态：两批GPU语义候选作业已结束，GPU0复查无进程并交还CRF；当前无本路线GPU计算进程；Qwen已完成且GPU1复查空闲后交还。关键产物 `/workspace/patientphex-runs/round2-20261023/ontology_rules/dev16-sapbert/augmented.jsonl`，来源与四折结果见[正式证据](evidence/round2/semantic-frozen-results.json)。

溯源：冻结SapBERT候选生成代码为 `016bb77/0084d33` 的 `gpu_candidates.py`（SHA dc5cc0…），后续 `0673f85` 为装配证据检查点；两者不混称。独立验证代理复评和逐字节重建4折/dev均通过，报告在外部 validation/ontology-sapbert-review.json。

最近已核验备份：`fc33f9f` 已推送本方案分支；CE3epoch与多来源候选融合结果作为下一检查点。

上下文CE3epoch已完成并释放GPU0给检索：作业 `proc_62ed89d4-d7a1-4d15-8079-71206cb163f6`；[固定对照证据](evidence/round2/context-ce-epoch3.json)。所有预设组合均未胜原树，不升级默认。

嵌套CRF3组及同源outer0重训完成，源快照SHA41ee61…，共享给序列关系SFT；自有输出 `nested-crf/inner{1,2,3}/`。多来源树fold0 .710359，[证据](evidence/round2/multi-candidate-fold0.json)，待跨折。

A800 GPU2只读环境和完整基座已交序列路线，私有解释器 `ontology-round2/qwen-venv/bin/python`；[CUDA/模型实测](evidence/round2/a800-runtime-verification.json)。本路线当前无GPU进程。

待运行：outer1嵌套3组CPU词典/CRF已完成，GPU SapBERT等待retrieval69链接释放；数据已上传且校验。HPO远监督双头检测CPU数据/代码完成，64,036训练行/87,134无标签候选，110,065切片核对，固定预算/阈值见实验第九阶段；不重复CE微调，尚无分数。
