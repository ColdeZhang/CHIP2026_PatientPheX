# 本体规则工作台账

实验 `E-20261023-007`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：固定 dev16 最好 **.646244333**（4折 OOF 候选→二阶段树校准；原 .612597）；仍未达到0.8。语义+缩写四内折均正增益，dev .630237。初版校准与禁用auto固定160对照 .636026 都只用train64，后者为负结果；嵌套train48验证已完成：auto .703426/off .705023，与abbr .704790持平。
- **记录入口**：[实验过程](experiments/E-20261023-007.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/ontology_rules/`，完整日志、预测、模型不得入 Git。
- **下一步**：Qwen全文关系为明确负结果（严格修复fold0 .581179；规则∩LLM .660433，均低于.704790），GPU1已释放给序列路线。短上下文BiomedBERT 1epoch及先验修正/固定融合已完成，最好融合.705075与同池树.705023持平，不升级默认；等序列路线释放GPU0做固定3epoch欠拟合对照。A800模型已复制校验完整，私有runtime仍在安装，尚不能宣称GPU可用。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

运行状态：两批GPU语义候选作业已结束，GPU0复查无进程并交还CRF；当前无本路线GPU计算进程；Qwen已完成且GPU1复查空闲后交还。关键产物 `/workspace/patientphex-runs/round2-20261023/ontology_rules/dev16-sapbert/augmented.jsonl`，来源与四折结果见[正式证据](evidence/round2/semantic-frozen-results.json)。

溯源：冻结SapBERT候选生成代码为 `016bb77/0084d33` 的 `gpu_candidates.py`（SHA dc5cc0…），后续 `0673f85` 为装配证据检查点；两者不混称。独立验证代理复评和逐字节重建4折/dev均通过，报告在外部 validation/ontology-sapbert-review.json。

最近已核验备份：`b89a8e1` 已推送本方案分支；CE1epoch全部结果与先验修正作为下一检查点。

上下文CE1epoch已完成并释放GPU：作业 `proc_d0d9c543-457a-40ae-871d-d30b6aab3b2d`；[固定对照证据](evidence/round2/context-ce-epoch1.json)。原始替换最好.637166，训练先验修正最好.660497，字典保留最好.704139；未胜同池HistGB.705023。固定3epoch待GPU交接，不跑dev。

嵌套CRF补充候选：外部 `nested-crf/run_nested.py` 顺序单线程启动3组train32→inner16，系统exec session8594；源快照SHA41ee61…，自有输出 `nested-crf/inner{1,2,3}/`，完成后共享给序列路线关系SFT作同折OOF输入，不重复训练。
