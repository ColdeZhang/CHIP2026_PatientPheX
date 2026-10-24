# GLiNER2.5 Task1 工作台账

2026-10-24 UTC，分支 explore/gliner25-task1；仅维护 solutions/gliner25_task1。

- **当前：完整训练与 F1 抽取均已成功，固定规范化部署中。** Fresh 32 篇/320 窗/10 epochs/1600 更新共 342.260 秒；完整 F1 16 篇/39 患者/164 窗推断 23.942 秒，1466 个窗口实体全部坐标合法。没有新的质量分数。
- **资源：** .100 GPU0 已于 11:58:02.427201 UTC 实际 0MiB/noapps 释放；本线不再占用。后续仅 .3 必要 Sap 一次 ≤180+30 秒，fresh 门后启动，实际释放通知 Qwen owner。
- **固定质量端点：** 强 Task1 8cb61ba9（Score .728697536469）∪全部规范化接受 mapped/NIL 实体，精确原子去重；另 new-only 完整诊断保原关系。两份完整预测共同冻结后各 1 次公共 v2，旧 0；主端点严格胜才候选。尚未访问本线 F1 gold，原 dev 最佳不变。
- **完整证据：** [训练/推断终态入口](evidence/formal-inference-complete.json)。正式 642 小件全部 SHA 回收；epoch5/final 模型、optimizer/scheduler/scaler/RNG 大产物有精确远端 locator/size/SHA，read-back 一致，但没有声称原生自动完整 resume。推断 508 件全部回收，保留每窗实际输入、全候选分数和原 API 输出。
- **正式固定合同：** 原生 .2 随机 query augmentation、全部 320 窗每轮恰好一次、19200 raw loss 分项有限、606 实际随机 schema 保存；无 holdout/eval/earlystop/选 epoch。峰 allocated 4.865GB、reserved 6.134GB；只取 final epoch10。
- **先前工程故障：** 静态旧 IDs 误判随机 query 与 no-eval best_metric=inf 的 JSON 哨兵错误均已保留，6 个真实更新无需重跑。见 [实验](experiments/E-20261024-GLINER25-TASK1-001.md)、[探针恢复](evidence/probe-final.json)。共享 metadata 仅处理未计算哨兵，不清洗真实 loss。
- **恢复目录：** R=/workspace/patientphex-runs/round2-20261023，本线 R/research/gliner25-task1-v1；所有启动意图、原 proc 身份和完整回收索引都在此。正式 proc_e9d09124、推断 proc_84227fbf 均实际 rc0，不重复启动。
- **最近已核远端：** 0794559e，含固定推断、规范化和两端点质量源码。当前阶段文档检查点随后推送。

下一步：固定 train32/public41099 规范化完成后，2 份完整预测共冻→2 次新 v2，记录强基线增量与失败/收益；随后提交推送并做小产物备份。不访问原 dev/官网。
