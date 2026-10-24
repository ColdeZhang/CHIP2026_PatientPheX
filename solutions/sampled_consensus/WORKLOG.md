# 非思考双种子共识台账

## 当前状态

固定主配方 `B ∪ (S1 ∩ S2)` 已完成内部F1、F0及原dev唯一迁移。原dev由 **.754361398199→.755239908536**（+.000878510337），micro/macro均微升；新增8=3TP5FP，全部1061实体/416旧关系/54池外关系及其他对象保持。按预写严格升门保留本线候选，未达到用户.8目标，不称盲测或新增高精度。未进行官方上传。

| 实验 | 完整控制与结果 | 终审/证据 |
|---|---|---|
| [SAMPLE-001](experiments/E-20261024-SAMPLE-001.md) | SFT32同native A800，80×2；F1 .686338981→.693653304，+34=16TP18FP；替换型多数诊断负完整保留 | PASS e5437e12；`evidence/result.json` |
| [SAMPLE-002](experiments/E-20261024-SAMPLE-002.md) | 同4f48 native A30，58×2；F0 .711507580→.721969518，+21=9TP12FP | PASS 1c110220；`evidence/F0-result.json` |
| [SAMPLE-003](experiments/E-20261024-SAMPLE-003.md) | full64模型1a703b、真正全文64×2；原dev .754361398→.755239909，+8=3TP5FP；仅1v2 | 完整结果/备份交集中增量审；`evidence/full64-result.json` |

原dev128已852.191541秒rc0，12:10:33.243420Z实际释放.100 GPU1为0MiB/noapps；64×2全EOS/strict，0截断/弃答。411内容原包284031B/01e7681e已逐SHA并整包读回，全部原IDs/raw/canonical/input/source合同通过。唯一预测ee39defb在commit68228730中于质量前冻结；之后仅1新v2，0旧/单seed，未训练。完整两seed原始关系299/343、交集289全部保留供后续预定融合，不能只取原dev正确新增。

## 接手入口

分支 `explore/sampled-consensus`；主入口 `full64/{cpu_gate,driver,infer,verify_terminal,assemble,quality}.py`，固定计划 `full64/plan.json`。外部R=`/workspace/patientphex-runs/round2-20261023`：

- F1：`research/nonthinking-sampled-consensus-native-20261024/terminal-manifest.json`，原包及质量包均全读回。
- F0：`research/nonthinking-sampled-consensus-F0-20261024/terminal-manifest.json`（ee822542）；375原内容、74质量内容均全读回。
- 原dev：`research/nonthinking-sampled-consensus-full64-20261024/terminal-manifest.json`（6236e9d8），完整actual命令/来源/quality/终态均绑定；两seed逐行诊断在`assembled-v1/row-diagnostics.jsonl`（b3830063）。
- 原dev全文准备：`research/nonthinking-consensus-full64-readiness-20261024/readiness.json`（ce842e9e）。旧normal64短提示不可冒称全文，新完整64为512091 tokens/pass，max13230；CPU独审1c110220只覆盖先前准备，不覆盖本次128质量。

三次GPU均已结束，禁止按旧start脚本重启。最后原devjob `proc_de171481-475e-41b2-806d-5ff33364cd8d` / PID2735800；原TERM12:31:20.925286Z/KILL12:31:50.925286Z保留，未延长。新GPU、额外原dev变体或官方动作需root新任务；本次已授权链已完整结案，仅待root集中增量审。

## 关键边界

F1候选训练32、F0候选训练48，所有目标折祖先排除；前两控制生成适配器均F2+F3训练32。原dev改为现成train64适配器1a703b，全部任务拟合祖先排原dev，无新训练。F1 A800/F0及原devA30保持native源码/版本/采样规则，不声称跨硬件逐位等价。内部收益未等比例迁原dev，强B已覆盖交集281/289；原dev微升是真实但有限的研发结果。

原dev质量备份已完成：96内容325226B/SHAb5b6dac5，远端fsync/逐SHA与整个实际readback完全一致。新best预测/metrics/recipe/source均在该包内，单独恢复入口`evidence/full64-recovery.json`及外部`best-recovery.json`；原128另包完整保留。原901/官方50ef未覆盖，未达.8不合并实现，branch最后持久化后释放任务槽。
