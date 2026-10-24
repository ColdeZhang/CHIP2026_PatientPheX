# Span normalization 工作台账

- 目标：原dev公开v2默认all/text Score≥.8。这里只运行train64内部文献分组验证，尚未打开原dev、未访问官方。
- 当前：E004三新GP+八评分11步于10:38:31.715630完整结束，344.730秒，全rc0无超时；10:39:20后实际核GPU0=4MiB/noapps已释放。随后MCP再次offline，结果文件回收暂阻，尚未fit/quality；原训练不重启。恢复外部`oof-fourfold-completion/collect_terminal.py`同名ZIP。见[运行指针](evidence/oof-completion-runtime-pointer.json)、[实验](experiments/E-20261024-SPAN-004-fourfold.md)。
- 决策门：已知F1 +.002115173201、F0 −.001065881294之后补另两折，明确研发而非盲测。训练/候选池/旧特征/160树/.3/nested/missing全部保持。两新全预测共冻后各唯一v2，再pooled64唯一v2；复用原F0/F1预测和指标，不逐折选模型。仅pooled严格升且≥3/4折非降才考虑原dev迁移，否则关闭该特征控制。
- 计算：复用GP13/03/12/02，完整train32 bytes/IDs/config精确核；新GP01只训一次，另新GP01348/GP01248，共3训练/8全边界pass。新GP01两目标使用的训练文件bytes SHA也精确相同。所有原候选保留，包括低于−2分数；不能只存最终实体。
- 源/备份：最近核远端`explore/span-normalization@c56227230bf40671ab6eb4b79b89928b4d415761`。新completion文件独立，F1/F0已审源SHA不变。外部主目录`/workspace/patientphex-runs/round2-20261023/stage8-span-normalization/`；当前子目录`oof-fourfold-completion/`，服务器同名目录在项目根下。长作业须先查intent/started/terminal，不盲重启UNKNOWN。

## 完成控制与有效结论

| 控制 | 方法与结果 | 证据/结论 |
|---|---|---|
| E001 GP+规范化直接追加 | BiomedBERT GlobalPointer/RoPE64，多标签CE，8epoch；train32 F2+F3→F1。强.728697536469→union.701293121901，GP-only.663575801621，关系同底座。95.520s/576更新，2新v2、0旧 | [实验](experiments/E-20261024-SPAN-001.md)、[结果](evidence/result.json)。直接追加负，不采用；不能扩大为所有学习融合无效 |
| E002 strict OOF GP feature F1 | 原强池101→103，三新GP+四score；277.219s，单LGB fit/单新v2。Score.728697536469→.730812709670；mention TP+3/FP−18，doc分项也升，关系exact | [结果](evidence/oof-result.json)。独审bcb53dba PASS；模型/预测小包e8dc958f、review+E001负结果补包c334f5ed已实际.3核验 |
| E003 相同方法 F0 | 102→104，三新GP+四score272.023s，CPU16.639s；Score.740572140440→.739506259147，mention略升/doc降，26加46删，关系exact | [结果](evidence/oof-f0-result.json)。独审a3f855a0 PASS有效负；小包31191e6a已实际.3核验，未直接迁原dev |

模型固定108,990,080参数，lr3e−5/batch16/max384/stride64/seed20261024，BF16 encoder、FP32 head/loss。原生池floor−2与E001阈值0仅用于直接追加；后继GP特征按原强池精确边界评分，无floor筛选，缺失valid=false/logit0且原因显式保存。

E001全部2133跨度/2134拆ID单元保留。强池外447单元仅12精确TP+3NIL通配；最终追加361单元51精确TP+6NIL、304FP，独立供给弱。供给诊断首次defaultdict导致边界统计污染，已保原错误、以`supply-corrected`修复；预测/质量不受影响。完整来源及可复现入口见[方法](NOTES.md)。

## 接手注意与下一步

1. 查当前E004部署/运行意图，完成八组CPU token/mapping/覆盖后才唯一启动。源checkpoint先推送；900+30原时钟不隐式延长。实际release同时报root与sequence。
2. GPU完成全量目标后，`collect_terminal.py`按原ZIP断点恢复，逐SHA核；`run_cpu_quality.py`先两个fit-only冻结，再统一质量入口，不提前看F2分数后改F3。
3. 完整四折正负结果均保存/commit/push；独立终审集中处理。原dev迁移需先报实际库存和预算，不将内部折当用户0.8。
4. HPO上下文reranker查重由root的`normalization_methods`完成：旧SapTop8新ID+±180上下文+BiomedBERT CE确已实跑负，不重复开GPU。报告外置`research/context-normalization-gap-20261024/REPORT.md`，SHA c246fe36…；不同于仅旧池same-span ranker未fit。
