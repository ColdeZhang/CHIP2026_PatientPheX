# 长上下文 GP 原跨度删除控制

独立预锁见 [decision.json](decision.json)：对原强基线每个实体，只在同精确 `(pmc_id,offset,length,text)` 有合法 GP 表示且完整17窗所有表示的 raw score 最大值严格小于0时删除。不可表示、无共同窗口、恰好0均保留；NaN/Inf、缺行、重复表示或同跨度冲突使全终点失败。同跨度的全部原实体对象（含复合ID）一致处理，其他对象与剩余顺序保持。与父方法的 protected union 终点独立。

生产接口已与 document owner 对齐，见 [interface.json](evidence/interface.json)。本代码沿已验 `boundary_veto` 的排他写入、装配、对象保护、byteexact零变更和单次v2入口，只替换判定/来源。只用标准库，源/结果独立；旧NO、9B和BOUNDARY删除控制已存在，本次变量是新长context GP状态，0为原二分类界。

**已按下列命令执行并负结案；不要重跑。** 结果见 [WORKLOG](WORKLOG.md) 和 [结果指针](evidence/result.json)。以下保留原复现入口；原目录中的排他intent不可覆盖。父完整终态/256更新ACK与全部17窗原件已由root核源：

```bash
S=/workspace/patientphex-worktrees/explore-long-context-span-veto/solutions/long_context_span_veto
D=/workspace/patientphex-runs/round2-20261023/research/biomedical-long-context-readiness-20261024/formal-v1/collected/output
O=/workspace/patientphex-runs/round2-20261023/research/long-context-span-veto-v1
mkdir -p "$O"
python "$S/run.py" plan --endpoint "$D" --output "$O/F1-v1" --plan-output "$O/plan-F1-v1.json"
python "$S/run.py" assemble --plan "$O/plan-F1-v1.json"
```

`plan`绑定完整终态和sidecar的SHA，不按分数挑配方；`assemble`核完整1313记录、窗口身份、表示/分数/max一致性，持久freeze唯一全量预测。这两步禁止打开真实gold。只有root完成来源静核后执行：

```bash
python "$S/run.py" quality --plan "$O/plan-F1-v1.json"
```

严格一次新v2 `all/text/wildcard=True`、旧0；零删除继承旧字节及metrics，gold0/v2=0。排他intent防重评。负结果不扫阈值或子集，不自动扩F0、原dev或官方。合成增量合同：`python "$S/contracts.py" --output /tmp/lctx-veto-new-contracts.json`。
