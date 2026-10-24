# F0 同 GLiNER2.5 提议链

当前仅CPU准备完成，未部署/启动/评分。只由同Task1 final10权重生成F0候选，非新质量臂；不得替用旧schema模型预测。初始[source/lineage](lineage.json)、[CPU薄合同](cpu-contract.json)、[实际推断计划](infer-plan.json)先由主线程commit/push。

固定16doc/47patient/172完整窗，原FP32/native phenotype query/threshold.5/overlap allow/cap192，600+30绝对截止。原模型193581591参数、权重54a88544；仅F0身份/计数绑定与耐久raw发布有机械变化。增加每窗原head tensors保存后再检查有限/坐标；API结果沿原入口先保存后映射。raw/IDs/候选全量保留；不重训、不扫描、无gold/quality。

正常后继是原strict train32 F2F3 + public41099 FP16 Sap/b59政策，全部mapped/NIL接受，拒绝项保留；不能用全部raw跨度代替accepted池。原Sap运行库Torch2.7.0/TF4.50.0(.3)→本次Torch2.7.1/TF4.51.3(.100)差异披露。GPU0 fresh门，GPU1不碰。原GLiNER runtime复用TF5.20.2。

设 `S=solutions/copy_grounded_decode/f0_proposer`，`D=/workspace/patientphex-runs/round2-20261023/research/copy-grounded-decode-v1/F0-proposer-v1`。以下是顺序入口；只有主线程确认source/plan已推送后才执行部署和start。

```bash
python "$S/remote.py" deploy --directory "$D/infer-v1"
python "$S/remote.py" start --directory "$D/infer-v1"
python "$S/remote.py" observe --directory "$D/infer-v1"
python "$S/collect.py" --directory "$D/infer-v1"
python "$S/prepare.py" normalize
# 此时先由主线程push新normalize-plan.json，再按实际必要query数执行后继。
python "$S/remote.py" pack --plan "$S/normalize-plan.json" --directory "$D/normalize-v1"
python "$S/remote.py" deploy --directory "$D/normalize-v1"
python "$S/remote.py" start --directory "$D/normalize-v1"
python "$S/remote.py" observe --directory "$D/normalize-v1"
python "$S/collect.py" --directory "$D/normalize-v1"
python "$S/finalize.py"
```

正常输出为 `$D/accepted/entities.json` 和 `manifest.json`，可与strongB fold0预测79d3568e逐offset合成有限三元组。`prepare.py normalize`在完整推断回收后只CPU计算必要query数量；若为0只产生`normalize-cpu-zero-query-plan.json`，不得启动GPU，应执行该计划CPU argv。当前不预言实际query数。

transport复用既有MCP分块/冻结package/唯一startclaim/UNKNOWN恢复；任何UNKNOWN先observe原意图，不重放start。收集只在原child结束后冻结并回读原件ZIP；rc非0仍保留失败产物，不能冒充成功。源完成后不得修改已冻结job的source/plan，后续机械恢复另记命名空间。无原始F0标签读取。
