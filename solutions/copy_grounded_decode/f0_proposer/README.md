# F0 同 GLiNER2.5 提议链

当前同权重F0推断已完整rc0（26.639秒，172窗/1207selected），正在回收原件；规范化尚未启动，评分0。只由同Task1 final10权重生成F0候选，非新质量臂；不得替用旧schema模型预测。初始[source/lineage](lineage.json)、[CPU薄合同](cpu-contract.json)、[实际推断计划](infer-plan.json)先由主线程commit/push。

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

## 独立必要证据传输门 v2

主线程看到原件包3,355,845B/103片回读耗时后，明确允许先完整回收selected/windows/inventory/终态：这是传输顺序改变，不改预测/接受规则。完整raw ZIP在服务器`.collection/proposer-output.zip`（cb201713）持久，旧collector/source/package未改；只停止本机自有只读collector190807，远端无kill。新 [essential-plan-v2](essential-plan-v2.json) 及 `collect_essential_v2.py` 先推送后执行，独立输出`$D/infer-essential-v2/collection.json`，schema明确essential全SHA而fullraw仍pending。

```bash
python "$S/collect_essential_v2.py" --directory "$D/infer-essential-v2"
python "$S/prepare_normalization_v2.py" normalize
# 主线程先push normalize-plan-essential-v2.json 后再pack/deploy/start。
python "$S/remote.py" pack --plan "$S/normalize-plan-essential-v2.json" --directory "$D/normalize-v1"
# 接上原normalize deploy/start/observe/collect四步，最后：
python "$S/finalize_essential_v2.py"
```

旧`prepare.py`及推断plan保持；新builder在所有172 windows唯一覆盖、selected总数逐窗一致、model/source/input身份核齐后才构必要query计划。个体raw IDs/heads诊断仍有原SHA/完整服务器locator；不得写allraw本地已齐。旧full collector可在不挤占关键MCP时续读原冻结包，禁重打包/重生成。
