# F0 全文 FLA 双终点评测准备

机械复用已成功的 F1 FLA 质量流程，固定 primary = FLA-alone、diagnostic = native N_full ∪ FLA。F0 native N 对照预测 `f87b2c9b…`、指标 `1e6ef503…`、Score `0.6966039747439822`。任一新终点 Score 严格增加才保留；都增加时取较高者，完全相同选 primary。macro 及全部分项披露，不设附加 macro 门。

原强三方对照 `c46f4662…` / `b92dbbb1…` 的 Score 为 `0.7095567430848545`。本次已只读核实其与 native N 的非 association 字节完全相同（118272 B，SHA `40e43b1b…`），16 篇顺序、47 患者 ID 顺序和 947 实体一致，见 `strong-comparator-check.json`。未来 summary 同时披露对此强对照的差值；仅超过 N 不能称 F0 全局最佳，低于或等于强三方不能称新最佳。强三方不重评、不作为第三终点，不改变既定 N 判据。

新 F0 输入为原 58 行 `d6332252…`、audit `2fa5c495…`、CPU 结果 `ccdcd461…`，415 CID / 1166 患者概念对、总 505090 tokens、最大 16245，模型输入 cap 16384。固定旧 SFT48 `8e32daee…` / config `165dbc59…`、greedy 1600；58 行全部 fresh、cache0，不复用旧 native 最长第30行。新第30行先运行并通过完整工程 gate，其余按长度/row_id 排序，质量入口核 gate 与 IDS/raw/format 的真实成员身份。

只调用 F0 `evaluate_full58.py`（`662aeccc…`）中纯 `canonical_gate/make_primary/assemble_two/delta` 和 `literal_helpers.py`（`1c70b9c4…`）；canonical_gate 上限本来就是 16384。**不调用其旧 main、reference_gate、verify_inference、retain macro 门、响应解析或缓存加载。** `assemble_two` 返回 `(alone, union, counts)`，两份预测按同一顺序绑定 primary/diagnostic，保持所有非 association 字节。

两份完整预测共同写入、fsync、冻结后才导入 evaluator 并首次读取 fold-0 gold `630c9e39…`；各一次 v2 all/text，旧 N 和强三方零重评。质量 wrapper 沿 CPU60 / wall65+TERM5，exclusive marker 防重复，保留失败、不自动重试。GPU 的900+TERM30、最长 gate及模型运行由父代理负责。

当前 `template.json` disabled，动态源和实际终态尚未绑定。完整 GPU 包回收后，父代理运行：

```bash
python bind_actual.py --template template.json \
  --inference-dir /absolute/complete-collected \
  --published-plan /absolute/published-F0-gpu-plan.json \
  --source evaluate.py --output /absolute/bound-plan.json
```

绑定器只核已发布计划与回收计划的字节/SHA、全部来源与完成成员，不读 gold。源码和 bound plan 保存后，再由父代理执行唯一质量入口：

```bash
python run_quality_once.py --plan /absolute/bound-plan.json \
  --source evaluate.py --output-dir /absolute/unique-F0-quality-execution
```

本次只做源码编译/静态合同及已有两份对照的字节核查，没有执行 binder、响应解析、tokenizer、模型、gold 读取或评分。这是已使用的 F0 开发控制；原 native 在 A800、新 FLA 在 A30，硬件/后端差异披露，不自动迁移原 dev、其他折或官方。
