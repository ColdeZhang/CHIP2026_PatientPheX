# 全文 Med 固定双结果评估入口

本目录仅适配与冻结源码，不执行模型、parser、validator、v2 或 MCP，也不读 gold。primary 为 fresh strict32 全文 Med；唯一 secondary 为旧 SFT32 与全文 Med 的关系集合并集。强对照始终为已缓存旧 SFT32（历史 Score 约 .660942）；短输入 Med 的既有结果不重评、不加入选择或其他组合。

`evaluate_medgemma.py` 沿用原短 Med 入口 `22aec1f4…`。逐字节非关联字段保留、literal union、格式检查、严格关系 micro+macro 增且 macro 不降的建议判据，以及两份完整预测共同冻结后各唯一一次既有 v2 的控制流不变。相对 primary 的 union 差值只报告，不另设选择条件；不触发自动迁移。

增量仅绑定全文来源和新状态：trainer `b486470e…`、infer `36d8185f…`、合同 `5188b958…`；训练 124 条/965436 tokens/7150监督/32更新，cap14000，新 fullcontext checkpoint schema。原标签链 `356109…` / `cb7…` 与可逆追加 `811e…` / `9e80…` 分列。源、adapter、完整 epoch002、实际推断 manifest、19件模型资产记录和最终持久 ACK 相互绑定。模型来源始终是资产 inventory `expected-files.json`（`690e0829…`），不使用代码清单冒充。

固定原 `ddba…` 的 80 条输入用于原 parser/CID→PID 框架与 baseline fallback；实际模型输入绑定完整 `2a95…`。评估前调用已审追加合同，验证原消息及所有 row/CID/PID/候选字段可逆保持、纯16文献原文来源不变。实际响应的每行输入 token 数还需匹配既有 native CPU 行审计，不仅满足14000上限。

持久路径固定 `.100` 的 `research/medgemma-relation-control/fullcontext-formal2ep/checkpoints/epoch-002`。ACK 必须包含完整 checkpoint state、9份训练源码，以及 `_cpu/reference/summary.json`、`_cpu/reference/row-token-audit.json`、`_cpu/preparation.json` 的精确 SHA。`evaluation-template.json` 保留原动态产物布局，指向本次 `fullcontext-preparation/formal-preparation/formal`。

owner 回收完整推断后才可运行：

```sh
python freeze_plan.py --formal-directory <本次formal目录> --output-directory <新paired-evaluation目录>
python evaluate_medgemma.py --directory <新paired-evaluation目录>
```

freeze_plan 仅把已完成产物绑定为三元引用，不读取 gold、不评分。两份预测完整落盘及共同 freeze 前，gold 仅保留历史路径和 SHA，不加入预先核验的 references。入口使用不可覆盖的 reservation 阻止重复评分；异常保留原状态，不自动再评。

`cpu-contract.json` 的16项已实际通过：小构造的重叠并集、字节保留、重复/字段错误拒绝、一次性 gate、源码中 freeze/gold/v2 顺序，以及真实全文80对原80的可逆输入框架。没有模拟完整神经终点，也没有新质量结果。`pretraining-amendment.json` 固定上述选择；源码差异见 `endpoint.diff`。
