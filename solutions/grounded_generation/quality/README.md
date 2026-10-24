# 固定 F1 双端点质量入口

目标是检验 whole-report 生成并正规化的患者关系能否补充强 N 的有限概念候选池。这里只准备 CPU 组装、共同冻结与评分源码；尚未组装真实预测、读取或哈希 held gold、调用 evaluator、运行模型。已通过源码 AST 检查和小型合成边界合同，详见 `synthetic-contract.json`。父线程负责冻结、实际绑定和唯一执行；无 Git 操作。

固定对照为全文 normal 单臂 `f33a30f1…`（154584 B），原 v2/all/text 指标 `875292c3…`，Score **0.676546354662594**。两端点均覆盖原 F1 16 文献、39 患者、1189 实体：

- `primary`：N ∪ new。实际只读确认N的412条已是唯一canonical HPO原子值，入口锁定这一前提，原关系列表及顺序完整保留prefix，再添加按 `(type, value)` 排序的新键。
- `new-only`：仅保留相同 new 键，按 `(type, value)` 排序。

两端点均可被保留：选择 Score 最高者，精确平分优先 primary，且必须严格高于强 N。另报 primary 并集假设是否通过；macro 完整披露，无额外否决。这个对照胜出只表示方法候选，不表示超过所有历史完整融合或官方成绩。不增加第三端点、阈值/排列/epoch 扫描或自动迁移其他 fold。

## 输入与组装

`plan-template.json` 含现有来源的实际 SHA。gold 只复制指定元数据，模板本身不可运行。完整原推断、提取和正规化结束后，父线程绑定 `actual` 的八件精确本地 ref：`collection`、`extraction_plan`、`extraction_manifest`、`quotes`、`decisions`、`normalization_manifest`、`normalizations`、`semantic_evidence`。设 `state=BOUND_COMPLETE39_AND_NORMALIZATIONS`，按既有授权设 `quality_execution_authorized=true`，在组装前一次冻结源/计划；组装与评分之间不改计划。

`assemble.py --plan BOUND_PLAN` 会核对全部 collection 原成员、39 个 raw/IDs、39 条 extraction decisions、完整患者覆盖及每个 anchor 与正规化记录的顺序和全部原字段。使用父线程 `held_finalize.py` 的 `grounded-held-normalization-complete-v1` manifest，连同 preparation、query collection/complete/scores 原身份检查；不重新编码、查 bank、拟合或执行正规化。正规化的 `mapped` 必须是当前 canonical 合法 HPO，复合值拆成原子项；`nil` 必须保留既定 strict literal unanimous train32 NIL 的原 quote；`abstain` 不产生关系。

完整 39 条不等于 39 条都 EOS/JSON 有效。既定 decoder 的整行 abstain 或部分无效 anchor 被拒结果完整计入，不能丢患者或重生答案。这里不重新 native decode/parser，只核原 raw/IDs 与已冻提取决策；当前定位词面是概念标准化锚点，不作为该出现位置归属于目标患者的证据。

基线没有 `patient` 字段：患者集合来自 annotation-free input 和 `association[].patient_id`。只 AST 提取已审 `common.py`（`c713d2f5…`）四个字节函数 `field_span/array_spans/phenotype_spans/mask_values`，不 import 旧计划或多数逻辑。仅替换 `association[].phenotype` 值片段，实体、PID 对象、所有其他 metadata、空白和换行逐字节保持。复合/alt-ID 去重沿本路线 `HPO.relation_keys`，NIL 的分号文本不拆分。

候选突破统计绑定原全文 80 行 `expanded80.jsonl`（`2a95bad1…`）及原 plan、`actual-input-and-assets.json`。复用 `code_map` 和 `patient_ids`；原口径 sum codes=587、sum codes×patients=1557，另给去重文献/患者池。先报告 new∖N、new 超出原患者候选池/文献候选池的键与数目；只有评分后才给这些集合的 TP/FP。出现新概念不自动等于正确贡献。

## 冻结与唯一评分

组装先保存两份完整 JSONL、两份 input-only validator 报告、贡献及覆盖合同，最后发布 `assembled/common-freeze.json`。任何失败保留原部分产物和 trace，不自动恢复。显式独立运行 `score.py --plan BOUND_PLAN` 才进入评分；监督进程和组装受 gold deny-open，评分子进程验证共同冻结之后才允许 gold 校验/读取。

固定 evaluator `ea061a98…`、validator `b3fb9f33…`；先以 canonical 名注册 validator，避免兄弟模块导入错误。预算 **30 秒 + TERM 1 秒**，按 supervisor 原单调时钟剩余时间计算，不续 deadline。每端点调用前独占 `call-start.json`，返回 metric 后立刻 fsync 保存再计算诊断；旧 N 指标复用、旧 evaluator 调用 0。已有完整指标不重评，有 attempt 无 metric 即阻断，失败无自动重试。两个唯一 `evaluate_report` 保留原 v2 默认 sensitivity 字段，不把它们作为新选臂。

gold 的重复 phenotype 是合法输入，诊断 truth 使用冻结 evaluator 的 `associations` 集合语义，不走预测重复键断言。结果保存五项指标、相对强 N 差值、增删 TP/FP、候选突破 TP/FP、获选端点与 primary 假设成败；Task1 的 menF1/docF1 必须保持一致。源码接口和实际未绑定项见 `source-contract.json`。
