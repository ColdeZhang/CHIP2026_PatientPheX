# 四源全文组合：固定两折、两方案 CPU 对照

这是看过已有组成方案结果之后登记的**事后开发集对照**。目前仅准备源码、计划和合成检查，没有读取实际预测/input/gold 正文，没有生成新实际预测或执行评分。父线程负责完成源码提交推送后，按唯一入口执行。

比较基线固定为已经完成的 `old ∪ reverse ∪ patient` 三方预测：F1 Score **0.6638418206599839**、F0 **0.7095567430848545**。来源来自 `triple-union-fulltext-handoff-0600/refs.json`（`c1ecab45…`），指标为 `14b708b2…` / `b92dbbb1…`；历史三方计划里的 `old_union_metrics` 是较弱的旧比较对象，本次不能使用。

| 方案 | 固定关联集合 |
|---|---|
| `primary-four-source` | old ∪ reverse ∪ patient ∪ fulltext |
| `diagnostic-without-old` | reverse ∪ patient ∪ fulltext，去掉 old 的替换诊断 |

fulltext 只使用已经冻结并终审的全文单独端点。F1：`f33a30f1…`；F0：`f87b2c9b…`。不读取或重新解析模型 raw，不修补答案，不训练，不扫描阈值。F1/F0 仍为各16篇、39/47患者、587/415候选、1,557/1,166候选对、1,189/947实体；不混入原dev的其他实体模板。

运行时逐源核 SHA、完整文献/患者、同一候选池、非关联字段语义。复用成功三方 `734aef13…` 的 `build_union` 原函数，先重建 old∪reverse∪patient，要求每患者的**原始概念字符串集合**等于既有三方；数组顺序可不同，不能把 HPO compound 拆分后的集合拿来蒙混通过。全文与其他来源的全部正关系必须属于相同已冻候选池。

四份输出均采用既有三方 JSONL 的原 UTF-8 模板，仅替换顶层 `association` 值。顶层其他字段及其空白、Unicode、数值写法、换行字节保持；患者及关联内其他 metadata 保持相同语义。只复用 F1 通用 `association_span` / `nonrelation_bytes` 原函数，不带入其旧 retain 条件。

一次执行严格按以下顺序完成：

1. 排他 `started.json` 在任何实际预测/input 读取之前创建；已有标记不能重新启动。
2. 两折四份完整预测全部生成；源/计划/四份预测 SHA 共同写入 `assembled/common-freeze.json`。目前四份预测尚不存在，不伪造预先 SHA。
3. H50：先注册精确 canonical `validate_predictions`，再载入 `evaluate`。四份格式检查全部通过。
4. 共同冻结核验通过后才允许任何 gold 哈希/打开。F1主、F1诊断、F0主、F0诊断各唯一一次公共 v2 all/text，共4次；旧指标直接复用，旧评分0次。
5. 写入指标、相对强三方的分项变化、双折门槛及候选选择。失败/中断保留已有标记、输出和调用意图，不自动重跑。

每个方案必须在**两折分别 Score 严格上升且 macro 不下降**才通过。如果两方案都通过，选两折 Score 增量合计较大的方案；精确平局保留 primary。诊断方案的身份不会因结果较好而改写。通过也不自动迁移原dev或官方；未来若另行迁移，必须保留35条 public-new。

唯一执行入口（父线程使用外部 runs 目录）：

```text
python control.py --directory /workspace/patientphex-runs/round2-20261023/research/four-source-fulltext-control
```

进程内部 CPU 上限60秒；普通 Python 执行，不能使用 `-O`。`entry.json` 绑定 source/plan/helper/合成合同，`plan.json` 的 gold 路径与 SHA 仅抄自已冻 metadata；准备过程未触 gold 文件。没有训练、远端或实际模型入口。

`synthetic-result.json` 记录22项纯合成检查，实际约0.019秒。覆盖 compound/NIL 原串、UTF-8/CRLF/数字字节、缺/重复患者、候选池越界、非关联漂移、三方重建错误、缺第四预测/第四SHA漂移必须先阻断gold、排他标记、严格Score门、macro门、两折选择和平局以及H50/冻结/评分顺序。未调用公共评测器。完整来源和计划见 `manifest.json`，后续实际结果由 owner 另行记录。
