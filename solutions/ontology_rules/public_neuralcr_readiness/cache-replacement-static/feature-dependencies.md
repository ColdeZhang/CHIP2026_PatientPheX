# BioBERT → NCR：候选、六来源字段与 HGB 静态依赖

结论：六个 `biobert_*` 特征确实是可独立识别的六列，但“删除六列”不等于移除 BioBERT。它还增加了独有候选键，并参与旧 HGB 的训练和所有旧 `selector_probability`。最干净入口是冻结的**前两公开模型原始池**，在其上重新合并 NCR 的真实输出并 fresh-fit HGB。此次仅读源码、既有 manifest 和压缩池 SHA；未解析 gold、未按 FN 挑样本、未重建池、未运行模型/训练/评分/MCP，也未改 Git。

下文行数/维度均来自已完成的冻结 manifest；本次核了实际文件 SHA，没有重新计算质量。完整路径、SHA、函数行号与 16 个压缩池的实际 SHA 核对结果在 [feature-dependencies.json](feature-dependencies.json)。下列方案源路径以 `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/` 为根，产物路径以 `/workspace/patientphex-runs/round2-20261023/` 为根。

## 一、应当复用的冻结基底

- **八个前两模型池**：`ontology_rules/public-phenotagger-top3-control/old-pools/manifest.json`，SHA `87264eb85a1801e249ac6f052faeec5ddeb03fa3044ecb014ca3b9c8d6d220bf`。其 `fold0..3/{train,input}.pool.jsonl.gz` 合计 **244794 行**。父目录名称虽有 `top3-control`，这里的 `old-pools` 完全是原 semantic/base/CRF + 原 PubMed + Bioformer，未加入 top3 或后来失败的 raw/prehybrid/Sap 特征。
- 精确重建源码：`public_phenotagger_top3_control/prepare_old_pools.py::main()` 内部 `build()`，调用 `calibrate_multi.pool()` → `select_public_candidates.merge_pool()` → `select_public_bioformer.merge_bioformer()`。它对 held 池核验与冻结旧分数文件去除 `selector_probability` 后完全相等；`origin_index` 及三层来源保留。无需重新调用旧 HGB 才能取得这些候选。
- 第一公开模型准备缓存：`sequence_crf/public-selector-control/prepared/{manifest.json,candidates.jsonl}`，6064 个键，权重 SHA `d58af55a…99d7ec3`，字段前缀 `public_`；其 native manifest 位于 `research/public-phenotagger-train64-cache/terminal/native-train64-cache/manifest.json`。
- 第二公开模型准备缓存：`sequence_crf/public-bioformer-selector-control/prepared/{manifest.json,candidates.jsonl}`，5869 个键，权重 SHA `867819ac…0024a1`，字段前缀 `bioformer_`；其 native manifest 位于 `research/public-phenotagger-bioformer-train64-cache/terminal/native-train64-final-cache/manifest.json`。
- 两公开缓存均声明 `public_model_task_fit=false`，且四字段 train64 输入 SHA 均为 `ee252384…c246`。前两模型候选及其 12 个来源字段可保留；不能把它们的神经分数用于 NCR。

## 二、BioBERT 的六列是什么

定义与合并：`public_phenotagger_biobert_selector/prepare_pool.py::{FEATURES,OLD_FEATURES,defaults,merge,main}`；HGB 入口：同目录 `fit_control.py::feature()`。

| 列 | 精确含义 |
|---|---|
| `biobert_present` | 当前规范 HPO 的精确 `(pmc_id,offset,length,identifier)` 键出现在 BioBERT 最终 native canonical/provenance 中；共享旧键和新键都可为 true。 |
| `biobert_score` | 同规范键全部实际 native hybrid 行的最大 score；它**不保证是原神经 posterior**，也可能来自词典或缩写流程。缺源时为浮点 0。 |
| `biobert_neural` | provenance 的 `raw_neural_match` 实际 OR：原 raw HPO/span 经原生 top1/5 位小数且严格 `>.95` 接受。 |
| `biobert_dictionary` | 原 raw HPO/span 实际存在于 dictionary raw 结果的 OR。 |
| `biobert_abbreviation` | 同原 raw HPO/span 在 abbreviation 后存在、之前不存在的 OR。 |
| `biobert_origin_unknown` | 此次所有已归因键均 false，缺源默认也 false；不由 score 猜测来源。 |

这些标志不是互斥类别，也不是从概率反推。真实证据构造在 `public_phenotagger_biobert_train64_cache/make_provenance.py`：用原 raw HPO/span 追 neural/dictionary/before/after，再对规范键 OR flags、保留 raw IDs/神经 head columns/实际 hybrid score 集合。`prepare_pool.py` 验证这些集合后才生成六字段。

HGB 的实际特征为 `calibrate_multi.features(input,candidate,base)` + 原 `public_*` 六列 + 原 `bioformer_*` 六列 + 新 `biobert_*` 六列。raw HPO IDs/head column 元数据不是额外可学习列。`fit_helpers.py` 虽来自 prehybrid 控制、含 `RAW_FEATURES` 常量，但当前调用只用其 provenance/labels/load helpers；当前 `feature()` 没有追加那些 raw 特征。

四折列数实为 **96→102、95→101、97→103、97→103**；full64 为 **97→103**。每一项实际新增列集合都恰为上述六列，没有藏在一列中的多个模型。不同折旧列数变化来自 `DictVectorizer` 的实际 `section/ptype` 类别词表，不能把 103 当任意新实验的固定 schema。

## 三、哪些候选仅来自 BioBERT

严格条件是 **BioBERT 增强池相对冻结前两模型池的精确键差集**，不是 `biobert_present=True`。`merge()` 复制所有旧候选并附六个默认字段；共享键只更新六字段，旧字段/值/类型/顺序不动；新的键追加在旧候选前缀后。

新增键的非来源默认值为 `probability=.5, similarity=0., exact=False, is_base=False, crf=False, semantic_source=False`；前两模型 12 字段均 false。`crf_span` 仍可 true，因为它仅表示相同跨度曾由 CRF 提供，不能据此认为同一个 HPO 键早已存在。原 semantic probability/Sap similarity 的缺失默认值也不是真实新计算的分数。

| 外折 | 旧训练行→BioBERT行 | BioBERT 独有训练行 | 旧 held 行→BioBERT行 | BioBERT 独有 held 行 |
|---|---:|---:|---:|---:|
| 0 | 46349→46733 | 384 | 14910→15028 | 118 |
| 1 | 45039→45403 | 364 | 16454→16595 | 141 |
| 2 | 45561→45907 | 346 | 15154→15309 | 155 |
| 3 | 46064→46480 | 416 | 15263→15349 | 86 |

八池合计 **244794→246804，新增2010个池行**；这不是 2010 个全局不同键，因为训练角色跨折重复。四个不相交 held16 池拼成 full64 时是 **61781→62281，新增500键**。原 dev16 是 **13144→13246，新增102键**。BioBERT native train64 的 5903 个规范键还包括与旧池共享的键，因此不能把 5903 全部当作要删的独有候选。

只删除增强池的六列，仍会留下上述新增候选；只将六列归零而继续使用旧模型，仍会留下训练依赖。回到冻结八池可直接避免这两种残留。

## 四、训练来源与不能直接复用的产物

1. 四折入口 `public_phenotagger_biobert_selector/fit_control.py::main()`：每个外折 train48 与 held16 不相交；train48 由三个不相交内层 held16 组成，各自 semantic/CRF 候选来自其余 train32。源码逐项核 generator/input/base/CRF manifest SHA 和文献集合，公开 input-only 模型不做本赛 fit，可跨折共享其固定前向结果。
2. full64 入口 `public_phenotagger_biobert_dev_transfer/prepare_train.py::main()`：拼四个外层 held16，候选生成器各训练于互补48；仅把行 `origin_index=None` 改回原 OOF 折号，其余候选不动。`fit_full64.py::main()` 再以 train64 标签 fresh-fit 一次，dev16 文献与 training_ids 无交集。此次只静态核源码和 metadata，没有重新读取这些 gold 内容。
3. 不能将 full64 HGB 用来评价 train64 子集并称留出验证，也不能为新的 holdout 简单切旧 OOF 池；任何新划分都须沿全部生成器重查 H40。

必须从新的 NCR 实际推断链移除的明确路径包括：

- `ontology_rules/public-biobert-selector-control/formal-control/fold*/fitted/model-vectorizer.pkl` 及其旧 `selector-scores.jsonl`；它们已学到 BioBERT 候选和六字段。
- `ontology_rules/public-biobert-dev-transfer/fitted/model-vectorizer.pkl`，SHA `42e941150d1974993ea403ad41af6bfbabe410fe334f1c90c50a895d0a44c9a7`，实际103列；以及 `prepared-train/train.pool.jsonl.gz`、`predicted/input.pool.jsonl.gz`、`predicted/selector-scores.jsonl`。
- `research/public-phenotagger-biobert-{train64,dev16}-cache/…` 中的模型 posterior、canonical/provenance 和基于其输出的新增候选；旧权重 `biobert_PT_v1.2.h5` SHA `1c4d799c…15d7573` 是121587826参数依赖。旧产物保留作历史证据，不删除文件，只从新生产依赖退出。
- 前两模型旧 HGB 本身也不能作为新 NCR HGB 继续用：候选集和特征语义已改变，仍须 fresh `DictVectorizer.fit_transform` + fresh HGB；可复用的是其冻结候选和训练来源。

原 dev 两模型的 `sequence_crf/public-bioformer-dev-transfer/fitted/selector-scores.jsonl`（SHA `c032ca68…59fd0e`）可仅作为旧候选容器：`predict_dev.py` 的做法是先去掉 `selector_probability`，保留每个旧 candidate 字段，再合并新来源。旧 `selector_probability` 不得成为 NCR 新分数；候选自带的原 semantic `probability` 仍按冻结旧字段保留。

## 五、以 NCR 替代的最小完整做法（建议，尚未实施）

- 从八个前两模型池出发；只用未来完整、独立生成的 NCR input-only native cache，按原生 1–7 词、严格 `>.8`、10 ensemble、原 offset/重叠政策及明确的当前 HPO 映射形成真实输出。不把683438 raw坐标当已预测候选。
- 用精确 doc+offset+length+规范 HPO 合并，新命名空间 `ncr_*`。共享键保留原 base/semantic/CRF、`public_*`、`bioformer_*` 全字段；新键使用明确的缺失来源默认语义。
- NCR 只有神经分支，不能照搬 BioBERT 的 17266 列 head mapping、244938 forward cache、5dp `>.95` gate 或 dictionary/abbreviation 标签。新特征可按真实 NCR 输出冻结设计；若保留同形六字段，dictionary/abbreviation=false 只能表示流程确实不存在，不能伪造旧 PhenoTagger provenance。此报告不预先决定最终新列数。
- 从新池重新构造原四折 HGB 特征与标签，fresh vectorizer/HGB，实际保存 feature_names，断言不含任何 `biobert_*`。以后如获准原 dev，再拼新四 held16 池 fresh full64，独立生成新 dev NCR cache并计算新分数。参数预算只有在实际模型、候选、特征和缓存依赖均移除后才能扣 BioBERT。

本次只追了 BioBERT 实体 cache/池/六字段/HGB/full64-dev 迁移链。其 `finalize_dev.py` 原样保留基线 nonentities/relations、没有重算关系支持；主线程后来关系/LLM分支若曾消费 BioBERT 产生的实体或 support 候选，还需另行核实际依赖。本报告不能单独证明整条最新生产管线已完全去除 BioBERT。
