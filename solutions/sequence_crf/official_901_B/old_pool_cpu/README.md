# 冻结 old-pool CPU 推断

目标：为当前 `9016863c` 方案的真实 B 推断恢复原 train64 CRF/Sap 底座。此目录所有学习模型只加载、不 fit、不操作官方平台。发现未保存的计数表后，父线程/root 明确授权只从原 train64/HPO 按原算法恢复 Gazetteer 与 Sap 词表计数；这两步真实读取 train64 标注，单独记录，不冒称全程未读训练数据。没有读取 dev gold 或评测。父线程统一 Git 和上传。

`load_predict.py` 固定加载 `8753c5f2` pipeline 和 `ee0f34ba` CRFsuite model；显式重绑模型路径。完整 CRF 预测与 confidence 分开保存，外来 Sap 原始候选保持原文件和指纹。选后 old-pool 使用原 `.65/.92`、原非重叠策略和保存的修正版 LR `.55`。

必须先在 `dev16.input.jsonl` 与旧预测缓存上验证：旧 day1 CRF `b7bcdeab` 实体精确相同，CRF+Sap `55f4305e` 完整 JSON 和文件字节精确相同。比较预测缓存不读取 gold，也不做质量评分。

已发现来源差异：day1 train64 原 pipeline `1a2c2578` 和后来的 `8753c5f2` 使用同一 `ee0f34ba` CRF 模型；前者关系 LR 正例4547，后者修正版4606。旧 `55f4305e` 的 augmentation 本就使用修正版源码重新拟合关系 LR；本入口直接加载后者保存的 LR，以完整回放证据确认。

原始缺口事实保留：规则 Gazetteer 当时没有完整序列化；Sap ordered bank labels/IDs、detector 已保存，但候选 priority 所需 `positive_heads` 未找到完整冻结对象。经上述明确授权，两个原表已经精确恢复，未使用 day1 learned80 资源。

CPU 阶段已完成，完整指纹见 [cpu-stage-summary.json](cpu-stage-summary.json)。外部输出根为 `/workspace/patientphex-runs/round2-20261023/research/official-901-B/old-pool-cpu/`。

| 产物 | 完成证据 |
|---|---|
| `dev-replay/old-pool.pred.jsonl` | 旧 CRF+Sap 完整 bytes SHA `55f4305e…`，16 篇实体及关系精确重现 |
| `B-crf/crf.pred.jsonl` | 100 篇/244 患者/5229 实体；SHA `f2cf44cb…`，所有实体原文坐标和患者覆盖检查通过 |
| `B-crf/crf.confidence.jsonl` | 5229 实体独立保留 `ner_probability/link_similarity`，SHA `bba518d6…` |
| `rules-table/gazetteer-state.pkl` | 原 train64 计数恢复后持久化并重新加载；dev 完整 bytes `5a994261…` 精确重现 |
| `rules-table/B.base.jsonl` | 100 篇/244 患者/5728 实体；SHA `b15bf7c4…` |
| `sap-table/sap64-lexical-state.json` | 原源码确定性计数；42575 bank labels/IDs 与既有文件逐项一致 |
| `sap-table/dev.raw-candidates.jsonl` | 12811 候选全部原字段与旧缓存逐对象一致，包括 priority/exact/passage |
| `sap-table/B.raw-candidates.jsonl` | 100 篇/90303 候选；SHA `78b90aaf…`，原字段全部保留 |
| `sap-table/B.query-requests.jsonl` | 每篇 unique 查询合计65673；SHA `a477c250…`，原文/顺序映射/精确ID全部核对 |

`restore_rule_table.py` 使用原 `56cdcb45` Gazetteer 构造算法，恢复41834词条/2279训练别名/1 blocked项；只在缓存预测精确通过后生成 B。`restore_sap_table.py` 仅恢复原 `b1005687` 的计数块，显式禁用 `System.__init__`、`fit_detector`、`Encoder.__init__`，不编码 bank、不拟合 detector。两脚本只运行一次；之后直接加载保存的计数状态。CRF/关系入口使用已有 `sequence-crf/.venv/bin/python`；Sap 词表恢复使用已有 `semantic-neural/venv/bin/python`（其中有 ahocorasick），没有安装新依赖。

GPU 交接请求每行为 `{pmc_id, texts, candidate_to_unique, exact_identifiers}`；`texts` 是该篇 `sorted(set(norm(candidate.text)))`，其余两列与原始候选逐项对齐。加载原 bank `4fc0e062…` 和 detector `adb8ff8c…`，仅编码新 B query；原 batch256/max_length32/FP16 backbone、FP32 CLS/L2 和 GPU dot 不变。所有完整模型、cache 与源码指纹在 Sap manifest；远端 fresh 存在性由父线程核验。

`assemble_frozen.py` 接收已有实体文件及可选 `--sap-candidates`，使用同一保存的 corrected LR 输出完整所有患者的 association。省略 Sap 参数就是 GLiNER 实体追加后的 relation-only CLI；已用旧55f输入验证输出仍逐字节 `55f4305e…`。该 CLI 不执行 NER 或任何 fit。GLiNER48 只用于 Task2，由父线程另行追加；共同规则/CRF/Sap 原始候选供 Task1 使用。

当前剩余：父线程完成 Sap B query 编码/打分、GLiNER48 及后续最高方案合成。本目录尚未声称完整 B old-pool 或官方提交已完成。源码编译、diff 空白检查、所有 B 实体/候选原文与查询映射检查通过；没有额外质量评分。
