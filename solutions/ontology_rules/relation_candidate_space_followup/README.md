# F0/F1 关系候选空间：来源盘点

当前仅完成来源、原始语义和计数定义冻结，**未读取 gold、未解析候选/提示/预测正文、未进行新空间计数或评分**。文件 SHA/大小均已实际读取核验。`input-source-inventory.json` 是最初盘点，随后 root 给定关系端点保存在独立 `relationship-baseline-addendum.json`；后继使用 `count-definitions-revision1.json`，不覆盖历史定义。

真实问题是：当前 LGB 选出的文档 HPO 是否存在于既定 oldpool∪public2 的合法关系提示空间。合法空间取实际 prompts 的 `code_map × patient_ids`，不能取模型预测为正的关系集合。Task1 的 `lightgbm.nested.jsonl` 只读 entities；其 association 是早期基线原字节副本。

来源已锁定：F0 Task1 79d3568e，旧提示 d387f5b8，public 提示 ffcaae74；F1 Task1 8cb61ba9，旧提示 ddba03ff，public 提示 bfe62979。完整 SHA、路径、候选输入池与三public canonical/provenance 均见 inventory。root 后续明确的 FN 对照为 public-new LGB support 两个既有端点：F0 94504b97、F1 2a65056f；原 metrics 仅复用。它们不是完整901管线在两折上的新合成。

已查最接近历史：F1 public2 相对 oldpool 的输入差集及 oracle 覆盖、F0 public2 差集、两折 LGB support gate 替换。它们分别比较旧供给或筛选已生成正关系；在这些源、冻结文件及路线笔记中未发现与“当前LGB-selected缺空间 + BioBERT来源归因 + 真实FN”完全等价的控制。此结论限定于已查范围，不宣称全仓所有历史均不存在。

两个关键边界：

- 这两折既有 public prompt 源直接接收全部 native canonical 行，只核 native neural policy 为 strict >.95；没有第二次 `hybrid_score > .95` 过滤。低分只能作为来源观察，不能预设成漏空间原因。原dictionary/abbreviation行保持。
- F1 Task1 LGB使用 train48，旧关系候选及SFT32使用train32；它们均按原来源使用，覆盖差异不能归为BioBERT单因素。BioBERT-present的某个mention也不等于文档HPO仅由BioBERT供给。

后续由owner统一冻结一次诊断入口：先核typed raw/compound/alt/当前HPO及患者空间等价、冻结全部缺空间分类，再读取授权的F0/F1 gold汇总linked FN机会。保留raw v2和canonical诊断两种口径，不能把oracle机会称为可恢复TP、预期涨分或官方评测。该任务不修改本次B输出。
