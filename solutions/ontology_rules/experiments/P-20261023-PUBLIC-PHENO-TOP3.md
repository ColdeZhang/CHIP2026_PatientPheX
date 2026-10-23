# 公开 PhenoTagger 已缓存 top3 / 字典多 ID 供给诊断

主线程授权的训练侧低成本上界：在既有 298 个“跨度已存在但缺正确 literal ID”的 FN 原子（287 个跨度）中，检查 PubMed/Bioformer 固定 raw top3 及原生字典多 ID 是否包含目标。仅研究是否值得随后统一扩展 ID 候选；这些 gold 查询不构成正式候选策略，不输出新预测/指标，不 fit、不读 originaldev、不加载权重或使用 GPU。

先远端实查三个 raw 文件可读（总约 368 MB），各 1,000 行解析约 0.012–0.014 秒，两个 17,266 列 label 文件相同，固定 head→current 和 dictionary→current 映射 SHA 均通过。只在远端扫描原缓存，不回传大 raw；CPU 硬限 120 秒。全文输入严格是已投影的 train64 四字段无标签副本，唯一诊断目标来自已冻结 `.3` 与旧候选池。

固定 top1/top2/top3、两模型分别/并集；不另选 posterior 门槛。映射仅作用于预测资源，gold 按 v2 原样 ID，复合原子分母不变。原 raw 不含的跨度单列，通过既有 native tokenizer/坐标区分 unigram、2–8gram 被过滤、超过上限及非词边界，不把它们称为 top3 分类失败。按原生字典实际匹配重放 first-ID 并与缓存一致后，检查其固定多 ID 列表中的后续 ID，区分“首 ID 已正确”与“仅后续 ID 正确”。

[冻结计划](../evidence/round2/public-phenotagger-top3-plan.json)记录所有源、缓存/映射、查询 SHA、预算和恢复路径。原生 hybrid/缩写/否定/greedy 未用新 ID 重解码；此处只是 raw 供给，不能称可实现的分数或独立泛化结果。已有放宽跨度候选后 1,697 个真正新 pool 键 0 接纳的负结果提示：即使新增 ID 有覆盖，也需另查 selector 是否具备区分它们的特征，不能直接扩大正式池。

状态：来源和计划已冻结，未运行正式诊断。
