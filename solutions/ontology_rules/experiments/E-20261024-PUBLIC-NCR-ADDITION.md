# E-20261024-PUBLIC-NCR-ADDITION：强BioBERT基底增加NCR

真实目标：新增公开NCR的候选/置信度能否改善现有强BioBERT实体选择。主对照为既有固定四fold Score .730993616，pre-B .727975656仅作归因；当前尚无本实验质量结果。

官方原PDF §3.1（页4）限制每个individual模型≤10B，§4.2（页10）明确两个8B模型组合。独审证据 `/workspace/patientphex-runs/round2-20261023/validation/model-parameter-rule-correction/review.json` SHA5bd082d3；原按全链总和强制去B是过严项目假设，尚未进行任何fresh拟合，现已撤销。旧源/计划与实际参数库存保留，不能将此纠正写成某模型质量负结果。NCR全部资源、CPU/GPU验证和纯输入缓存继续有效，不重NN。

方法预先固定：保留强B八池（246804行）及六字段/候选/provenance，NCR原生1–7词、strict>.8和原重叠后按固定公共当前HPO规则映射；新增仅ncr_present/ncr_score，score为实际原float32当前exact-key最大值，保留raw来源位模式。共享候选原前缀字段值/类型/顺序保持，原B来源不得清除。

四fold保持原文献48/16、训练候选内层32→16来源，只有固定公共NCR输入cache可跨fold共享；新增下游模型不放宽H40来源隔离。fresh vectorizer/HGB160树/15叶、原完整参数/seed/early_stopping=False、固定.3及原nested，实际F/树state/模型SHA运行后保存。900秒+TERM30是整个正式窗口，包含4fit/infer与单一终点；数据/map/pool准备独立在此前完成。各fold完整held预测先共同冻结，再固定v2 all/text及格式核验；不生成训练posterior，不扫cut，不自动迁原dev或官方。

比较预定：报告强B与NCR的pooled Score/menF1/docF1及逐fold差，pre-B仅辅助；保留方向需强B实体两F1均值严格改善且至少3/4fold非降。原dev≥.8仍是最终项目目标，不用内部控制冒充达标。

状态：完整NCR缓存CPU最终核验通过（3553e219）；新源/计划准备中，无actual当前映射/新池/拟合/评分。回传连接异常不改变远端完整缓存事实；优先取raw spans与head，远端全量SHA/5341绑定证明支持输入准备，本地完整NPY不是额外门槛。原失败/恢复见 [缓存记录](../public_neuralcr_readiness/full-train64-cache-attempt2/README.md)。
