# Sap64 新查询打分器准备

这是独立新增入口；前一阶段已经完成的六件源码/说明保持原字节。`encode_sap_queries.py` 没有执行 GPU、网络或训练；主线程控制远端部署、预算和启动。完整参数草稿及指纹见 [sap-query-scorer-preparation.json](sap-query-scorer-preparation.json)。

入口先核原语义源码 `b1005687`、模型五件及 manifest、42575 行 bank `4fc0e062`、标签映射 `a369ec3d`、detector `adb8ff8c` 和所有输入 SHA。逐篇全部候选坐标、文本、排序去重和索引映射先校验，再加载编码器。不会调用 `System` 构造器或任何 `fit`，不会重编码 bank。

GPU 路径调用原 `Encoder`：batch256、max_length32、FP16 backbone、FP32 CLS/L2。先编码逐篇 `sorted(set(norm(text)))`，然后映射回原候选次序，再对完整候选矩阵调用保存的 LR `predict_proba` 和原 `nearest`，保留原 GPU dot 的512候选分块。不能先对 unique 向量打分再映射，否则矩阵分块会改变。TF32 等实际环境值原样记录，不擅自切换。

已有查询缓存必须与原 reconstruction manifest 中 SHA 相同，或具有本入口新建的 source/signature/SHA 回执。每篇记录 `fresh-encoding` 或 `verified-cache`。每篇完整分数先原子写盘，再执行比较；最终完整 JSONL 与确定性 gzip 再冻结。异常保存 `failure.json` 及当前篇位置，保留已写文件，不补空篇、不截断、不降 batch、不自动重试。`--output-dir` 必须是尚不存在的新目录。

`--mode cpu-cache` 隐藏 CUDA，不实例化 `Encoder`，仅加载旧 embedding、bank 和 detector。它要求原 reconstruction manifest 和参考 scored 文件，比较原候选/ID完全相同、概率误差≤1e-8、相似度误差≤2e-6，以及 `.65/.92` 两阈值决策零差异。容差沿用已有 Sap64 reconstruction 合同，没有新质量评测。

本地实际完成的 CPU 检查：100篇90303候选映射和原文全通过；错误索引反例被拒绝；保存 detector 的 sklearn `predict_proba` 与 float64 logistic 公式在9条固定合成向量上逐位相同；`fit` 被显式拒绝。已生成16篇12811候选的 dev 查询请求，供远端 cache-only 回放。**本地缺少原 bank 和 query.npy，真实 dev 向量回放尚未执行。** 检查证据在外部 `old-pool-cpu/scorer-preparation/local-cpu-checks.json`。

B 准备输入为100篇、90303候选、每篇 unique 合计65673、308个编码 batch、231个 dot block；单篇最大2975个 unique/4414候选。预算由主线程依据此实际数量确定。本入口没有默认超时或自行授权启动。

GLiNER 后续可复用相同入口：raw 行支持 `candidates` 或 `entities`，原 `score/offset/length/text` 及其他字段都保留；`origin` 仅进运行元数据。准备请求时遵守旧 `link_gliner_candidates.py`：norm(text) 在 frozen bank labels 内则填固定 `exact_identifiers`，否则 null；原 `ontology_exact` 由准备阶段保留在 raw。新 scorer 只追加 `identifier/probability/similarity`，不根据 origin 改算法。不得传入已含这些分数字段的旧打分结果。

计划中的完整 argv 是草稿，远端目录可由父线程统一改写；输入 SHA、源/模型/算法不变。成功终点必须以实际 `manifest.json` 和完整输出 SHA 为准，当前准备完成不表示 B 分数已生成。
