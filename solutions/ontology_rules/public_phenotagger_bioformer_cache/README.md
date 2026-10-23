# 公开 PhenoTagger Bioformer train64 无标签缓存归档

本目录保存研究 helper 已完成的 Bioformer train64 缓存任务的 7 个稳定源码/配置文件，全部与外部原件逐字节相同。方案 owner 只负责 Git 保存；本次归档没有训练、推断、读取输入或 gold，也没有重跑缓存。

- `run_train64.py`：独立 Bioformer 公共权重、配置与 tokenizer；固定原 5 位概率舍入后严格 `>.95`、词典与缩写解码，244,938 个独特输入全部使用 Bioformer 前向，不借用 PubMed 模型 logits。
- `make_provenance.py`：整理实际 neural/dictionary/abbreviation 来源，保持 native canonical 字节。
- `launch_cache.py`：原有界 GPU/CPU 启动入口的冻结副本。
- `collect_and_verify.py`：已执行的回收及输入契约检查器，保存以便恢复，不需要再次执行。
- `plan.json`、`input-manifest.json`、`source-equivalence.json`：冻结策略、输入指纹和与原 native 解码/舍入代码的等价证据；输入 manifest 不含全文。

[来源索引](source-index.json) 记录原路径、实际 SHA-256、大小和外部终点报告指针。外部 completion 报告 64 篇、5,869 个 canonical 候选，原任务完成且 GPU 已释放；这是缓存实施自核，没有质量评分，也不代表独立质量审查。输入、完整预测、raw、权重及日志留仓库外。

实际模型核验范围是：CPU 对全部 137 个变量做 NaN/逐数组等值核验；正式运行原生拓扑加载和数量核对，并对真实首批 GPU 输出执行设备、有限值、概率和及类别数检查。没有单独进行 GPU 全部权重数组的等值重放。源码保留旧通用注释的原字节；恢复与判断核验覆盖范围时，以外部实际 CPU 报告、GPU 首批报告及 completion 的明确说明为准。

本模型来源与 PubMed 公共模型分别标识，不能将其 logits 或 manifest 冒充 PubMed 来源。后续固定四 fold 选择器质量以独立实验记录为准。源码依赖原远端资源目录，不应在此归档目录直接运行；完整恢复位置见索引和计划。上游工具 MIT 许可见 [PHENOTAGGER-LICENSE](../PHENOTAGGER-LICENSE)。
