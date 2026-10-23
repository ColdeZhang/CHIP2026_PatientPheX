# 公开 PhenoTagger 原 dev16 无标签缓存归档

本目录保存研究 helper 已完成的原 dev16 输入缓存任务的 7 个稳定源码/配置文件，全部与外部原件逐字节相同。方案 owner 只负责 Git 保存；本次归档没有训练、推断、读取原 dev 输入或 gold，也没有重跑缓存。

- `run_dev16.py`：沿固定公开 PubMed PhenoTagger、原 5 位概率舍入后严格 `>.95`、词典与缩写解码政策，复用相同 lower-token 前向缓存；不读取质量标签。
- `make_provenance.py`：整理实际 neural/dictionary/abbreviation 来源，保持 native canonical 字节。
- `launch_cache.py`：原有界 GPU/CPU 启动入口的冻结副本。
- `collect_and_verify.py`：已执行的回收及输入契约检查器，保存以便恢复，不需要再次执行。
- `plan.json`、`input-manifest.json`、`source-equivalence.json`：冻结策略、输入指纹和三核心块等价证据。输入 manifest 仅含标识、长度/坐标与哈希，没有全文或患者字段值。

[来源索引](source-index.json) 记录原路径、实际 SHA-256、大小和外部终点报告指针。外部 completion 报告 16 篇、1,402 个 canonical 候选，原任务完成且 GPU 已释放；它属于缓存实施自核，不代表抽取质量或独立质量审查。输入、完整预测、raw、权重及日志保留仓库外。

本任务与 train64 的 `.95`/`.5` 候选可行性研究分开。后续选择器质量以其独立实验记录为准。源码依赖原远端资源目录，不应在此归档目录直接运行；完整恢复位置见索引和计划。上游工具 MIT 许可见 [PHENOTAGGER-LICENSE](../PHENOTAGGER-LICENSE)。
