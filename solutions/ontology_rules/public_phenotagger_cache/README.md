# 公开 PhenoTagger train64 无标签缓存入口

本目录保存研究helper已冻结并实际部署的源码/小计划原字节。ontology为Git归档owner，helper负责本次剩余48篇唯一native推断和64篇缓存合并；不要在本地重新执行已有prepare_input或覆盖远端产物。

- `prepare_input.py`：原train64投影为pmc_id/pmid/patient/full_text；16篇旧输入逐对象相同，余48严格互斥。实际input64/48、byte-proof留外部。
- `run_remaining48.py`：同公开native模型/词典/候选修复/5dp>.95/abbr协议，复用既有相同lower-token前向缓存；不因已见nested质量结果改变策略。
- `unify_cache64.py`：旧16完整canonical原行保留，与余48按input64顺序合并；逐行坐标核对，额外sidecar记录实际neural/dict/abbr来源，不从概率猜来源。
- `build_cache_driver.py`：构建/冻结研究侧入口的生成脚本，只作恢复依据，不是重复推断入口。
- `plan.json`：实际资源/源码/公共权重SHA、600秒GPU+TERM30、CPU准备180秒/union120秒。真实managed job/终点留外部研究目录；本目录不表示任务已完成。

原研究资源根 `/workspace/patientphex-runs/round2-20261023/research/public-phenotagger-train64-cache/`；远端部署为公开模型持久目录的 `train64-cache/`，源码使用其既有cwd资源布局。`source-index.json`给出Git文件与实际原路径/完整SHA对应；固定model/dictionary资源和权重留仓库外。没有原dev输入/质量标签传入native模型。

上游工具MIT许可见 [PHENOTAGGER-LICENSE](../PHENOTAGGER-LICENSE)。当前目标是一次统一输入缓存供本方案固定outer0迁移及sequence严格OOF学习，各实验结果分别记录，缓存自身不产生质量分数。
