# 公开模型训练集 top3 查表

目标：给既有严格 OOF 候选池提供可追溯的模型概率特征，先验证接口，不产生新正式预测或质量指标。

输入是 ontology owner 从四个 held 池及各自三个 inner OOF 池取得的全部 61,293 个跨度，只有 `pmc_id/offset/length`。不使用之前 298 个错误条件查询。两个公开模型各自的 train64 raw 缓存和完整列映射均按 SHA 冻结；无 gold、originaldev、权重读取、NN、GPU 或 fit。

运行 `prepare_plan.py <whitelist> <SHA>` 冻结纯输入和来源，再由 `deploy_and_start.py` 部署到独立远端目录并执行一次 `timeout --signal=TERM --kill-after=10s 120s ... extract_lookup.py`。作业退出后运行 `collect_and_check.py`，分块回收 gzip 并检查完整 SHA、全部查询键、原列映射和聚合数值。实际 job、预算和结果由 `job.json`、`output/summary.json`、`completion.json` 记录。

输出每个跨度恰好一行；`pubmed` 和 `bioformer` 为独立列表，包含 `hpo_id,p,rank,observed,observations`。`p` 是同一当前 HPO 下实际观察原列的最大概率，`rank` 是原 top3 的最小排名，原始 `(column,raw_hpo_id,p,rank)` 均保留。不是完整 canonical 后验，不跨模型取 max，不加阈值。无 raw 匹配时列表为空且 `*_raw_present=false`；有 raw 但全部当前无效时也可能空列表，二者明确区分。

提取器逐字节核查三个原始大文件，各模型 296,984 行，原文坐标、完整列序、实际概率降序和两模型原始跨度集合一致性。大 lookup/raw 仅保存在持久产物目录；Git 由方案 owner 收编小源码、冻结计划及完成索引。
