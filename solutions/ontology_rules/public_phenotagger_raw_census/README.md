# 原始跨度全输入候选普查

本任务只统计供给规模和已有错误的实际流失阶段，不产生新候选文件、训练、预测或质量分数。

事前固定三项机械政策，遍历无标签 train64 的全部原 raw 坐标：A 两模型原 top1 且非 None/current-valid；B 原 top1..3 全部 current-valid；C 原 top1/current-valid 且沿原 5 位舍入后严格 `>.95`，绕过 native hybrid/abbr。C 没有新增概率门槛，用于区分原合并消解损失。三项都逐原 fold 与所有旧键取并集，仅计算新增跨度、ID 与三 nested 输入块/完整池的倍率。

`prepare_input.py` 只投影旧池键与所属 pool 位图，不读标签/score；实际旧 244794 行合为70615个唯一键，投影仅421539B。启动前构造父目录的普通文件操作错误已修复，首次没有读数据或生成实验输出。

`census.py` 先完成全输入计数并保存，再读前次冻结的217个诊断原子，按实际 raw top1 None、原门槛、native hybrid、abbr、current-ID 映射顺序分层。阶段诊断不参与全输入集合构造。实际已有 before/after 缓存用于确认阶段，不重跑NN；仅在 hybrid 阶段定位，不反推具体竞争获胜项。

唯一 remote CPU 运行限120秒＋TERM5秒，完整raw SHA核对，输出小报告。所有原缓存留原远端，projection和逐项审计留外部 `ontology_rules/public-phenotagger-raw-census/`。入口为 `remote_io.py deploy-start/status/collect --root <外部目录> --client-dir <主库scripts>`。行数线性耗时估计只是资源代理，不是正式训练结果或训练授权。
