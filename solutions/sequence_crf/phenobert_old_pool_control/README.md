# 原候选池的公开 pair 三列控制

本目录复用已审旧池 Sap 控制的标签、HGB160、.3 greedy 与九cut实现；只把新增字段改为已审PhenoBERT原子三类均值。原Bioformer候选/旧字段/顺序原样，不含top3新增ID与六rank字段、不含Sap新cos、不增加固定ID臂。

完整计划和来源见 `../evidence/round2/phenobert-old-pool-plan.json`，外存 `sequence_crf/phenobert-old-pool-control/`。`prepare_plans.py` 依次生成prepare/fit/calibration三阶段实际引用；旧依赖先核SHA再保存不可变快照。`prepare_pools.py` 复用相邻 `phenobert_pair_control/features.py` 和已审135000公共缓存；不跑NN。运行实际命令、输出和预算写正式run-state；无条件重跑会被已有目录/文件guard拒绝。
