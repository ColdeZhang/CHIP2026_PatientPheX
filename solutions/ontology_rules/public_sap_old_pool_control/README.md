# 原Bioformer旧池加两个公共Sap特征

此目录独立于 `public_sap_feature_control`，仅使用原Bioformer旧池，无top3新增候选/六raw特征。`prepare_pools.py` 复用已核公共lookup；`run_control.py` 四fold共享120秒，`calibrate_cut.py` 预定九点共享30秒、36完整预测冻结后评分。

正式计划、终点与恢复位置见 `evidence/round2/public-sap-old-pool-{formal-plan,cut-plan,results}.json`；外存 `/workspace/patientphex-runs/round2-20261023/ontology_rules/public-sap-old-pool-control`。HGB未序列化，完整源码/数据/全概率/预测均保留。此控制已负结案，不自动重复运行、不原dev。
