# 公开Sap真实特征控制

入口按顺序为 `prepare_request.py`、`transport.py` 的 prepare/encode/cosine、`prepare_pools.py`、`run_control.py`、`calibrate_cut.py`。路径与SHA见 evidence/round2/public-sap-feature-{plan,formal-plan,cut-plan,results}.json；完整外部根目录为 `/workspace/patientphex-runs/round2-20261023/ontology_rules/public-sap-feature-control`。

固定同top3全池，仅加 `public_sap_cosine`/`public_sap_observed`；单ID取公共name/EXACT别名max cosine，`;` compound取各原子max的min；缺失显式标记。旧字段/键/顺序全保留，无新ID/跨度/任务训练别名。`numeric_audit_sample.py` 只提取固定小数值证据，无模型或质量调用。

四fold同严格nested来源/HGB160，实际217.670秒，原240秒限未改；事前九cut和tie规则完整36预测冻结后再读质量。结果未超原best，已关闭本配方，不自动重复运行。最终向量留远端持久项目，完整池/概率/预测留外部；未序列化HGB，必要时须以实际源/数据/计划进行明确标记的重建。
